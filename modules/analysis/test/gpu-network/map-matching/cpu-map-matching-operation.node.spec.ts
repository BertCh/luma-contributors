// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMapMatchingOnCPU,
  CPUMapMatchingOperation,
  type CPUMapMatchingInput
} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {GPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  createCutoffMatchingFixture,
  createMatchingFixture,
  getGeographicSnapComparison,
  getMatchingGraphInputs
} from './map-matching-operation-fixture';

const NONE = GPU_MAP_MATCHING_NONE;

it('retains the finite score cutoff: far-away candidates remain unmatched with zero likelihood', () => {
  const {input, props} = createCutoffMatchingFixture();
  // Distance 1e9 / sigma 1e-6 gives an emission about -5e29, finite but below -1e29.
  // Neither point can be picked. The first point is never a break; the second is unmatched.
  const result = computeMapMatchingOnCPU(input, props);
  expect(Array.from(result.matchedEdges)).toEqual([NONE, NONE]);
  expect(Array.from(result.breaks)).toEqual([0, 1]);
  expect(Array.from(result.trackLogLikelihoods)).toEqual([0]);
  for (const values of [result.matchedFractions, result.matchedOffsets, result.snapDistances])
    expect(Array.from(values)).toEqual([-1, -1]);
  expect(result.snappedPositions).toEqual(input.positions);
  expect(result.matchedCount[0]).toBe(0);
  expect(result.breakCount[0]).toBe(1);
  expect(Array.from(result.traversedCounts)).toEqual([0]);
});

it('preserves the inherited traversal requirement layout at capacity one', () => {
  const {input, props} = createMatchingFixture();
  const result = computeMapMatchingOnCPU(input, {...props, traversedEdgeCapacity: 1});
  // Routes are [0,1,2] and [3,4,5]. Only their first entries fit, while counts retains need.
  // This pins the contributor's existing section 8 departure pending a coordinated API change.
  expect(Array.from(result.traversedEdges)).toEqual([0, 3]);
  expect(Array.from(result.traversedCounts)).toEqual([3, 3]);
  expect(result.traversedOverflow[0]).toBe(1);
});

it('ignores traversed capacity when no traversed output is requested', () => {
  const {input, props} = createMatchingFixture();
  const shapes = Object.fromEntries(
    Object.entries(getMatchingGraphInputs(input)).map(([port, value]) => [
      port,
      {
        format: value.format,
        length: value.values.length / (value.format === 'float32x2' ? 2 : 1)
      }
    ])
  );
  for (const capacity of [0, -1]) {
    const minimalProps = {
      ...props,
      output: {matchedEdges: 'matchedEdges'},
      traversedEdgeCapacity: capacity
    };
    expect(() => new CPUMapMatchingOperation(minimalProps)).not.toThrow();
    expect(() =>
      constructNetworkOperationContributor(new GPUMapMatchingOperation(minimalProps), shapes)
    ).not.toThrow();
  }
  expect(
    Array.from(computeMapMatchingOnCPU(input, {...props, traversedEdgeCapacity: 0}).matchedEdges)
  ).toEqual([0, 0, 2, 3, 5]);
  expect(() => computeMapMatchingOnCPU(input, {...props, traversedEdgeCapacity: -1})).toThrow(
    /traversedEdgeCapacity/
  );
  expect(() => new CPUMapMatchingOperation({...props, traversedEdgeCapacity: 0})).toThrow(
    /traversedEdgeCapacity/
  );
});

it('rejects the review longitude perturbation with a meter-scale coordinate bound', () => {
  const {input, props} = createMatchingFixture();
  input.topology = {
    nodePositions: new Float32Array([179.97, 0, 179.99, 0]),
    offsets: new Uint32Array([0, 1, 1]),
    neighbors: new Uint32Array([1])
  };
  input.positions = new Float32Array([179.975, 0.001]);
  input.trackOffsets = new Uint32Array([0, 1]);
  input.parameters = getGPUMapMatchingParameterValues({sigma: 0.2, beta: 0.5, searchRadius: 0.4});
  const context = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'kilometers'
  } as const;
  const result = computeMapMatchingOnCPU(input, {
    ...props,
    spatialContext: context,
    cellSize: 0.02
  });
  // Equatorial horizontal segment: closest point has the fix's longitude and latitude zero.
  const expected = [Math.fround(179.975), 0];
  expect(Array.from(result.snappedPositions)).toEqual(expected);
  const wrong = getGeographicSnapComparison(
    [179.98300170898438, 0],
    expected,
    input.topology,
    context
  );
  expect(wrong.distance).toBeGreaterThan(0.8); // More than 800 m, while the radius is only 400 m.
  expect(wrong.tolerance).toBeLessThan(0.004); // Less than four meters at the equator.
  expect(wrong.distance).toBeGreaterThan(wrong.tolerance);
  const oneUlp = getGeographicSnapComparison(
    [expected[0] + 2 ** -16, 0],
    expected,
    input.topology,
    context
  );
  expect(oneUlp.distance).toBeLessThanOrEqual(oneUlp.tolerance);
});

/** Independent oracle: Floyd-Warshall network distances and exhaustive candidate sequences.
 * This has no shortest-path queue, dynamic programming, backpointers or score renormalization.
 * It is restricted to these small planar fixtures with no breaks and no fix filtering.
 */
function enumeratePaths(input: CPUMapMatchingInput, candidateCount: number) {
  const {offsets, neighbors, nodePositions} = input.topology;
  const nodeCount = offsets.length - 1;
  const sources: number[] = [];
  const lengths: number[] = [];
  const distances = Array.from({length: nodeCount}, (_, source) =>
    Array.from({length: nodeCount}, (_, target) => (source === target ? 0 : Infinity))
  );
  for (let source = 0; source < nodeCount; source++) {
    for (let edge = offsets[source]; edge < offsets[source + 1]; edge++) {
      sources[edge] = source;
      const target = neighbors[edge];
      lengths[edge] = Math.hypot(
        nodePositions[target * 2] - nodePositions[source * 2],
        nodePositions[target * 2 + 1] - nodePositions[source * 2 + 1]
      );
      distances[source][target] = Math.min(distances[source][target], lengths[edge]);
    }
  }
  for (let middle = 0; middle < nodeCount; middle++)
    for (let source = 0; source < nodeCount; source++)
      for (let target = 0; target < nodeCount; target++)
        distances[source][target] = Math.min(
          distances[source][target],
          distances[source][middle] + distances[middle][target]
        );
  const [sigma, beta, radius, factor, slack] = input.parameters;
  const candidates = Array.from({length: input.positions.length / 2}, (_, point) => {
    const found = lengths
      .map((length, edge) => {
        const source = sources[edge];
        const target = neighbors[edge];
        // All fixture edges are horizontal and directed east. Projection is a clamped x coordinate.
        const snapped = Math.max(
          nodePositions[source * 2],
          Math.min(nodePositions[target * 2], input.positions[point * 2])
        );
        return {
          edge,
          fraction: (snapped - nodePositions[source * 2]) / length,
          distance: Math.hypot(
            input.positions[point * 2] - snapped,
            input.positions[point * 2 + 1] - nodePositions[source * 2 + 1]
          )
        };
      })
      .filter(candidate => candidate.distance <= radius);
    return found
      .sort((first, second) => first.distance - second.distance || first.edge - second.edge)
      .slice(0, candidateCount);
  });
  const edges: number[] = [];
  const likelihoods: number[] = [];
  for (let track = 0; track + 1 < input.trackOffsets.length; track++) {
    const first = input.trackOffsets[track];
    const last = input.trackOffsets[track + 1];
    let best = -Infinity;
    let bestPath: number[] = [];
    const visit = (point: number, path: number[], likelihood: number) => {
      if (point === last) {
        if (likelihood > best) {
          best = likelihood;
          bestPath = path;
        }
        return;
      }
      candidates[point].forEach((candidate, slot) => {
        let score =
          likelihood -
          Math.log(sigma * Math.sqrt(2 * Math.PI)) -
          candidate.distance ** 2 / (2 * sigma ** 2);
        if (point > first) {
          const previous = candidates[point - 1][path[path.length - 1]];
          const route =
            previous.edge === candidate.edge && candidate.fraction >= previous.fraction
              ? (candidate.fraction - previous.fraction) * lengths[candidate.edge]
              : (1 - previous.fraction) * lengths[previous.edge] +
                distances[neighbors[previous.edge]][sources[candidate.edge]] +
                candidate.fraction * lengths[candidate.edge];
          const straight = Math.hypot(
            input.positions[point * 2] - input.positions[point * 2 - 2],
            input.positions[point * 2 + 1] - input.positions[point * 2 - 1]
          );
          if (route > factor * straight + slack) return;
          score -= Math.log(beta) + Math.abs(straight - route) / beta;
        }
        visit(point + 1, [...path, slot], score);
      });
    };
    visit(first, [], 0);
    edges.push(...bestPath.map((slot, step) => candidates[first + step][slot].edge));
    likelihoods.push(Math.fround(best));
  }
  return {edges, likelihoods};
}

it('matches exhaustive path enumeration with Floyd-Warshall routes, on two competing roads and two tracks', () => {
  const {input, props} = createMatchingFixture();
  const expected = enumeratePaths(input, props.candidateCount!);
  const result = computeMapMatchingOnCPU(input, props);
  expect(Array.from(result.matchedEdges)).toEqual(expected.edges);
  expect(expected.edges).toEqual([0, 0, 2, 3, 5]); // Middle fix chooses the farther road.
  expect(Array.from(result.trackLogLikelihoods)).toEqual(expected.likelihoods);
  expect(Array.from(result.matchedFractions)).toEqual([
    Math.fround(0.2),
    0.5,
    Math.fround(0.8),
    Math.fround(0.2),
    Math.fround(0.8)
  ]);
  expect(Array.from(result.matchedOffsets)).toEqual([2, 5, 8, 2, 8]);
  expect(Array.from(result.snapDistances)).toEqual([0.5, 2.5, 0.5, 0.5, 1]);
  expect(Array.from(result.snappedPositions)).toEqual([2, 0, 5, 0, 28, 0, 2, 3, 28, 3]);
  expect(Array.from(result.breaks)).toEqual([0, 0, 0, 0, 0]);
  expect(Array.from(result.traversedEdges)).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);
  expect(Array.from(result.traversedCounts)).toEqual([3, 3]);
  expect(result.matchedCount[0]).toBe(5);
  for (const values of [
    result.breakCount,
    result.overflow,
    result.routeOverflow,
    result.traversedOverflow
  ])
    expect(values[0]).toBe(0);
});

it('agrees with exhaustive sequences across changing noise, transition scales and competing distances', () => {
  for (let scene = 0; scene < 32; scene++) {
    const {input, props} = createMatchingFixture();
    // Every candidate pair remains inside its road segment, and neither road can reach the other.
    // Vary each observation independently, with nonzero differences so choices are not score ties.
    for (let point = 0; point < props.pointCount; point++)
      input.positions[point * 2 + 1] =
        (((scene * (point + 3) + point * 7) % 29) + 1) / 10 + (point + 1) * 0.0073;
    input.parameters = getGPUMapMatchingParameterValues({
      sigma: 0.5 + scene / 16,
      beta: 0.75 + scene / 8,
      searchRadius: 4,
      routeFactor: 2,
      routeSlack: 5
    });
    const expected = enumeratePaths(input, props.candidateCount!);
    const actual = computeMapMatchingOnCPU(input, props);
    expect(Array.from(actual.matchedEdges), `scene ${scene}`).toEqual(expected.edges);
    expect(Array.from(actual.trackLogLikelihoods), `scene ${scene}`).toEqual(expected.likelihoods);
  }
});

it('the CPU graph callback maps every optional port and validates supplied acceleration-table lengths', async () => {
  const {input, props} = createMatchingFixture();
  // An overflowing table is a valid acceleration input; exact CPU matching ignores its contents.
  input.routeTable = {
    offsets: new Uint32Array(9),
    targets: new Uint32Array(1),
    costs: new Float32Array(1),
    costLimit: new Float32Array([100]),
    overflow: new Uint32Array([1])
  };
  props.routeTable = {
    entryCount: 1,
    offsets: 'tableOffsets',
    targets: 'tableTargets',
    costs: 'tableCosts',
    costLimit: 'tableCostLimit',
    status: {overflow: 'tableOverflow'}
  };
  const operation = new CPUMapMatchingOperation(props);
  const graphInputs = getMatchingGraphInputs(input);
  const inputs = Object.fromEntries(
    Object.entries(operation.inputs).map(([port, value]) => {
      const inputValue = graphInputs[value];
      const length = inputValue.values.length / (inputValue.format === 'float32x2' ? 2 : 1);
      return [
        port,
        {
          format: inputValue.format,
          length,
          chunks: [{values: inputValue.values, length}],
          pack: () => inputValue.values
        }
      ];
    })
  );
  const expected = computeMapMatchingOnCPU(input, props);
  const actual = await operation.execute({inputs});
  for (const [port, values] of Object.entries(expected)) expect(actual[port], port).toEqual(values);
  input.routeTable.costs = new Float32Array(2);
  expect(() => computeMapMatchingOnCPU(input, props)).toThrow(/routeTable.costs/);
  input.routeTable.costs = new Float32Array(1);
  input.routeTable.offsets = new Uint32Array(8);
  expect(() => computeMapMatchingOnCPU(input, props)).toThrow(/routeTable.offsets/);
  const minimal = new CPUMapMatchingOperation({
    ...props,
    routeTable: undefined,
    output: {matchedEdges: 'onlyEdges'}
  });
  expect(Object.keys(minimal.outputs)).toEqual(['onlyEdges']);
});

it('fix filtering, no candidate breaks, separators and capacity counts follow hand-computed chains', () => {
  const {input, props} = createMatchingFixture();
  input.positions = new Float32Array([2, 0.5, 2.5, 0.5, 28, 0.5, 15, 50, 5, 0.5]);
  input.trackOffsets = new Uint32Array([0, 5, 5]); // Second track is empty.
  const result = computeMapMatchingOnCPU(input, {
    ...props,
    minimumFixSpacingSigmas: 1,
    traversedEdgeCapacity: 2
  });
  expect(Array.from(result.matchedEdges)).toEqual([0, NONE, 2, NONE, 0]);
  expect(Array.from(result.breaks)).toEqual([0, 0, 0, 1, 1]);
  expect(Array.from(result.matchedFractions)).toEqual([
    Math.fround(0.2),
    -1,
    Math.fround(0.8),
    -1,
    0.5
  ]);
  expect(Array.from(result.snapDistances)).toEqual([0.5, -1, 0.5, -1, 0.5]);
  expect(Array.from(result.traversedEdges)).toEqual([0, 1, 0, 0]);
  expect(Array.from(result.traversedCounts)).toEqual([5, 0]); // 0,1,2,NONE,0.
  expect(result.traversedOverflow[0]).toBe(1);
  expect(result.breakCount[0]).toBe(2);
  expect(result.matchedCount[0]).toBe(3);
  expect(result.trackLogLikelihoods[1]).toBe(0);
});

it('bounds the whole route inclusively and retains nearest-edge tie order', () => {
  const {input, props} = createMatchingFixture();
  input.positions = new Float32Array([2, 0, 28, 0, 5, 0]);
  input.trackOffsets = new Uint32Array([0, 3]);
  input.parameters = getGPUMapMatchingParameterValues({
    sigma: 1,
    beta: 1,
    searchRadius: 1,
    routeFactor: 0,
    routeSlack: 26
  });
  const result = computeMapMatchingOnCPU(input, {...props, candidateCount: 1});
  expect(Array.from(result.breaks)).toEqual([0, 0, 1]);
  input.parameters = getGPUMapMatchingParameterValues({
    sigma: 1,
    beta: 1,
    searchRadius: 1,
    routeFactor: 0,
    routeSlack: 25
  });
  expect(Array.from(computeMapMatchingOnCPU(input, {...props, candidateCount: 1}).breaks)).toEqual([
    0, 1, 1
  ]);
  input.positions = new Float32Array([5, 1.5]);
  input.trackOffsets = new Uint32Array([0, 1]);
  input.parameters = getGPUMapMatchingParameterValues({sigma: 1, beta: 1, searchRadius: 2});
  expect(computeMapMatchingOnCPU(input, props).matchedEdges[0]).toBe(0);
});

it('has no grid or route capacity cap and reconstructs a route longer than 128 nodes', () => {
  const edgeCount = 160;
  const input: CPUMapMatchingInput = {
    positions: new Float32Array([0.25, 0, 159.75, 0]),
    trackOffsets: new Uint32Array([0, 2]),
    parameters: getGPUMapMatchingParameterValues({
      sigma: 1,
      beta: 1,
      searchRadius: 0.1,
      routeFactor: 1,
      routeSlack: 1
    }),
    topology: {
      offsets: Uint32Array.from({length: edgeCount + 2}, (_, node) => Math.min(node, edgeCount)),
      neighbors: Uint32Array.from({length: edgeCount}, (_, edge) => edge + 1),
      nodePositions: Float32Array.from({length: (edgeCount + 1) * 2}, (_, component) =>
        component % 2 ? 0 : component / 2
      )
    }
  };
  const result = computeMapMatchingOnCPU(input, {
    cellSize: 0.001,
    candidateCount: 1,
    traversedEdgeCapacity: edgeCount
  });
  expect(Array.from(result.matchedEdges)).toEqual([0, 159]);
  expect(Array.from(result.traversedEdges)).toEqual(
    Array.from({length: edgeCount}, (_, edge) => edge)
  );
  expect(result.traversedCounts[0]).toBe(160);
  expect(result.overflow[0]).toBe(0);
  expect(result.routeOverflow[0]).toBe(0);
  expect(result.trackLogLikelihoods[0]).toBe(Math.fround(-Math.log(2 * Math.PI)));
});

it('recovers the first-discovered branch of an equal-length diamond route', () => {
  const {input, props} = createMatchingFixture();
  input.topology = {
    offsets: new Uint32Array([0, 1, 3, 4, 5, 6, 6]),
    neighbors: new Uint32Array([1, 2, 3, 4, 4, 5]),
    nodePositions: new Float32Array([-2, 0, -1, 0, 0, 1, 0, -1, 1, 0, 2, 0])
  };
  input.positions = new Float32Array([-1.5, 0, 1.5, 0]);
  input.trackOffsets = new Uint32Array([0, 2]);
  input.parameters = getGPUMapMatchingParameterValues({
    sigma: 1,
    beta: 2,
    searchRadius: 0.1,
    routeFactor: 2,
    routeSlack: 1
  });
  const result = computeMapMatchingOnCPU(input, {...props, candidateCount: 1});
  expect(Array.from(result.matchedEdges)).toEqual([0, 5]);
  expect(Array.from(result.traversedEdges)).toEqual([0, 1, 3, 5]);
  expect(result.traversedCounts[0]).toBe(4);
  // Route = 1 + 2 sqrt(2); straight = 3. Both symmetric diamond branches attain it.
  expect(result.trackLogLikelihoods[0]).toBe(
    Math.fround(-Math.log(2 * Math.PI) - Math.log(2) - (2 * Math.sqrt(2) - 2) / 2)
  );
});

it('matches an exact-score Viterbi tie by the final candidate and routes backward via a legal loop', () => {
  const {input, props} = createMatchingFixture();
  input.positions = new Float32Array([2, 1, 28, 2]);
  input.trackOffsets = new Uint32Array([0, 2]);
  const tied = computeMapMatchingOnCPU(input, props);
  // Both roads have squared emission distances 1 + 4. The final fix is closer to road 3..5,
  // so its candidate slot is first and wins the equal final scores, as in the GPU backward pass.
  expect(Array.from(tied.matchedEdges)).toEqual([3, 5]);
  input.topology = {
    offsets: new Uint32Array([0, 1, 2]),
    neighbors: new Uint32Array([1, 0]),
    nodePositions: new Float32Array([0, 0, 10, 0])
  };
  input.positions = new Float32Array([8, 0, 2, 0]);
  input.parameters = getGPUMapMatchingParameterValues({
    sigma: 1,
    beta: 2,
    searchRadius: 1,
    routeFactor: 2,
    routeSlack: 2
  });
  const loop = computeMapMatchingOnCPU(input, {...props, candidateCount: 1});
  expect(Array.from(loop.matchedEdges)).toEqual([0, 0]);
  expect(Array.from(loop.breaks)).toEqual([0, 0]);
  expect(Array.from(loop.traversedEdges)).toEqual([0, 1, 0, 0]);
  expect(loop.traversedCounts[0]).toBe(3);
  // Same-edge backward travel is not a direct hop: 2 + 10 + 2 = 14, versus straight 6.
  expect(loop.trackLogLikelihoods[0]).toBe(Math.fround(-Math.log(2 * Math.PI) - Math.log(2) - 4));
});

it('great-circle seam, equivalent longitudes, collapsed edges and kilometer/custom-radius units have closed forms', () => {
  const {input, props} = createMatchingFixture();
  input.topology = {
    offsets: new Uint32Array([0, 1, 1]),
    neighbors: new Uint32Array([1]),
    nodePositions: new Float32Array([179, 0, -179, 0])
  };
  input.positions = new Float32Array([180, 0, -180, 0]);
  input.trackOffsets = new Uint32Array([0, 1, 2]);
  input.parameters = getGPUMapMatchingParameterValues({sigma: 1, beta: 1, searchRadius: 1});
  const spatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'kilometers',
    sphereRadius: 1000
  } as const;
  const result = computeMapMatchingOnCPU(input, {...props, spatialContext});
  expect(Array.from(result.matchedFractions)).toEqual([0.5, 0.5]);
  expect(Array.from(result.snapDistances)).toEqual([0, 0]);
  expect(Array.from(result.matchedOffsets)).toEqual([
    Math.fround((1000 * Math.PI) / 180),
    Math.fround((1000 * Math.PI) / 180)
  ]);
  // WGSL's nearest-even wrap leaves either endpoint of [-180,180] legal.
  expect(Array.from(result.snappedPositions)).toEqual([180, 0, 180, 0]);
  input.topology.nodePositions = new Float32Array([180, 0, 180, 0]);
  expect(
    Array.from(computeMapMatchingOnCPU(input, {...props, spatialContext}).matchedFractions)
  ).toEqual([0, 0]);
});

it('keeps finite likelihoods beyond the GPU integer accumulator range', () => {
  const pointCount = 50;
  const sigma = Math.fround(0.0001);
  const input: CPUMapMatchingInput = {
    positions: Float32Array.from({length: pointCount * 2}, (_, component) =>
      component % 2 ? 1 : component / 2 + 1
    ),
    trackOffsets: new Uint32Array([0, pointCount]),
    parameters: getGPUMapMatchingParameterValues({sigma, beta: 1, searchRadius: 2}),
    topology: {
      offsets: new Uint32Array([0, 1, 1]),
      neighbors: new Uint32Array([1]),
      nodePositions: new Float32Array([0, 0, 100, 0])
    }
  };
  const result = computeMapMatchingOnCPU(input, {cellSize: 10, candidateCount: 1});
  // All snap distances are exactly 1 and route equals straight; each transition contributes 0.
  const expected = pointCount * (-Math.log(sigma) - 0.5 * Math.log(2 * Math.PI) - 0.5 / sigma ** 2);
  expect(expected).toBeLessThan(-(2 ** 31));
  expect(result.trackLogLikelihoods[0]).toBe(Math.fround(expected));
  expect(result.breakCount[0]).toBe(0);
});

it('ignores unused slots and invalid targets and clamps monotonic network offsets', () => {
  const {input, props} = createMatchingFixture();
  input.topology = {
    offsets: new Uint32Array([1, 3, 99]),
    neighbors: new Uint32Array([1, 99, 1, 99]),
    nodePositions: new Float32Array([0, 0, 10, 0])
  };
  input.positions = new Float32Array([5, 0]);
  input.trackOffsets = new Uint32Array([0, 1]);
  expect(computeMapMatchingOnCPU(input, props).matchedEdges[0]).toBe(2);
});

it('validates malformed arrays and both operation constructions, including deferred GPU validation', () => {
  const {input, props} = createMatchingFixture();
  for (const [name, changed] of [
    ['positions', {...input, positions: new Float32Array(3)}],
    ['trackOffsets', {...input, trackOffsets: new Uint32Array([0, 6])}],
    ['parameters', {...input, parameters: new Float32Array(7)}],
    [
      'topology.offsets',
      {...input, topology: {...input.topology, offsets: new Uint32Array([0, 3, 2])}}
    ],
    [
      'topology.nodePositions',
      {...input, topology: {...input.topology, nodePositions: new Float32Array(2)}}
    ],
    ['topology.weights', {...input, topology: {...input.topology, weights: new Float32Array(2)}}]
  ] as const)
    expect(() => computeMapMatchingOnCPU(changed, props)).toThrow(name);
  const shapes = Object.fromEntries(
    Object.entries(getMatchingGraphInputs(input)).map(([port, value]) => [
      port,
      {format: value.format, length: value.values.length / (value.format === 'float32x2' ? 2 : 1)}
    ])
  );
  expect(() =>
    constructNetworkOperationContributor(new GPUMapMatchingOperation(props), shapes)
  ).not.toThrow();
  for (const changed of [
    {...props, cellSize: 0},
    {...props, minimumFixSpacingSigmas: -1}
  ]) {
    expect(() => new CPUMapMatchingOperation(changed)).toThrow(/match-test/);
    expect(() =>
      constructNetworkOperationContributor(new GPUMapMatchingOperation(changed), shapes)
    ).toThrow(/match-test/);
  }
  for (const Operation of [CPUMapMatchingOperation, GPUMapMatchingOperation]) {
    expect(
      () => new Operation({...props, output: {...props.output, matchedEdges: 'positions'}})
    ).toThrow(/match-test/);
    expect(() => new Operation({...props, pointCount: 0})).toThrow(/match-test/);
    expect(() => new Operation({...props, traversedEdgeCapacity: undefined})).toThrow(/match-test/);
  }
  expect(() =>
    constructNetworkOperationContributor(
      new GPUMapMatchingOperation({...props, candidateCount: 9}),
      shapes
    )
  ).toThrow(/candidateCount/);
  expect(
    () =>
      new CPUMapMatchingOperation({
        ...props,
        candidateCount: 9,
        routeNodeBudget: 1000,
        entryCapacity: 1
      })
  ).not.toThrow();
});
