// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeVectorStraightSkeletonOnCPU,
  CPUVectorStraightSkeletonOperation,
  type CPUVectorStraightSkeletonInput,
  type CPUVectorStraightSkeletonResult
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';
import {
  GPUVectorStraightSkeletonOperation,
  type VectorStraightSkeletonOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton-operation';
import {GPUVectorStraightSkeleton} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton';
import fixtures from '../../oracles/fixtures/geometry-straight-skeleton.json';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function createInput(
  positions: readonly number[],
  ringOffsets = [0, positions.length / 2]
): CPUVectorStraightSkeletonInput {
  return {
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.of(0, ringOffsets.length - 1),
    featureOffsets: Uint32Array.of(0, 1)
  };
}

function createProps(): VectorStraightSkeletonOperationProps {
  return {
    id: 'vector-test',
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    polygonCount: 1,
    maximumVertices: 16,
    maximumEvents: 32,
    segmentCapacity: 64,
    offsetDistance: 0.25,
    output: {
      segments: 'segments',
      sourceEdges: 'sourceEdges',
      offsetPositions: 'offsetPositions',
      offsetLinks: 'offsetLinks',
      status: 'status'
    }
  };
}

function checkSnapshot(result: CPUVectorStraightSkeletonResult, slots: number): void {
  let count = 0;
  for (let vertex = 0; vertex < slots; vertex++) {
    const previous = result.offsetLinks[vertex * 2];
    const next = result.offsetLinks[vertex * 2 + 1];
    if (previous === 0xffffffff) {
      expect(next).toBe(0xffffffff);
      continue;
    }
    count++;
    expect(previous).toBeLessThan(slots);
    expect(next).toBeLessThan(slots);
    expect(result.offsetLinks[previous * 2 + 1]).toBe(vertex);
    expect(result.offsetLinks[next * 2]).toBe(vertex);
  }
  expect(result.status[2]).toBe(count);
}

function computeContributorOnInterpreter(
  input: CPUVectorStraightSkeletonInput,
  props: VectorStraightSkeletonOperationProps
): CPUVectorStraightSkeletonResult {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const operation = new GPUVectorStraightSkeletonOperation(props);
  const views: Record<string, GraphDataView> = {};
  for (const [port, values] of Object.entries(input)) {
    views[port] = createTransientView(
      graph,
      port,
      port === 'positions' ? 'float32x2' : 'uint32',
      port === 'positions' ? values.length / 2 : values.length
    );
    memory.write(views[port], values);
  }
  for (const [port, output] of Object.entries(operation.outputs))
    views[port] = createTransientView(graph, port, output.format, output.length);
  const contributor = operation.createContributor(
    <Format extends GPUGraphVectorFormat>(port: string) => views[port] as GraphDataView<Format>
  );
  if (!(contributor instanceof GPUVectorStraightSkeleton))
    throw new Error('expected GPUVectorStraightSkeleton');
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  return {
    segments: memory.readFloat32(views['segments']),
    sourceEdges: memory.readUint32(views['sourceEdges']),
    offsetPositions: memory.readFloat32(views['offsetPositions']),
    offsetLinks: memory.readUint32(views['offsetLinks']),
    status: memory.readUint32(views['status'])
  };
}

it.each(
  fixtures.cases
)('matches all independent CGAL 6.0.1 arcs and offset vertices: $name', fixture => {
  const result = computeVectorStraightSkeletonOnCPU(
    createInput(fixture.positions, fixture.ringOffsets),
    {
      maximumVertices: 32,
      maximumEvents: 64,
      segmentCapacity: 128,
      offsetDistance: fixture.offsetDistance
    }
  );
  // CGAL produces zero-length simultaneous-event arcs; the contributor deliberately omits them.
  const expected = fixture.arcs
    .map(arc => arc.slice(0, 4))
    .filter(arc => Math.hypot(arc[0] - arc[2], arc[1] - arc[3]) > 1e-5);
  expect(Array.from(result.status)).toEqual([
    expected.length,
    expected.length,
    fixture.offsets.flat().length,
    0
  ]);
  for (let row = 0; row < result.status[0]; row++) {
    const actual = result.segments.subarray(row * 4, row * 4 + 4);
    const match = expected.findIndex(
      arc =>
        Math.min(
          Math.hypot(...arc.map((value, index) => value - actual[index])),
          Math.hypot(arc[2] - actual[0], arc[3] - actual[1], arc[0] - actual[2], arc[1] - actual[3])
        ) < 3e-6
    );
    expect(match).toBeGreaterThanOrEqual(0);
    expected.splice(match, 1);
  }
  expect(expected).toHaveLength(0);
  const expectedPoints = fixture.offsets.flat().map(point => [...point]);
  for (let vertex = 0; vertex < 160; vertex++) {
    if (result.offsetLinks[vertex * 2] === 0xffffffff) continue;
    const match = expectedPoints.findIndex(
      point =>
        Math.hypot(
          point[0] - result.offsetPositions[vertex * 2],
          point[1] - result.offsetPositions[vertex * 2 + 1]
        ) < 1e-6
    );
    expect(match).toBeGreaterThanOrEqual(0);
    expectedPoints.splice(match, 1);
  }
  expect(expectedPoints).toHaveLength(0);
  checkSnapshot(result, 160);
});

it('matches hand-computed rectangle provenance, ordering, snapshot links and one output rounding', () => {
  const result = computeVectorStraightSkeletonOnCPU(createInput([0, 0, 4, 0, 4, 2, 0, 2]), {
    maximumVertices: 4,
    maximumEvents: 8,
    segmentCapacity: 8,
    offsetDistance: 1 / 3
  });
  expect(Array.from(result.status)).toEqual([5, 5, 4, 0]);
  expect(Array.from(result.segments.subarray(0, 20))).toEqual([
    4, 0, 3, 1, 4, 2, 3, 1, 0, 0, 1, 1, 0, 2, 1, 1, 1, 1, 3, 1
  ]);
  expect(Array.from(result.sourceEdges.subarray(0, 10))).toEqual([0, 1, 1, 2, 3, 0, 2, 3, 3, 0]);
  expect(Array.from(result.offsetLinks.subarray(0, 8))).toEqual([3, 1, 0, 2, 1, 3, 2, 0]);
  expect(Array.from(result.offsetPositions.subarray(0, 8))).toEqual(
    Array.from(
      Float32Array.of(1 / 3, 1 / 3, 4 - 1 / 3, 1 / 3, 4 - 1 / 3, 2 - 1 / 3, 1 / 3, 2 - 1 / 3)
    )
  );
  expect(result.segments.subarray(20).every(value => value === 0)).toBe(true);
  expect(result.sourceEdges.subarray(10).every(value => value === 0)).toBe(true);
  expect(result.offsetLinks.subarray(8).every(value => value === 0xffffffff)).toBe(true);
});

it('captures two components after the notch split, with new slots and reciprocal links', () => {
  const fixture = fixtures.cases.find(value => value.name === 'split-notch')!;
  const result = computeVectorStraightSkeletonOnCPU(createInput(fixture.positions), {
    maximumVertices: 7,
    maximumEvents: 16,
    segmentCapacity: 32,
    offsetDistance: 1
  });
  expect(Array.from(result.status)).toEqual([11, 11, 8, 0]);
  checkSnapshot(result, 39);
  const visited = new Set<number>();
  const componentSizes: number[] = [];
  for (let root = 0; root < 39; root++) {
    if (result.offsetLinks[root * 2] === 0xffffffff || visited.has(root)) continue;
    let current = root;
    let size = 0;
    do {
      expect(visited.has(current)).toBe(false);
      visited.add(current);
      size++;
      current = result.offsetLinks[current * 2 + 1];
    } while (current !== root);
    componentSizes.push(size);
  }
  expect(componentSizes).toEqual([4, 4]);
  // Original reflex slot 4 died; the two split births occupy slots 7 and 8.
  expect(result.offsetLinks[8]).toBe(0xffffffff);
  expect(result.offsetLinks[14]).not.toBe(0xffffffff);
  expect(result.offsetLinks[16]).not.toBe(0xffffffff);
  // The four vertices on the bottom supporting edge all have y=offsetDistance.
  const bottom = [...visited].filter(vertex => result.offsetPositions[vertex * 2 + 1] === 1);
  expect(bottom).toHaveLength(4);
});

it('keeps polygon blocks, reversed closure, hole merging, capacity counts and extinction', () => {
  const fixture = fixtures.cases.find(value => value.name === 'asymmetric-hole')!;
  const firstCount = fixture.positions.length / 2;
  const input: CPUVectorStraightSkeletonInput = {
    positions: Float32Array.from([...fixture.positions, 20, 0, 20, 2, 24, 2, 24, 0, 20, 0]),
    ringOffsets: Uint32Array.of(0, 4, firstCount, firstCount + 5),
    polygonOffsets: Uint32Array.of(0, 2, 3),
    // Both polygons belong to one feature; changing feature grouping cannot alter skeletons.
    featureOffsets: Uint32Array.of(0, 2)
  };
  const result = computeVectorStraightSkeletonOnCPU(input, {
    maximumVertices: 8,
    maximumEvents: 16,
    segmentCapacity: 2,
    offsetDistance: 1.25
  });
  expect(Array.from(result.status)).toEqual([2, 16, 8, 2, 2, 5, 0, 2]);
  // At t=1.25 the expanded hole cuts through the bottom of the shrunken shell.
  // The resulting single boundary has these eight hand-computed corners.
  const expectedOffset = [
    [1.25, 1.25],
    [1.75, 1.25],
    [1.75, 6.25],
    [7.25, 6.25],
    [7.25, 1.25],
    [10.75, 1.25],
    [10.75, 8.75],
    [1.25, 8.75]
  ];
  for (let vertex = 0; vertex < 40; vertex++) {
    if (result.offsetLinks[vertex * 2] === 0xffffffff) continue;
    const match = expectedOffset.findIndex(
      point =>
        point[0] === result.offsetPositions[vertex * 2] &&
        point[1] === result.offsetPositions[vertex * 2 + 1]
    );
    expect(match).toBeGreaterThanOrEqual(0);
    expectedOffset.splice(match, 1);
  }
  expect(expectedOffset).toHaveLength(0);
  expect(
    result.sourceEdges.subarray(4, 8).every(value => value >= firstCount && value < firstCount + 4)
  ).toBe(true);
  expect(result.offsetLinks.subarray(80).every(value => value === 0xffffffff)).toBe(true);
  const regrouped = computeVectorStraightSkeletonOnCPU(
    {...input, featureOffsets: Uint32Array.of(0, 1, 2)},
    {
      maximumVertices: 8,
      maximumEvents: 16,
      segmentCapacity: 2,
      offsetDistance: 1.25
    }
  );
  expect(regrouped).toEqual(result);
});

it('removes the iteration budget but retains the historical output-slot limit', () => {
  const input = createInput([0, 0, 4, 0, 4, 2, 0, 2]);
  // The early snapshot fits the initial slots. Later events need no output slots and finish
  // even with a zero event budget and no initial-vertex padding.
  expect(
    Array.from(
      computeVectorStraightSkeletonOnCPU(input, {
        maximumVertices: 4,
        maximumEvents: 0,
        segmentCapacity: 8,
        offsetDistance: 0.25
      }).status
    )
  ).toEqual([5, 5, 4, 0]);
  expect(
    Array.from(
      computeVectorStraightSkeletonOnCPU(input, {
        maximumVertices: 4,
        maximumEvents: 0,
        segmentCapacity: 8,
        offsetDistance: 1.25
      }).status
    )
  ).toEqual([0, 0, 0, 4]);
});

it('terminates a repeated wavefront at the same time and retains the zero-distance snapshot', () => {
  const input = createInput([5, 3.5, 1.5, 2.5, 1, 8, -6.5, -3.7, -1.6, -2.2, -1, -5, 3, -2]);
  const props = {
    ...createProps(),
    maximumVertices: 32,
    maximumEvents: 0,
    segmentCapacity: 128,
    offsetDistance: 0,
    eventTolerance: 0.1
  };
  const result = computeVectorStraightSkeletonOnCPU(input, props);
  // The signed shoelace sum is positive (about 116.73), so initial slot order is unchanged.
  // At distance zero all seven original corners and their circular links must survive capture.
  // A repeated same-time state cannot make progress: the CPU reports numerical failure (8).
  expect(Array.from(result.status.subarray(2))).toEqual([7, 8]);
  expect(result.status[0]).toBe(result.status[1]);
  expect(result.status[0]).toBeGreaterThan(0);
  expect(result.status[0]).toBeLessThan(128);
  expect(Array.from(result.offsetPositions.subarray(0, 14))).toEqual(Array.from(input.positions));
  expect(Array.from(result.offsetLinks.subarray(0, 14))).toEqual([
    6, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 0
  ]);
  checkSnapshot(result, 32);
  expect(result.segments.every(Number.isFinite)).toBe(true);
  // Independently derive the GPU budget result: capture precedes the zero-budget check,
  // and emission follows it, hence seven offset vertices, no arcs, work flag 4.
  const interpreted = computeContributorOnInterpreter(input, props);
  expect(Array.from(interpreted.status)).toEqual([0, 0, 7, 4]);
  expect(interpreted.segments.every(value => value === 0)).toBe(true);
  expect(interpreted.sourceEdges.every(value => value === 0)).toBe(true);
  expect(interpreted.offsetLinks).toEqual(result.offsetLinks);
  interpreted.offsetPositions.forEach((value, index) =>
    expect(Math.abs(value - result.offsetPositions[index])).toBeLessThanOrEqual(3e-5)
  );
});

it('accepts a zero-polygon batch with terminal-only offset levels and five empty outputs', async () => {
  const input: CPUVectorStraightSkeletonInput = {
    positions: new Float32Array(),
    ringOffsets: Uint32Array.of(0),
    polygonOffsets: Uint32Array.of(0),
    featureOffsets: Uint32Array.of(0, 0)
  };
  const props = {
    ...createProps(),
    polygonCount: 0,
    maximumVertices: 32,
    maximumEvents: 0,
    segmentCapacity: 128,
    offsetDistance: 0,
    eventTolerance: 0.1
  };
  // Zero polygons multiply every fixed-block output length by zero, including status.
  const result = computeVectorStraightSkeletonOnCPU(input, props);
  for (const values of Object.values(result)) expect(values).toHaveLength(0);
  expect(
    computeVectorStraightSkeletonOnCPU({...input, featureOffsets: Uint32Array.of(0)}, props)
  ).toEqual(result);
  const gpu = new GPUVectorStraightSkeletonOperation(props);
  for (const output of Object.values(gpu.outputs)) expect(output.length).toBe(0);
  expect(computeContributorOnInterpreter(input, props)).toEqual(result);
  const inputs = Object.fromEntries(
    Object.entries(input).map(([port, values]) => {
      const length = port === 'positions' ? values.length / 2 : values.length;
      return [
        port,
        {
          format: port === 'positions' ? ('float32x2' as const) : ('uint32' as const),
          length,
          chunks: [{values, length}],
          pack: () => values
        }
      ];
    })
  );
  const cpu = new CPUVectorStraightSkeletonOperation(props);
  for (const descriptor of Object.values(cpu.outputs)) expect(descriptor.length).toBe(0);
  expect(await cpu.execute({inputs})).toEqual(result);
});

it('honours degenerate and unsupported input flags and the inclusive offset event boundary', () => {
  const options = {maximumVertices: 4, maximumEvents: 8, segmentCapacity: 16, offsetDistance: 1};
  const atEvent = computeVectorStraightSkeletonOnCPU(
    createInput([0, 0, 4, 0, 4, 2, 0, 2]),
    options
  );
  expect(Array.from(atEvent.status)).toEqual([5, 5, 4, 0]);
  expect(Array.from(atEvent.offsetPositions.subarray(0, 8))).toEqual([1, 1, 3, 1, 3, 1, 1, 1]);
  for (const positions of [
    [0, 0, 1, 0],
    [0, 0, 0, 0, 4, 2, 0, 2],
    [0, 0, Infinity, 0, 4, 2, 0, 2],
    [0, 0, 4, 0, 4, 2, 2, 3, 0, 2]
  ]) {
    expect(
      Array.from(computeVectorStraightSkeletonOnCPU(createInput(positions), options).status)
    ).toEqual([0, 0, 0, 1]);
  }
  // A zero-area ring is a terminal ridge component, not an invalid polygon.
  const degenerate = computeVectorStraightSkeletonOnCPU(createInput([0, 0, 1, 0, 2, 0]), {
    ...options,
    offsetDistance: 0
  });
  expect(Array.from(degenerate.status)).toEqual([2, 2, 0, 0]);
  expect(Array.from(degenerate.segments.subarray(0, 8))).toEqual([0, 0, 1, 0, 1, 0, 2, 0]);
  const empty = computeVectorStraightSkeletonOnCPU(
    {
      positions: new Float32Array(),
      ringOffsets: Uint32Array.of(0),
      polygonOffsets: Uint32Array.of(0, 0),
      featureOffsets: Uint32Array.of(0, 1)
    },
    options
  );
  expect(Array.from(empty.status)).toEqual([0, 0, 0, 0]);
  expect(empty.offsetLinks.every(value => value === 0xffffffff)).toBe(true);
});

it('rejects inconsistent packed arrays with the argument name', () => {
  const input = createInput([0, 0, 4, 0, 4, 2, 0, 2]);
  const options = {segmentCapacity: 8, offsetDistance: 0};
  for (const [argument, values] of [
    ['positions', Float32Array.of(0)],
    ['ringOffsets', Uint32Array.of(0, 5)],
    ['ringOffsets', Uint32Array.of(3, 1)],
    ['ringOffsets', new Uint32Array()],
    ['ringOffsets', Uint32Array.of(5)],
    ['polygonOffsets', Uint32Array.of(0, 2)],
    ['polygonOffsets', Uint32Array.of(1, 0)],
    ['polygonOffsets', Uint32Array.of(2)],
    ['polygonOffsets', new Uint32Array()],
    ['featureOffsets', Uint32Array.of(0, 2)],
    ['featureOffsets', Uint32Array.of(1, 0)],
    ['featureOffsets', Uint32Array.of(2)],
    ['featureOffsets', new Uint32Array()]
  ] as const)
    expect(() =>
      computeVectorStraightSkeletonOnCPU({...input, [argument]: values}, options)
    ).toThrow(new RegExp(argument));
});

it.each([
  GPUVectorStraightSkeletonOperation,
  CPUVectorStraightSkeletonOperation
])('%s checks graph output shapes and names at construction', Operation => {
  expect(() => new Operation(createProps())).not.toThrow();
  expect(() => new Operation({...createProps(), polygonCount: 0})).not.toThrow();
  expect(() => new Operation({...createProps(), polygonCount: -1})).toThrow(/^vector-test/);
  expect(() => new Operation({...createProps(), segmentCapacity: 0})).toThrow(/^vector-test/);
  expect(
    () =>
      new Operation({...createProps(), output: {...createProps().output, segments: 'positions'}})
  ).toThrow(/^vector-test/);
  expect(
    () =>
      new Operation({...createProps(), output: {...createProps().output, sourceEdges: 'segments'}})
  ).toThrow(/^vector-test/);
});

it('validates CPU options at construction and removes GPU-only ceilings', () => {
  for (const [name, value] of [
    ['maximumVertices', 2],
    ['maximumEvents', -1],
    ['segmentCapacity', 0],
    ['offsetDistance', -1],
    ['offsetDistance', -1e-50],
    ['offsetDistance', NaN],
    ['eventTolerance', 0],
    ['eventTolerance', Infinity]
  ] as const)
    expect(() => new CPUVectorStraightSkeletonOperation({...createProps(), [name]: value})).toThrow(
      new RegExp(`^vector-test .*${name}`)
    );
  expect(
    () =>
      new CPUVectorStraightSkeletonOperation({
        ...createProps(),
        maximumVertices: 1025,
        maximumEvents: 4097
      })
  ).not.toThrow();
});

it('executes every CPU graph port and checks explicit polygonCount', async () => {
  const input = createInput([0, 0, 4, 0, 4, 2, 0, 2]);
  const inputs = Object.fromEntries(
    Object.entries(input).map(([port, values]) => {
      const length = port === 'positions' ? values.length / 2 : values.length;
      return [
        port,
        {
          format: port === 'positions' ? ('float32x2' as const) : ('uint32' as const),
          length,
          chunks: [{values, length}],
          pack: () => values
        }
      ];
    })
  );
  const result = await new CPUVectorStraightSkeletonOperation(createProps()).execute({inputs});
  const expected = computeVectorStraightSkeletonOnCPU(input, createProps());
  for (const [port, values] of Object.entries(expected)) expect(result[port]).toEqual(values);
  expect(() =>
    new CPUVectorStraightSkeletonOperation({...createProps(), polygonCount: 2}).execute({inputs})
  ).toThrow(/polygonOffsets/);
});

it.each([
  'ordinary',
  'overflow',
  'extinct',
  'unsupported',
  'empty'
] as const)('checks the headless scene against interpreted contributor kernels: %s', scene => {
  // This supplements the independent CGAL/closed-form oracles above. It is not a substitute
  // for running the comparison spec on real WebGPU, especially for compiler behaviour.
  const input: CPUVectorStraightSkeletonInput = {
    positions: Float32Array.of(
      0,
      0,
      12,
      0,
      12,
      10,
      0,
      10,
      3,
      2,
      3,
      5,
      6,
      5,
      6,
      2,
      20,
      0,
      28,
      0,
      28,
      2,
      22,
      2,
      22,
      6,
      28,
      6,
      28,
      8,
      20,
      8
    ),
    ringOffsets: Uint32Array.of(0, 4, 8, 16),
    polygonOffsets: scene === 'empty' ? Uint32Array.of(0, 0, 0) : Uint32Array.of(0, 2, 3),
    featureOffsets: Uint32Array.of(0, 2)
  };
  const props: VectorStraightSkeletonOperationProps = {
    ...createProps(),
    polygonCount: 2,
    maximumVertices: scene === 'unsupported' ? 7 : 8,
    maximumEvents: 32,
    segmentCapacity: scene === 'overflow' ? 2 : 32,
    offsetDistance: scene === 'extinct' ? 6 : 1.25
  };
  const interpreted = computeContributorOnInterpreter(input, props);
  const result = computeVectorStraightSkeletonOnCPU(input, props);
  for (const [port, expected] of Object.entries(result)) {
    const actual = interpreted[port as keyof CPUVectorStraightSkeletonResult];
    expect(actual.length).toBe(expected.length);
    for (let index = 0; index < expected.length; index++) {
      if (expected instanceof Float32Array)
        expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(3e-5);
      else expect(actual[index] === expected[index]).toBe(true);
    }
  }
});
