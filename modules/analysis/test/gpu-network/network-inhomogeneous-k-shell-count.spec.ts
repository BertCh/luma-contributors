// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkInhomogeneousK} from '../../src/gpu-network/gpu-network-inhomogeneous-k';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// The Ang correction divides each pair by m(u, t), the number of network locations at exactly the
// pair distance t from the source u. These cases use edge lengths that are not dyadic, so the f32
// distance table (d[v] = fl(d[u] + length), as a GPU relaxation writes it) differs from the real
// sums by a rounding step. The expected counts come from the geometry of each network, never from
// the kernel's own root arithmetic.

type TestNetwork = {
  nodeCount: number;
  edges: [number, number][];
  /** Edge lengths as the kernel sees them (already rounded to f32). */
  lengths: number[];
};

/** Shortest-path table in f32, as a GPU relaxation produces it: every sum is rounded. */
function getFloat32DistanceTable(network: TestNetwork, sourceNodes: number[]): number[] {
  const table: number[] = [];
  for (const sourceNode of sourceNodes) {
    const distances = new Float32Array(network.nodeCount).fill(Infinity);
    const settled = new Uint8Array(network.nodeCount);
    distances[sourceNode] = 0;
    for (;;) {
      let nearest = -1;
      for (let node = 0; node < network.nodeCount; node++) {
        if (!settled[node] && (nearest < 0 || distances[node] < distances[nearest])) {
          nearest = node;
        }
      }
      if (nearest < 0 || distances[nearest] === Infinity) {
        break;
      }
      settled[nearest] = 1;
      network.edges.forEach(([first, second], edge) => {
        const other = first === nearest ? second : second === nearest ? first : -1;
        if (other >= 0) {
          const candidate = Math.fround(distances[nearest] + network.lengths[edge]);
          distances[other] = Math.min(distances[other], candidate);
        }
      });
    }
    table.push(...distances);
  }
  return table;
}

/**
 * Runs the Ang-corrected estimator with every node as an event (or the nodes of `eventNodes`)
 * and checks the numerators. `shellCounts[u][v]` and `pairDistances[u][v]`, indexed by event,
 * are the independent truth the numerators are compared with: sum over ordered pairs within the
 * radius of 1 / (lambda_u lambda_v m(u, t)).
 */
async function expectAngNumerators(
  device: Device,
  network: TestNetwork,
  intensity: number[],
  radii: number[],
  pairDistances: number[][],
  shellCounts: number[][],
  eventNodes?: number[],
  verifyNormalized = false,
  correction: 'ang' | 'none' = 'ang'
): Promise<void> {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
    format: Format,
    values: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      format === 'uint32x2' ? values.length / 2 : values.length
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, count: number) => {
    const buffer = createOutputBuffer(device, count);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, count),
      buffer
    };
  };
  const events = eventNodes ?? Array.from({length: network.nodeCount}, (_, node) => node);
  const numerator = output('float32', radii.length);
  const normalized = output('float32', radii.length);
  const status = output('uint32', radii.length);
  graph.add(
    new GPUNetworkInhomogeneousK({
      eventNodeIds: input('uint32', events),
      intensity: input('float32', intensity),
      nodeDistances: input('float32', getFloat32DistanceTable(network, events)),
      nodeCount: network.nodeCount,
      edgeNodes: input('uint32x2', network.edges.flat()),
      edgeLengths: input('float32', network.lengths),
      radii: input('float32', radii),
      correction,
      output: {
        k: normalized.view,
        numerator: numerator.view,
        normalization: output('float32', 3).view,
        status: status.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const actual = await readFloat32(numerator.buffer, radii.length);
  const actualNormalized = verifyNormalized
    ? await readFloat32(normalized.buffer, radii.length)
    : undefined;
  expect(await readUint32(status.buffer, radii.length)).toEqual(radii.map(() => 0));
  for (let radiusIndex = 0; radiusIndex < radii.length; radiusIndex++) {
    let expected = 0;
    for (let source = 0; source < events.length; source++) {
      for (let destination = 0; destination < events.length; destination++) {
        if (source === destination) {
          continue;
        }
        const pairDistance = pairDistances[source][destination];
        // The radii of these cases are kept away from every pair distance, so the f32 threshold
        // test cannot disagree with the real one.
        expect(Math.abs(pairDistance - radii[radiusIndex])).toBeGreaterThan(1e-3);
        if (pairDistance <= radii[radiusIndex]) {
          expected +=
            1 / (intensity[source] * intensity[destination] * shellCounts[source][destination]);
        }
      }
    }
    expect(Math.abs(actual[radiusIndex] - expected)).toBeLessThanOrEqual(2e-5 * expected);
    if (actualNormalized) {
      const expectedNormalized =
        expected / network.lengths.reduce((total, length) => total + length, 0);
      expect(Math.abs(actualNormalized[radiusIndex] - expectedNormalized)).toBeLessThanOrEqual(
        2e-5 * expectedNormalized
      );
    }
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
}

function getPathNetwork(lengths: number[]): TestNetwork & {positions: number[]} {
  const roundedLengths = lengths.map(Math.fround);
  const positions = [0];
  for (const length of roundedLengths) {
    positions.push(positions[positions.length - 1] + length);
  }
  return {
    nodeCount: positions.length,
    edges: roundedLengths.map((_, edge) => [edge, edge + 1]),
    lengths: roundedLengths,
    positions
  };
}

it('Ang shell count on a path with tenth-unit edge lengths matches the hand table', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Node positions 0, 0.7, 2.0, 2.9, 5.0, 5.6. From u the shell at distance t = |x_v - x_u| is
  // the destination itself plus its mirror image 2 x_u - x_v when that lies on the path [0, 5.6].
  //   u = 0 and u = 5 are path ends: no mirror image, m = 1.
  //   u = 1 (0.7): mirror of v = 0 is 1.4 (on the path); of v = 2..5 it is negative.
  //   u = 2 (2.0): mirrors 4.0, 3.3, 1.1 for v = 0, 1, 3; negative for v = 4, 5.
  //   u = 3 (2.9): mirror of v = 0 is 5.8 (past the end); 5.1, 3.8, 0.8, 0.2 for v = 1, 2, 4, 5.
  //   u = 4 (5.0): mirrors of v = 0..3 are 10, 9.3, 8, 7.1 (past the end); of v = 5 it is 4.4.
  const network = getPathNetwork([0.7, 1.3, 0.9, 2.1, 0.6]);
  const shellCounts = [
    [0, 1, 1, 1, 1, 1],
    [2, 0, 1, 1, 1, 1],
    [2, 2, 0, 2, 1, 1],
    [1, 2, 2, 0, 2, 2],
    [1, 1, 1, 1, 0, 2],
    [1, 1, 1, 1, 1, 0]
  ];
  const pairDistances = network.positions.map(source =>
    network.positions.map(destination => Math.abs(destination - source))
  );
  await expectAngNumerators(
    device,
    network,
    [1, 2, 1, 4, 2, 1],
    [1.0, 2.5, 4.0, 6.0],
    pairDistances,
    shellCounts
  );
});

it('Ang shell count on a long path with irrational edge lengths is one or two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Forty nodes, block lengths between 60 and 200 from square roots, so no sum is exact in f32.
  const network = getPathNetwork(
    Array.from({length: 39}, (_, edge) => 60 + 140 * ((Math.sqrt(2) * (edge + 1)) % 1))
  );
  const {positions} = network;
  const pathEnd = positions[positions.length - 1];
  const pairDistances = positions.map(source =>
    positions.map(destination => Math.abs(destination - source))
  );
  const shellCounts = positions.map(source =>
    positions.map(destination => {
      const mirror = 2 * source - destination;
      // The mirror image is never within a millimetre of a path end, so the count is unambiguous.
      expect(Math.min(Math.abs(mirror), Math.abs(mirror - pathEnd))).toBeGreaterThan(
        mirror === destination ? -1 : 1e-3
      );
      return mirror >= 0 && mirror <= pathEnd ? 2 : 1;
    })
  );
  const intensity = positions.map((_, node) => 1 + (node % 3));
  await expectAngNumerators(
    device,
    network,
    intensity,
    [250.5, 1000.25, 2500.75, 6000.5],
    pairDistances,
    shellCounts
  );
});

it('Ang shell count on a four-cycle is two, and one for the antipodal pair', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Sides 0.7, 1.1, 0.9, 1.3 around the ring 0-1-2-3-0: perimeter 4, so a level set below
  // distance 2 has one location on each way round, and the level set at exactly 2 is one point.
  // Shortest distances: (0,1) 0.7, (0,2) 1.8, (0,3) 1.3, (1,2) 1.1, (1,3) 2.0 both ways, (2,3) 0.9.
  const network: TestNetwork = {
    nodeCount: 4,
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0]
    ],
    lengths: [0.7, 1.1, 0.9, 1.3].map(Math.fround)
  };
  const pairDistances = [
    [0, 0.7, 1.8, 1.3],
    [0.7, 0, 1.1, 2.0],
    [1.8, 1.1, 0, 0.9],
    [1.3, 2.0, 0.9, 0]
  ];
  const shellCounts = [
    [0, 2, 2, 2],
    [2, 0, 2, 1],
    [2, 2, 0, 2],
    [2, 1, 2, 0]
  ];
  // Unit intensities: the numerators are 4, 8 and 10 ordered pairs at one half, then two more at 1.
  await expectAngNumerators(
    device,
    network,
    [1, 1, 1, 1],
    [1.0, 1.5, 1.9, 2.5],
    pairDistances,
    shellCounts
  );
});

type TestGrid = TestNetwork & {columnPositions: number[]; rowPositions: number[]};

function getGridNetwork(columnPositions: number[], rowPositions: number[]): TestGrid {
  const columnCount = columnPositions.length;
  const edges: [number, number][] = [];
  const lengths: number[] = [];
  for (let row = 0; row < rowPositions.length; row++) {
    for (let column = 0; column < columnCount; column++) {
      const node = row * columnCount + column;
      if (column + 1 < columnCount) {
        edges.push([node, node + 1]);
        lengths.push(columnPositions[column + 1] - columnPositions[column]);
      }
      if (row + 1 < rowPositions.length) {
        edges.push([node, node + columnCount]);
        lengths.push(rowPositions[row + 1] - rowPositions[row]);
      }
    }
  }
  return {
    nodeCount: columnCount * rowPositions.length,
    edges,
    lengths,
    columnPositions,
    rowPositions
  };
}

/** Running sums of f32-rounded spacings, exact in binary64 for these few terms. */
function getPositions(spacings: number[]): number[] {
  const positions = [0];
  for (const spacing of spacings) {
    positions.push(positions[positions.length - 1] + Math.fround(spacing));
  }
  return positions;
}

/**
 * Level-set size on a full rectangular grid, where the network distance from a node to any
 * location on a grid line is the Manhattan distance. Each grid line holds the solutions of
 * |along - alongSource| = t - |across - acrossSource|; a node at distance t lies on a row and a
 * column and is counted once. `margin` receives how far the nearest solution is from becoming a
 * node or leaving the grid, so the caller can assert the count is unambiguous.
 */
function getGridShellCount(
  grid: TestGrid,
  sourceNode: number,
  pairDistance: number,
  margin: {value: number}
): number {
  const {columnPositions, rowPositions} = grid;
  const sourceX = columnPositions[sourceNode % columnPositions.length];
  const sourceY = rowPositions[Math.floor(sourceNode / columnPositions.length)];
  const exactTolerance = 1e-9;
  let count = 0;
  const countLine = (
    acrossPositions: number[],
    acrossSource: number,
    alongPositions: number[],
    alongSource: number
  ) => {
    const alongEnd = alongPositions[alongPositions.length - 1];
    for (const across of acrossPositions) {
      const reach = pairDistance - Math.abs(across - acrossSource);
      const candidates =
        Math.abs(reach) <= exactTolerance
          ? [alongSource]
          : reach > 0
            ? [alongSource - reach, alongSource + reach]
            : [];
      if (Math.abs(reach) > exactTolerance) {
        margin.value = Math.min(margin.value, Math.abs(reach));
      }
      for (const along of candidates) {
        const outside = Math.max(-along, along - alongEnd);
        if (outside <= exactTolerance) {
          count++;
        }
        if (Math.abs(outside) > exactTolerance) {
          margin.value = Math.min(margin.value, Math.abs(outside));
        }
        for (const nodeAlong of alongPositions) {
          if (Math.abs(along - nodeAlong) > exactTolerance) {
            margin.value = Math.min(margin.value, Math.abs(along - nodeAlong));
          }
        }
      }
    }
  };
  countLine(rowPositions, sourceY, columnPositions, sourceX);
  countLine(columnPositions, sourceX, rowPositions, sourceY);
  for (const nodeY of rowPositions) {
    for (const nodeX of columnPositions) {
      const nodeDistance = Math.abs(nodeX - sourceX) + Math.abs(nodeY - sourceY);
      if (Math.abs(nodeDistance - pairDistance) <= exactTolerance) {
        count--;
      }
    }
  }
  return count;
}

function getGridPairDistances(grid: TestGrid): number[][] {
  const columnCount = grid.columnPositions.length;
  const nodes = Array.from({length: grid.nodeCount}, (_, node) => node);
  return nodes.map(source =>
    nodes.map(
      destination =>
        Math.abs(
          grid.columnPositions[destination % columnCount] -
            grid.columnPositions[source % columnCount]
        ) +
        Math.abs(
          grid.rowPositions[Math.floor(destination / columnCount)] -
            grid.rowPositions[Math.floor(source / columnCount)]
        )
    )
  );
}

it('Ang shell count on an irregular grid with irrational spacings matches the Manhattan level sets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = getGridNetwork(
    getPositions([Math.SQRT2, Math.sqrt(3) / 2, Math.sqrt(5), Math.sqrt(7) / 3]),
    getPositions([Math.PI / 3, Math.E / 2, Math.sqrt(11) / 2])
  );
  const pairDistances = getGridPairDistances(grid);
  const margin = {value: Infinity};
  const shellCounts = pairDistances.map((row, source) =>
    row.map((pairDistance, destination) =>
      source === destination ? 0 : getGridShellCount(grid, source, pairDistance, margin)
    )
  );
  // No level-set location is nearly a node or nearly off the grid, so binary64 decides each count.
  expect(margin.value).toBeGreaterThan(1e-3);
  // By hand for the corner source and its right-hand neighbour, t = sqrt(2) = 1.414: the
  // neighbour itself on row 0, (1.414 - 1.047, 1.047) on row 1 and (0, 1.414) on column 0.
  expect(shellCounts[0][1]).toBe(3);
  const intensity = Array.from({length: grid.nodeCount}, (_, node) => 1 + (node % 4) / 2);
  await expectAngNumerators(
    device,
    grid,
    intensity,
    [1.05, 2.15, 3.35, 5.05, 9.5],
    pairDistances,
    shellCounts
  );
});

/**
 * Level-set sizes by enumeration: every edge is cut into pieces of half a length unit and a
 * breadth-first search gives the distance of every cut point in half units. With whole-unit edge
 * lengths every level-set location is a cut point, so the count is exact integer arithmetic.
 */
function getEnumeratedShellCounts(
  nodeCount: number,
  edges: [number, number][],
  lengthUnits: number[]
): number[][] {
  const neighbors: number[][] = Array.from({length: nodeCount}, () => []);
  edges.forEach(([first, second], edge) => {
    let previous = first;
    for (let step = 1; step < 2 * lengthUnits[edge]; step++) {
      neighbors.push([previous]);
      neighbors[previous].push(neighbors.length - 1);
      previous = neighbors.length - 1;
    }
    neighbors[previous].push(second);
    neighbors[second].push(previous);
  });
  const shellCounts: number[][] = [];
  for (let source = 0; source < nodeCount; source++) {
    const steps = new Int32Array(neighbors.length).fill(-1);
    steps[source] = 0;
    let frontier = [source];
    while (frontier.length > 0) {
      const next: number[] = [];
      for (const location of frontier) {
        for (const neighbor of neighbors[location]) {
          if (steps[neighbor] < 0) {
            steps[neighbor] = steps[location] + 1;
            next.push(neighbor);
          }
        }
      }
      frontier = next;
    }
    shellCounts.push(
      Array.from({length: nodeCount}, (_, destination) =>
        steps.reduce((count, step) => count + (step === steps[destination] ? 1 : 0), 0)
      )
    );
  }
  return shellCounts;
}

it('Ang shell count on an irregular grid with tenth-unit spacings matches enumerated level sets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Spacings in tenths: many nodes and edge crossings are at exactly equal real distances, while
  // their f32 table entries differ by a rounding step. Exact ties must still count once.
  const columnUnits = [7, 11, 3, 13];
  const rowUnits = [9, 5, 17];
  const toPositions = (units: number[]) => {
    const positions = [0];
    for (const unit of units) {
      positions.push(positions[positions.length - 1] + unit);
    }
    return positions;
  };
  const unitGrid = getGridNetwork(toPositions(columnUnits), toPositions(rowUnits));
  const shellCounts = getEnumeratedShellCounts(
    unitGrid.nodeCount,
    unitGrid.edges,
    unitGrid.lengths
  );
  // Largest level set of this grid, by the enumeration: twelve locations.
  expect(Math.max(...shellCounts.flat())).toBe(12);
  const network: TestNetwork = {
    ...unitGrid,
    lengths: unitGrid.lengths.map(units => Math.fround(units * 0.1))
  };
  const pairDistances = getGridPairDistances(unitGrid).map(row => row.map(units => units * 0.1));
  const intensity = Array.from({length: network.nodeCount}, (_, node) => 1 + (node % 3));
  await expectAngNumerators(
    device,
    network,
    intensity,
    [0.95, 2.05, 3.55, 6.45],
    pairDistances,
    shellCounts
  );
});

it('Ang shell count takes nodes on the shell that a very short edge joins as one location', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Node 0 is the source and node 1 the destination, 1000 away on its own branch, so the pair
  // distance is 1000 and the tolerance 0.01. A second branch reaches the shell through edges
  // shorter than the tolerance, so several of its nodes are within the tolerance of the shell.
  // From node 1 back to node 0 the only location at 1000 is node 0 itself: m = 1 in every case.
  const cases: {name: string; edges: [number, number, number][]; shellCount: number}[] = [
    {
      // Nodes at 999.995 and 1000.005 joined by an edge of 0.01: one crossing between them, plus
      // the destination. Counting both nodes gave 3.
      name: 'two nodes',
      edges: [
        [0, 1, 1000],
        [0, 2, 999.995],
        [2, 3, 0.01],
        [3, 4, 50]
      ],
      shellCount: 2
    },
    {
      // Nodes at 999.992, 1000 and 1000.008 in a row: still one crossing, plus the destination.
      name: 'three nodes in a row',
      edges: [
        [0, 1, 1000],
        [0, 2, 999.992],
        [2, 3, 0.008],
        [3, 4, 0.008],
        [4, 5, 50]
      ],
      shellCount: 2
    },
    {
      // Two branches at 999.996 and 999.997 meet in a node at 1000.004: the shell crosses each
      // branch once, so two crossings plus the destination. The meeting node is merged once,
      // although two edges merge it.
      name: 'two branches meeting past the shell',
      edges: [
        [0, 1, 1000],
        [0, 2, 999.996],
        [0, 3, 999.997],
        [2, 4, 0.008],
        [3, 4, 0.007],
        [4, 5, 50]
      ],
      shellCount: 3
    }
  ];
  for (const {name, edges, shellCount} of cases) {
    const network: TestNetwork = {
      nodeCount: Math.max(...edges.flatMap(([first, second]) => [first, second])) + 1,
      edges: edges.map(([first, second]) => [first, second]),
      lengths: edges.map(([, , length]) => Math.fround(length))
    };
    try {
      await expectAngNumerators(
        device,
        network,
        [1, 1],
        [1500],
        [
          [0, 1000],
          [1000, 0]
        ],
        [
          [0, shellCount],
          [1, 0]
        ],
        [0, 1]
      );
    } catch (error) {
      throw new Error(`${name}: ${(error as Error).message}`);
    }
  }
});

it('inhomogeneous K preserves mixed intensity scales and a large network-length divisor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await expectAngNumerators(
    device,
    {nodeCount: 2, edges: [[0, 1]], lengths: [1]},
    [2e-38, 3e38].map(Math.fround),
    [1.5],
    [
      [0, 1],
      [1, 0]
    ],
    [
      [0, 1],
      [1, 0]
    ],
    undefined,
    true,
    'none'
  );
  const length = Math.fround(3e38);
  await expectAngNumerators(
    device,
    {nodeCount: 2, edges: [[0, 1]], lengths: [length]},
    [1e-19, 1e-19].map(Math.fround),
    [Math.fround(3.2e38)],
    [
      [0, length],
      [length, 0]
    ],
    [
      [0, 1],
      [1, 0]
    ],
    undefined,
    true,
    'none'
  );
});
