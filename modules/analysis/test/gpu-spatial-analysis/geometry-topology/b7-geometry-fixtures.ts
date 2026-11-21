// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView, type GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {createPolygonizeDanglePruningNodes} from '../../../src/gpu-spatial-analysis/geometry-topology/polygonize-dangle-pruning';
import type {OracleAllocate} from './noding-polygonize-output-oracles-fixtures';

export type CreateGeometryFixture = (
  graph: GPUCommandGraph,
  allocate: OracleAllocate
) => {
  output: object;
  nodes: readonly GPUCommandNode[];
  expected: Record<string, number[]>;
  reciprocalExpected?: Record<string, number[]>;
};

export const NODING_PARAMETER_CASES = [
  'tiny',
  'ordinary',
  'horizontal',
  'tiny-horizontal',
  'exact-dyadic',
  'close-cuts',
  'three-fifths'
] as const;

export function createNodingParameterCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  geometry: (typeof NODING_PARAMETER_CASES)[number]
) {
  // Independent /usr/bin/python3 Fraction and struct.pack('f') calculations.
  // Ordinary GPU division may use direct division or a rounded reciprocal.
  // Pin the exact pre-stream outputs of each form, including its inexact cases.
  const cases = {
    tiny: {length: 2 ** -63, cuts: [2 ** -88], parameters: [2.9802322387695312e-8]},
    ordinary: {length: 12, cuts: [9], parameters: [0.75]},
    horizontal: {
      length: 11,
      cuts: [8.25],
      parameters: [0.75],
      reciprocal: [0.7499999403953552],
      reciprocalEnd: 0.9999999403953552
    },
    'tiny-horizontal': {length: 11 * 2 ** -80, cuts: [8.25 * 2 ** -80], parameters: [0.75]},
    'exact-dyadic': {
      length: 529,
      cuts: [504.203125],
      parameters: [0.9531249403953552],
      reciprocal: [0.953125]
    },
    'close-cuts': {
      length: 11,
      cuts: [2.750000476837158, 2.7500007152557373],
      parameters: [0.2500000596046448, 0.2500000596046448],
      reciprocal: [0.2500000298023224, 0.2500000596046448],
      reciprocalEnd: 0.9999999403953552
    },
    'three-fifths': {
      length: 5,
      cuts: [3],
      parameters: [0.6000000238418579],
      reciprocal: [0.5999999642372131]
    }
  } satisfies Record<
    (typeof NODING_PARAMETER_CASES)[number],
    {
      length: number;
      cuts: number[];
      parameters: number[];
      reciprocal?: number[];
      reciprocalEnd?: number;
    }
  >;
  const {length, cuts, parameters} = cases[geometry];
  const diagonal = geometry === 'ordinary';
  const crossingEdges = cuts.flatMap(cut => [cut, 0, cut, length]);
  const coordinates = diagonal ? [0, 0, 12, 4, 9, 3, 9, 5] : [0, 0, length, 0, ...crossingEdges];
  const edgeCount = cuts.length + 1;
  const outputCount = cuts.length + edgeCount;
  const parameterViews = (splits: number[], end = 1) => ({
    sourceStartParameters: [0, ...splits, ...cuts.map(() => 0)],
    sourceEndParameters: [...splits, end, ...cuts.map(() => end)]
  });
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const output = {
    endpoints: allocate('endpoints', 'float32x4', outputCount),
    sourceFeatureIds: allocate('features', 'uint32', outputCount),
    sourceRingIds: allocate('rings', 'uint32', outputCount),
    sourceEdgeIds: allocate('edges', 'uint32', outputCount),
    sourceStartParameters: allocate('starts', 'float32', outputCount),
    sourceEndParameters: allocate('ends', 'float32', outputCount),
    status: {
      count: scalar('count'),
      requiredCount: scalar('required'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidates'),
      uncertainCount: scalar('uncertain')
    }
  };
  const noding = new GPULineNoding({
    lines: {
      kind: 'lines',
      positions: allocate('positions', 'float32x2', 2 * edgeCount, coordinates),
      lineOffsets: allocate(
        'offsets',
        'uint32',
        edgeCount + 1,
        Array.from({length: edgeCount + 1}, (_, row) => 2 * row)
      )
    },
    intersectionCapacity: 8,
    output
  });
  const endpoints = diagonal
    ? [0, 0, 9, 3, 9, 3, 12, 4, 9, 3, 9, 5]
    : [
        ...[0, ...cuts].flatMap((first, row) => [first, 0, [...cuts, length][row], 0]),
        ...crossingEdges
      ];
  const selected = cases[geometry];
  return {
    output,
    nodes: noding.getCommandNodes(graph),
    reciprocalExpected:
      'reciprocal' in selected
        ? parameterViews(
            selected.reciprocal,
            'reciprocalEnd' in selected ? selected.reciprocalEnd : 1
          )
        : undefined,
    expected: {
      'status.count': [outputCount],
      'status.requiredCount': [outputCount],
      'status.overflow': [0],
      'status.candidateOverflow': [0],
      'status.uncertainCount': [0],
      endpoints,
      sourceEdgeIds: [
        ...new Array<number>(edgeCount).fill(0),
        ...cuts.map((_, row) => 2 * (row + 1))
      ],
      ...parameterViews(parameters)
    }
  };
}

export function createGroupedPruningCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  squareCount: number,
  distant: boolean
) {
  const values: number[] = [];
  const groups: number[] = [];
  function addSquare(x: number, y: number, sideLength: number, group: number) {
    const corners = [
      [x, y],
      [x + sideLength, y],
      [x + sideLength, y + sideLength],
      [x, y + sideLength]
    ];
    for (let side = 0; side < 4; side++) {
      const first = corners[side];
      const last = corners[(side + 1) % 4];
      values.push(...first, ...last, ...last, ...first);
      groups.push(group, group);
    }
  }
  const center = 8 * Math.floor(squareCount / 2);
  for (let square = 0; square < squareCount; square++) {
    addSquare(8 * square - center, 8 * square - center, 1, 9);
  }
  // Four float32 steps (512 each) separate the distinct distant corners from neighboring keys.
  if (distant) addSquare(2 ** 32, 2 ** 32, 2048, 4);

  const endpointCount = groups.length;
  const endpoints = allocate('endpoints', 'float32x4', endpointCount, values);
  const candidateCounts = allocate('visits', 'uint32', endpointCount);
  const pruning = createPolygonizeDanglePruningNodes(graph, {
    id: 'grouped-pruning',
    endpoints,
    groups: allocate('groups', 'uint32', endpointCount, groups),
    count: allocate('count', 'uint32', 1, [endpointCount]),
    vertexTolerance: 1 / 64,
    candidateCounts
  });
  const faceEndpoints = allocate('face-endpoints', 'float32x4', endpointCount);
  const readback = createWGSLKernelNode(graph, {
    id: 'read-face-endpoints',
    operation: 'test',
    bindings: [
      {name: 'faces', view: pruning.endpoints, type: 'f32', access: 'read'},
      {name: 'result', view: faceEndpoints, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 4 * endpointCount,
    body: 'result[resultOffset + index] = faces[facesOffset + index];'
  });
  return {
    output: {candidateCounts, endpoints: faceEndpoints},
    nodes: [...pruning.nodes, readback],
    // Each corner has exactly two coincident directed starts and all other
    // corners in its group are at least one unit away. All squares are closed.
    expected: {candidateCounts: new Array<number>(endpointCount).fill(2), endpoints: values}
  };
}

export function createGroupedRingCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  tolerance: 'explicit' | 'derived'
) {
  // For the derived case, the largest offset is 2^15, so 2^15 * 2^-21 = 1/64.
  const distant = tolerance === 'explicit' ? 2 ** 32 : 2 ** 15 - 256;
  const distantEnd = tolerance === 'explicit' ? distant + 2048 : 2 ** 15;
  const endpoints = allocate(
    'endpoints',
    'float32x4',
    8,
    [
      [0, 0, 1 - 1 / 128, 0],
      [1 + 1 / 128, 0, 1, 1],
      [1, 1, 0, 1],
      [0, 1, 0, 0],
      [distant, distant, distantEnd, distant],
      [distantEnd, distant, distantEnd, distantEnd],
      [distantEnd, distantEnd, distant, distantEnd],
      [distant, distantEnd, distant, distant]
    ].flat()
  );
  const groups = allocate('groups', 'uint32', 8, [9, 9, 9, 9, 4, 4, 4, 4]);
  const output = {
    positions: allocate('positions', 'float32x2', 10),
    ringOffsets: allocate('ring-offsets', 'uint32', 3),
    ringGroups: allocate('ring-groups', 'uint32', 2),
    segmentRings: allocate('segment-rings', 'uint32', 8),
    status: {
      count: allocate('count', 'uint32', 1),
      overflow: allocate('overflow', 'uint32', 1),
      openSegmentCount: allocate('open', 'uint32', 1)
    }
  };
  const assembly = new GPUSegmentRingAssembly({
    endpoints,
    groups,
    vertexTolerance: tolerance === 'explicit' ? 1 / 64 : undefined,
    output
  });
  // The first ring has one gap of exactly 1/64 across x=1. The other joins
  // are exact. Each group thus contributes one four-segment closed ring.
  return {
    output,
    nodes: assembly.getCommandNodes(graph),
    expected: {
      'status.count': [2],
      'status.overflow': [0],
      'status.openSegmentCount': [0],
      ringOffsets: [0, 5, 10],
      ringGroups: [9, 4],
      segmentRings: [0, 0, 0, 0, 1, 1, 1, 1]
    }
  };
}

export function createRingBucketCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  distant: boolean
) {
  const localCount = 32;
  const coordinates = Array.from({length: localCount}, (_, row) => [row * 8, 0, row * 8 + 1, 0]);
  const groups = new Array<number>(localCount).fill(9);
  if (distant) {
    coordinates.push([2 ** 32, 0, 2 ** 32 + 2048, 0]);
    groups.push(4);
  }
  const segmentCount = groups.length;
  const assembly = new GPUSegmentRingAssembly({
    id: 'ring-buckets',
    endpoints: allocate('endpoints', 'float32x4', segmentCount, coordinates.flat()),
    groups: allocate('groups', 'uint32', segmentCount, groups),
    vertexTolerance: 1 / 64,
    output: {
      ringOffsets: allocate('offsets', 'uint32', 2),
      positions: allocate('positions', 'float32x2', 1),
      status: {count: allocate('count', 'uint32', 1), overflow: allocate('overflow', 'uint32', 1)}
    }
  });
  const nodes = assembly.getCommandNodes(graph);
  const hashIndex = nodes.findIndex(node => node.id === 'ring-buckets-hash');
  const keyUse = nodes[hashIndex]?.resources?.find(
    use =>
      'buffer' in use &&
      use.buffer instanceof GraphDataView &&
      use.buffer.buffer.id === 'ring-buckets-keys'
  );
  if (!keyUse || !('buffer' in keyUse) || !(keyUse.buffer instanceof GraphDataView)) {
    throw new Error('Ring hash kernel did not declare its key view');
  }
  // Count occupants of each local hash bucket using the assembler's actual keys.
  // The extent-based implementation put all 32 local starts in the same bucket.
  const occupancy = allocate('occupancy', 'uint32', localCount);
  const countBuckets = createWGSLKernelNode(graph, {
    id: 'read-ring-buckets',
    operation: 'test',
    bindings: [
      {name: 'keys', view: keyUse.buffer, type: 'u32', access: 'read'},
      {name: 'occupancy', view: occupancy, type: 'u32', access: 'read_write'}
    ],
    invocationCount: localCount,
    body: `let key = keys[keysOffset + index];
  var count = 0u;
  for (var row = 0u; row < ${segmentCount}u; row++) {
    if (keys[keysOffset + row] == key) { count++; }
  }
  occupancy[occupancyOffset + index] = count;`
  });
  return {
    nodes: [...nodes.slice(0, hashIndex + 1), countBuckets],
    output: {occupancy},
    expected: {occupancy: new Array<number>(localCount).fill(1)}
  };
}

export function createRingExtentCase(graph: GPUCommandGraph, allocate: OracleAllocate) {
  const extentEndpoint = 2 ** 127;
  const height = 1 / 4;
  const corners = [
    [-extentEndpoint, 0],
    [0, 0],
    [extentEndpoint, 0],
    [extentEndpoint, height],
    [0, height],
    [-extentEndpoint, height]
  ];
  const endpoints = allocate(
    'endpoints',
    'float32x4',
    corners.length,
    corners.flatMap((corner, row) => [...corner, ...corners[(row + 1) % corners.length]])
  );
  const output = {
    positions: allocate('positions', 'float32x2', 7),
    ringOffsets: allocate('ring-offsets', 'uint32', 2),
    status: {
      count: allocate('count', 'uint32', 1),
      overflow: allocate('overflow', 'uint32', 1),
      openSegmentCount: allocate('open', 'uint32', 1)
    }
  };
  const assembly = new GPUSegmentRingAssembly({
    endpoints,
    vertexTolerance: 1 / 64,
    output
  });
  // The rectangle's area is 2^126, but its relative x span exceeds float32.
  // Until the area path handles that span, the returned ring must signal overflow.
  return {
    output,
    nodes: assembly.getCommandNodes(graph),
    expected: {
      'status.overflow': [1]
    }
  };
}
