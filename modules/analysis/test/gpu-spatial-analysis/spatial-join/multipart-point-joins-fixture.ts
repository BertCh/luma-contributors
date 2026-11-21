// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {expect} from 'vitest';
import {
  GPUPointInPolygonJoin,
  type GPUPointInPolygonBoundaryRule
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {
  addPointsInPolygonsChoroplethRecipe,
  type GPUChoroplethStatistic
} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import type {FixtureInput, FixtureOutput} from '../outline-geometry/geometry-fixture';

const pointFeatures = [
  [
    [0.5, 0.5],
    [5, 1]
  ],
  [
    [4, 2],
    [4, 3]
  ],
  [[20, 20]],
  [],
  [
    [1.5, 1.5],
    [0.5, 3]
  ],
  [
    [-2, 2],
    [10, 2]
  ],
  [
    [2, 5],
    [5, 5]
  ],
  [[4, 4]],
  [[12, 2]],
  [[0, 0]],
  [
    [2, 2],
    [3, 3],
    [3, 3]
  ],
  [[1, 1]],
  [
    [6, 1],
    [6, 7],
    [30, 30]
  ]
];
const lineFeatures = [
  [
    [
      [-3, 2.5],
      [11, 2.5]
    ],
    [
      [1, 3],
      [3, 3]
    ]
  ],
  [
    [
      [20, 20],
      [21, 21]
    ]
  ],
  [],
  [
    [
      [0.5, 0.5],
      [0.5, 3.5]
    ],
    [
      [4.5, 0.5],
      [7.5, 0.5]
    ]
  ],
  [
    [
      [1.5, 0.5],
      [1.5, 3.5]
    ]
  ],
  [
    [
      [-2, 1],
      [-2, 3]
    ]
  ],
  [
    [
      [9, 1],
      [11, 3]
    ]
  ],
  [
    [
      [1, 5],
      [3, 7]
    ],
    [
      [5, 5],
      [7, 7]
    ]
  ],
  [
    [
      [3, 2],
      [5, 2]
    ],
    [
      [7, 2],
      [9, 2]
    ]
  ],
  [
    [
      [2, 3],
      [2, 5]
    ]
  ],
  [
    [
      [6, 3],
      [6, 7]
    ]
  ],
  [
    [
      [-5, 1],
      [-3, 1]
    ]
  ],
  [
    [
      [0.5, 0.5],
      [3, 0.5]
    ],
    [
      [0.5, 3.5],
      [3, 3.5]
    ]
  ]
];
const rings = [
  [
    [0, 0],
    [4, 0],
    [4, 4],
    [0, 4]
  ],
  [
    [1, 1],
    [2, 1],
    [2, 2],
    [1, 2]
  ],
  [
    [4, 0],
    [8, 0],
    [8, 4],
    [4, 4]
  ],
  [
    [8, 0],
    [12, 0],
    [12, 4],
    [8, 4]
  ],
  [
    [0, 4],
    [4, 4],
    [4, 8],
    [0, 8]
  ],
  [
    [-4, 0],
    [0, 0],
    [0, 4],
    [-4, 4]
  ],
  [
    [4, 4],
    [8, 4],
    [8, 8],
    [4, 8]
  ]
];
const offsets = (groups: unknown[][]) =>
  groups.reduce<number[]>(
    (result, group) => [...result, result[result.length - 1] + group.length],
    [0]
  );
const integers = (values: number[]): FixtureInput => ({
  format: 'uint32',
  values: Uint32Array.from(values)
});
const positions = (values: number[]): FixtureInput => ({
  format: 'float32x2',
  values: Float32Array.from(values)
});
const floats = (values: number[]): FixtureInput => ({
  format: 'float32',
  values: Float32Array.from(values)
});
export const multipartInputs: Record<string, FixtureInput> = {
  points: positions(pointFeatures.flat(2)),
  pointOffsets: integers(offsets(pointFeatures)),
  sourceIds: integers([90, 80, 70, 60, 50, 40, 30, 20, 10, 100, 110, 120, 130]),
  values: floats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
  lines: positions(lineFeatures.flat(3)),
  lineOffsets: integers(offsets(lineFeatures.flat())),
  lineFeatures: integers(offsets(lineFeatures)),
  weights: floats([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
  polygonPositions: positions(rings.flat(2)),
  polygonFeatures: integers([0, 1, 2, 3, 4, 5, 6]),
  polygonOffsets: integers([0, 2, 3, 4, 5, 6, 7]),
  ringOffsets: integers(offsets(rings)),
  polygonIds: integers([600, 100, 500, 300, 200, 400]),
  starts: positions([0, 0, -4, 6]),
  ends: positions([12, 0, 8, 6]),
  radius: floats([1])
};
type Views = Record<string, GraphDataView>;
function initializeImportedOutputs(output: Views): void {
  // Nonzero sentinels make an unwritten zero status or empty-feature mask fail the oracle.
  for (const view of Object.values(output)) {
    const buffer = view.buffer.defaultBuffer as Buffer | undefined;
    if (buffer) buffer.write(new Uint32Array(view.length).fill(0xdeadbeef), view.byteOffset);
  }
}
const uint32 = (length: number): FixtureOutput => ({format: 'uint32', length});
const float32 = (length: number): FixtureOutput => ({format: 'float32', length});
export const joinOutputs = {
  assignment: uint32(13),
  counts: uint32(6),
  membership: uint32(78),
  ids: uint32(13),
  count: uint32(1),
  requiredCount: uint32(1),
  matchOverflow: uint32(1),
  matchCandidateOverflow: uint32(1),
  truncated: uint32(13),
  overflow: uint32(1),
  candidateOverflow: uint32(1),
  candidateCount: uint32(1),
  uncertainCount: uint32(1)
};
export const bufferOutputs = {
  mask: uint32(13),
  rows: uint32(13),
  distances: float32(13),
  ids: uint32(13),
  count: uint32(1),
  requiredCount: uint32(1),
  matchOverflow: uint32(1),
  matchCandidateOverflow: uint32(1),
  truncated: uint32(13),
  overflow: uint32(1),
  candidateOverflow: uint32(1),
  draw: uint32(1)
};
export const lineOutputs = {
  lengths: float32(6),
  weighted: float32(6),
  segments: uint32(6),
  overflow: uint32(1),
  candidateOverflow: uint32(1)
};
export const recipeOutputs = {
  counts: uint32(6),
  sums: float32(6),
  overflow: uint32(1),
  uncertainCount: uint32(1)
};
const points = (input: Views) => ({
  kind: 'points' as const,
  positions: input['points'] as GraphDataView<'float32x2'>,
  featureOffsets: input['pointOffsets'] as GraphDataView<'uint32'>,
  sourceIds: input['sourceIds'] as GraphDataView<'uint32'>
});
const polygons = (input: Views) => ({
  kind: 'polygons' as const,
  positions: input['polygonPositions'] as GraphDataView<'float32x2'>,
  featureOffsets: input['polygonFeatures'] as GraphDataView<'uint32'>,
  polygonOffsets: input['polygonOffsets'] as GraphDataView<'uint32'>,
  ringOffsets: input['ringOffsets'] as GraphDataView<'uint32'>
});
const compact = (output: Views) => ({
  ids: output['ids'] as GraphDataView<'uint32'>,
  status: {
    count: output['count'] as GraphDataView<'uint32'>,
    requiredCount: output['requiredCount'] as GraphDataView<'uint32'>,
    overflow: output['matchOverflow'] as GraphDataView<'uint32'>,
    candidateOverflow: output['matchCandidateOverflow'] as GraphDataView<'uint32'>
  }
});
export function createMultipartJoin(
  input: Views,
  output: Views,
  rule: GPUPointInPolygonBoundaryRule = 'closed',
  capacity = 128,
  leafCapacity = 8
) {
  initializeImportedOutputs(output);
  return new GPUPointInPolygonJoin({
    points: points(input),
    polygons: {...polygons(input), sourceIds: input['polygonIds'] as GraphDataView<'uint32'>},
    boundaryRule: rule,
    candidateCapacity: capacity,
    candidateTileCapacity: Math.min(3, capacity),
    leafCapacity,
    spatialSort: false,
    querySort: false,
    edgeSlabs: false,
    output: {
      pointFeatureIds: output['assignment'] as GraphDataView<'uint32'>,
      polygonMembershipCounts: output['counts'] as GraphDataView<'uint32'>,
      featurePolygonMembership: output['membership'] as GraphDataView<'uint32'>,
      candidateTruncated: output['truncated'] as GraphDataView<'uint32'>,
      matches: compact(output),
      status: {
        overflow: output['overflow'] as GraphDataView<'uint32'>,
        candidateOverflow: output['candidateOverflow'] as GraphDataView<'uint32'>,
        candidateCount: output['candidateCount'] as GraphDataView<'uint32'>,
        uncertainCount: output['uncertainCount'] as GraphDataView<'uint32'>
      }
    }
  });
}
export function createMultipartBuffer(input: Views, output: Views, leafCapacity = 2) {
  initializeImportedOutputs(output);
  return new GPUBufferSelection({
    points: points(input),
    features: {
      kind: 'segments',
      starts: input['starts'] as GraphDataView<'float32x2'>,
      ends: input['ends'] as GraphDataView<'float32x2'>
    },
    radius: input['radius'] as GraphDataView<'float32'>,
    candidateCapacity: 128,
    leafCapacity,
    spatialSort: false,
    querySort: false,
    output: {
      outputMask: output['mask'] as GraphDataView<'uint32'>,
      nearestFeatureRows: output['rows'] as GraphDataView<'uint32'>,
      distances: output['distances'] as GraphDataView<'float32'>,
      matches: compact(output),
      drawInstanceCount: output['draw'] as GraphDataView<'uint32'>,
      candidateTruncated: output['truncated'] as GraphDataView<'uint32'>,
      status: {
        overflow: output['overflow'] as GraphDataView<'uint32'>,
        candidateOverflow: output['candidateOverflow'] as GraphDataView<'uint32'>
      }
    }
  });
}
export function createMultipartLines(input: Views, output: Views, pairCapacity = 128) {
  initializeImportedOutputs(output);
  return new GPULineLengthPerPolygon({
    positions: input['lines'] as GraphDataView<'float32x2'>,
    lineOffsets: input['lineOffsets'] as GraphDataView<'uint32'>,
    featureOffsets: input['lineFeatures'] as GraphDataView<'uint32'>,
    lineWeights: input['weights'] as GraphDataView<'float32'>,
    polygons: polygons(input),
    pairCapacity,
    output: {
      clippedLengths: output['lengths'] as GraphDataView<'float32'>,
      weightedLengths: output['weighted'] as GraphDataView<'float32'>,
      segmentCounts: output['segments'] as GraphDataView<'uint32'>,
      status: {
        overflow: output['overflow'] as GraphDataView<'uint32'>,
        candidateOverflow: output['candidateOverflow'] as GraphDataView<'uint32'>
      }
    }
  });
}
export function createMultipartRecipe(
  graph: GPUCommandGraph,
  input: Views,
  output: Views,
  backend: 'group' | 'zonal' = 'zonal',
  statistic: GPUChoroplethStatistic = 'sum'
) {
  initializeImportedOutputs(output);
  return addPointsInPolygonsChoroplethRecipe(graph, {
    points: points(input),
    polygons: {...polygons(input), candidateCapacity: 128, boundaryRule: 'closed'},
    values: input['values'] as GraphDataView<'float32'>,
    statistic,
    backend,
    output: {
      counts: output['counts'] as GraphDataView<'uint32'>,
      ...(statistic !== 'count' ? {featureValues: output['sums'] as GraphDataView<'float32'>} : {}),
      status: {
        overflow: output['overflow'] as GraphDataView<'uint32'>,
        uncertainCount: output['uncertainCount'] as GraphDataView<'uint32'>
      }
    }
  });
}
// Literal Shapely results from multipart-point-joins.oracle.py. Repeated parts count once.
const noFeature = 4294967295;
export const polygonMatches = {
  closed: [
    [0, 1],
    [0, 1],
    [],
    [],
    [0],
    [2, 4],
    [3, 5],
    [0, 1, 3, 5],
    [2],
    [0, 4],
    [0],
    [0],
    [1, 5]
  ],
  open: [[0, 1], [], [], [], [0], [2, 4], [3, 5], [], [], [], [0], [], [1, 5]],
  'half-open': [[0, 1], [1], [], [], [0], [2, 4], [3, 5], [5], [], [0], [0], [], [1, 5]]
};
export function assertMultipartJoin(
  result: Record<string, number[]>,
  rule: GPUPointInPolygonBoundaryRule
) {
  const matches = polygonMatches[rule];
  const polygonIds = [600, 100, 500, 300, 200, 400];
  expect(result['assignment']).toEqual(
    matches.map(rows => (rows.length ? polygonIds[rows[0]] : noFeature))
  );
  expect(result['membership']).toEqual(
    matches.flatMap(rows => Array.from({length: 6}, (_, polygon) => Number(rows.includes(polygon))))
  );
  expect(result['counts']).toEqual(
    rule === 'closed'
      ? [7, 4, 2, 2, 2, 3]
      : rule === 'open'
        ? [3, 2, 1, 1, 1, 2]
        : [4, 3, 1, 1, 1, 3]
  );
  const ids = [90, 80, 70, 60, 50, 40, 30, 20, 10, 100, 110, 120, 130].filter(
    (_, row) => matches[row].length
  );
  expect(result['ids'].slice(0, ids.length)).toEqual(ids);
  expect(result['count']).toEqual([ids.length]);
  expect(result['requiredCount']).toEqual([ids.length]);
  expect(result['truncated']).toEqual(Array(13).fill(0));
  for (const field of [
    'overflow',
    'candidateOverflow',
    'matchOverflow',
    'matchCandidateOverflow',
    'uncertainCount'
  ])
    expect(result[field], field).toEqual([0]);
  // Bounding-box candidates are counted before exact classification (hole and boundary included).
  expect(result['candidateCount']).toEqual([25]);
}
export function assertMultipartBuffer(result: Record<string, number[]>) {
  expect(result['mask']).toEqual([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]);
  expect(result['rows']).toEqual([
    0,
    noFeature,
    noFeature,
    noFeature,
    noFeature,
    noFeature,
    1,
    noFeature,
    noFeature,
    0,
    noFeature,
    0,
    0
  ]);
  expect(result['distances']).toEqual([0.5, -1, -1, -1, -1, -1, 1, -1, -1, 0, -1, 1, 1]);
  expect(result['ids'].slice(0, 5)).toEqual([90, 30, 100, 120, 130]);
  for (const field of ['count', 'requiredCount', 'draw']) expect(result[field], field).toEqual([5]);
  expect(result['truncated']).toEqual(Array(13).fill(0));
  for (const field of ['overflow', 'candidateOverflow', 'matchOverflow', 'matchCandidateOverflow'])
    expect(result[field], field).toEqual([0]);
}
export function assertMultipartLines(result: Record<string, number[]>) {
  // Float32 clipping and sorted summation: 3e-5 absolute tolerance at lengths below 131.
  for (const [name, expected] of Object.entries({
    lengths: [18, 10, 6.82842712474619, 3.8284271247461903, 6, 5.82842712474619],
    weighted: [130, 55, 38.62741699796952, 36.45584412271572, 33, 61.45584412271572]
  })) {
    expect(result[name].length).toBe(6);
    for (let row = 0; row < 6; row++) {
      expect(Number.isFinite(result[name][row])).toBe(true);
      expect(Math.abs(result[name][row] - expected[row])).toBeLessThan(3e-5);
    }
  }
  expect(result['segments']).toEqual([8, 5, 3, 2, 3, 2]);
  expect(result['overflow']).toEqual([0]);
  expect(result['candidateOverflow']).toEqual([0]);
}
export function assertMultipartRecipe(
  result: Record<string, number[]>,
  statistic: GPUChoroplethStatistic = 'sum'
) {
  expect(result['counts']).toEqual([7, 4, 2, 2, 2, 3]);
  if (statistic !== 'count') {
    // Shapely memberships and values 1..13; means divide the literal sums by literal counts.
    const expected =
      statistic === 'sum'
        ? [49, 24, 15, 15, 16, 28]
        : statistic === 'mean'
          ? [7, 6, 7.5, 7.5, 8, 28 / 3]
          : statistic === 'minimum'
            ? [1, 1, 6, 7, 6, 7]
            : [12, 13, 9, 8, 10, 13];
    for (let row = 0; row < 6; row++) {
      expect(Number.isFinite(result['sums'][row])).toBe(true);
      expect(Math.abs(result['sums'][row] - expected[row])).toBeLessThan(1e-6);
    }
  }
  expect(result['overflow']).toEqual([0]);
  expect(result['uncertainCount']).toEqual([0]);
}
