// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCPUOperation,
  GPUCPUVector,
  GPUCommandGraph,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import {createCPUBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import {createLineLengthPerPolygonOperation} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon-operation';
import {createCPULineLengthPerPolygonOperation} from '../../../src/gpu-spatial-analysis/line-density/cpu-line-length-per-polygon-operation';
import {createPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join-operation';
import {createCPUPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {createPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-operation';
import {createCPUPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/recipes/cpu-points-in-polygons-choropleth-operation';
import type {FixtureInput} from '../outline-geometry/geometry-fixture';
import {
  bufferOperationProps,
  lineOperationProps,
  pointJoinOperationProps,
  choroplethOperationProps
} from './multipart-point-operations-fixture';

const positions = (values: number[]): FixtureInput => ({
  format: 'float32x2',
  values: Float32Array.from(values)
});
const floats = (values: number[]): FixtureInput => ({
  format: 'float32',
  values: Float32Array.from(values)
});
const integers = (values: number[]): FixtureInput => ({
  format: 'uint32',
  values: Uint32Array.from(values)
});
export type ReviewCase = {
  name: string;
  inputs: Record<string, FixtureInput>;
  gpuInputs?: Record<string, FixtureInput>;
  gpu(): GPUContributorOperation;
  cpu(): GPUCPUOperation;
  expected: Record<string, number[]>;
};
/** Keep logical offsets/lengths while giving empty inputs a legal physical storage allocation. */
export function resolveReviewInputViews(
  entry: ReviewCase,
  graph: GPUCommandGraph,
  inputs: Record<string, GraphDataView>
): void {
  if (!entry.gpuInputs) return;
  if (entry.inputs['values'])
    inputs['values'] = graph.createDataView(inputs['values'].buffer, {
      format: 'float32',
      byteOffset: inputs['values'].byteOffset + 4,
      length: 2
    });
  if (entry.inputs['lines']?.values.length === 0)
    inputs['lines'] = graph.createDataView(inputs['lines'].buffer, {
      format: 'float32x2',
      length: 0
    });
}
export const adjacentPolygons = {
  polygonPositions: positions([0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 4, 0, 4, 2, 2, 2]),
  polygonFeatures: integers([0, 1, 2]),
  polygonOffsets: integers([0, 1, 2]),
  ringOffsets: integers([0, 4, 8])
};
export const holePolygon = {
  polygonPositions: positions([0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 3, 1, 3, 3, 1, 3]),
  polygonFeatures: integers([0, 1]),
  polygonOffsets: integers([0, 2]),
  ringOffsets: integers([0, 4, 8])
};
export function reviewGeometry(
  inputs: Record<string, FixtureInput>,
  kind: 'points' | 'lines' | 'polygons'
) {
  const value = (name: string) => ({
    value: name,
    length: inputs[name].values.length / (inputs[name].format === 'float32x2' ? 2 : 1)
  });
  return kind === 'polygons'
    ? {
        kind,
        positions: value('polygonPositions'),
        featureOffsets: value('polygonFeatures'),
        polygonOffsets: value('polygonOffsets'),
        ringOffsets: value('ringOffsets')
      }
    : kind === 'lines'
      ? {
          kind,
          positions: value('lines'),
          lineOffsets: value('lineOffsets'),
          featureOffsets: value('lineFeatures')
        }
      : {kind, positions: value('points'), featureOffsets: value('pointOffsets')};
}
export function reviewCPUInputs(
  operation: GPUCPUOperation,
  inputs: Record<string, FixtureInput>
): Record<string, GPUCPUVector> {
  return Object.fromEntries(
    Object.entries(operation.inputs).map(([port, name]) => {
      const {format, values} = inputs[name],
        length = values.length / (format === 'float32x2' ? 2 : 1);
      return [port, {format, length, chunks: [{values, length}], pack: () => values}];
    })
  );
}
export function createReviewCases(): ReviewCase[] {
  const cases: ReviewCase[] = [];
  // R5: the unusable edge is empty; the only usable unit horizontal edge has one bbox pair,
  // so pairCapacity 1 is complete. R7: empty sums/counts and candidate losses are all zero.
  // R6: each point is in exactly one finite vertex envelope. Only the first envelope belongs
  // to an invalid ring, hence one uncertain pair; the second polygon retains row 1/value 11.
  const bufferInputs = {
    points: positions([29, 29, -2, 5]),
    pointOffsets: integers([0, 2]),
    sourceIds: integers([91]),
    starts: positions([0, 0]),
    ends: positions([100, 100]),
    radius: floats([0])
  };
  const bufferProps = {
    ...bufferOperationProps,
    points: {...reviewGeometry(bufferInputs, 'points'), sourceIds: {value: 'sourceIds', length: 1}},
    features: {
      kind: 'segments' as const,
      starts: {value: 'starts', length: 1},
      ends: {value: 'ends', length: 1}
    },
    leafCapacity: 1,
    matchCapacity: 1
  };
  cases.push({
    name: 'R1 exact collinear zero-radius segment hit',
    inputs: bufferInputs,
    gpu: () => createBufferSelectionOperation(bufferProps),
    cpu: () => createCPUBufferSelectionOperation(bufferProps),
    expected: {
      mask: [1],
      distances: [0],
      rows: [0],
      ids: [91],
      count: [1],
      requiredCount: [1],
      draw: [1],
      overflow: [0],
      candidateOverflow: [0],
      matchOverflow: [0],
      matchCandidateOverflow: [0],
      truncated: [0]
    }
  });
  for (const entry of [
    {
      name: 'R2 shared edge belongs to both clipped intersections',
      polygons: adjacentPolygons,
      coordinates: [2, 0.5, 2, 1.5],
      offsets: [0, 2],
      features: [0, 1],
      capacity: 128,
      expected: {lengths: [1, 1], weighted: [3, 3], segments: [1, 1]}
    },
    {
      name: 'R2 hole edge belongs to the closed clipped intersection',
      polygons: holePolygon,
      coordinates: [1, 1.5, 1, 2.5],
      offsets: [0, 2],
      features: [0, 1],
      capacity: 128,
      expected: {lengths: [1], weighted: [3], segments: [1]}
    },
    {
      name: 'R5 nonfinite edge cannot consume the valid part candidate slot',
      polygons: adjacentPolygons,
      coordinates: [0.5, 0.5, NaN, NaN, 0.5, 0.5, 1.5, 0.5],
      offsets: [0, 2, 4],
      features: [0, 2],
      capacity: 1,
      expected: {lengths: [1, 0], weighted: [3, 0], segments: [1, 0]}
    },
    {
      name: 'R7 all empty multiline features clear every output',
      polygons: adjacentPolygons,
      coordinates: [],
      offsets: [0],
      features: [0, 0],
      capacity: 1,
      expected: {lengths: [0, 0], weighted: [0, 0], segments: [0, 0]}
    }
  ]) {
    const inputs = {
      ...entry.polygons,
      lines: positions(entry.coordinates),
      lineOffsets: integers(entry.offsets),
      lineFeatures: integers(entry.features),
      weights: floats([3])
    };
    const props = {
      ...lineOperationProps,
      lines: reviewGeometry(inputs, 'lines'),
      polygons: reviewGeometry(inputs, 'polygons'),
      lineWeights: {value: 'weights', length: 1},
      pairCapacity: entry.capacity
    };
    cases.push({
      name: entry.name,
      inputs,
      ...(entry.coordinates.length === 0
        ? {gpuInputs: {...inputs, lines: positions([99, 99])}}
        : {}),
      gpu: () => createLineLengthPerPolygonOperation(props),
      cpu: () => createCPULineLengthPerPolygonOperation(props),
      expected: {...entry.expected, overflow: [0], candidateOverflow: [0]}
    });
  }
  const invalidInputs = {
    ...adjacentPolygons,
    polygonPositions: positions([0, 0, 2, 0, NaN, 2, 0, 2, 2, 0, 4, 0, 4, 2, 2, 2]),
    points: positions([1, 1, 3, 1]),
    pointOffsets: integers([0, 1, 2]),
    values: floats([7, 11])
  };
  const countInputs = {
    ...adjacentPolygons,
    points: positions([0.5, 0.5, 3, 0.5, 3, 1, 10, 10]),
    pointOffsets: integers([0, 2, 3, 4])
  };
  const countProps = {
    ...pointJoinOperationProps,
    points: reviewGeometry(countInputs, 'points'),
    polygons: reviewGeometry(countInputs, 'polygons'),
    matchCapacity: 2,
    output: {...pointJoinOperationProps.output, pointMatchCounts: 'pointMatchCounts'}
  };
  cases.push({
    name: 'R8 distinct polygon match counts are per input feature',
    inputs: countInputs,
    gpu: () => createPointInPolygonJoinOperation(countProps),
    cpu: () => createCPUPointInPolygonJoinOperation(countProps),
    expected: {
      assignment: [0, 1, 4294967295],
      counts: [1, 2],
      pointMatchCounts: [2, 1, 0],
      membership: [1, 1, 0, 1, 0, 0],
      ids: [0, 1],
      count: [2],
      requiredCount: [2],
      candidateCount: [3],
      overflow: [0],
      candidateOverflow: [0],
      uncertainCount: [0],
      truncated: [0, 0, 0],
      matchOverflow: [0],
      matchCandidateOverflow: [0]
    }
  });
  const sparseProps = {
    ...countProps,
    output: {...countProps.output, featurePolygonMembership: undefined}
  };
  const {membership: ignoredMembership, ...sparseExpected} = cases[cases.length - 1].expected;
  void ignoredMembership;
  cases.push({
    name: 'R8 sparse counts omit the membership matrix',
    inputs: countInputs,
    gpu: () => createPointInPolygonJoinOperation(sparseProps),
    cpu: () => createCPUPointInPolygonJoinOperation(sparseProps),
    expected: sparseExpected
  });
  const joinProps = {
    ...pointJoinOperationProps,
    points: reviewGeometry(invalidInputs, 'points'),
    polygons: reviewGeometry(invalidInputs, 'polygons'),
    matchCapacity: 1
  };
  cases.push({
    name: 'R6 nonfinite polygon preserves the valid assignment and uncertainty',
    inputs: invalidInputs,
    gpu: () => createPointInPolygonJoinOperation(joinProps),
    cpu: () => createCPUPointInPolygonJoinOperation(joinProps),
    expected: {
      assignment: [4294967295, 1],
      counts: [0, 1],
      membership: [0, 0, 0, 1],
      ids: [1],
      count: [1],
      requiredCount: [1],
      uncertainCount: [1],
      candidateCount: [2],
      truncated: [0, 0],
      overflow: [0],
      candidateOverflow: [0],
      matchOverflow: [0],
      matchCandidateOverflow: [0]
    }
  });
  for (const backend of ['zonal', 'group'] as const) {
    const offsetInputs = {
      ...adjacentPolygons,
      points: positions([1, 1, 1, 1]),
      pointOffsets: integers([0, 1, 2]),
      values: floats([7, 11])
    };
    const offsetProps = {
      ...choroplethOperationProps,
      points: reviewGeometry(offsetInputs, 'points'),
      polygons: reviewGeometry(offsetInputs, 'polygons'),
      values: {value: 'values', length: 2},
      backend
    };
    cases.push({
      name: `R3 ${backend} packed offset values sum logical rows`,
      inputs: offsetInputs,
      gpuInputs: {...offsetInputs, values: floats([99, 7, 11, 99])},
      gpu: () => createPointsInPolygonsChoroplethOperation(offsetProps),
      cpu: () => createCPUPointsInPolygonsChoroplethOperation(offsetProps),
      expected: {counts: [2, 0], sums: [18, 0], overflow: [0], uncertainCount: [0]}
    });
    const props = {
      ...choroplethOperationProps,
      points: reviewGeometry(invalidInputs, 'points'),
      polygons: reviewGeometry(invalidInputs, 'polygons'),
      values: {value: 'values', length: 2},
      backend
    };
    cases.push({
      name: `R6 ${backend} choropleth preserves valid polygon sum`,
      inputs: invalidInputs,
      gpu: () => createPointsInPolygonsChoroplethOperation(props),
      cpu: () => createCPUPointsInPolygonsChoroplethOperation(props),
      expected: {counts: [0, 1], sums: [0, 11], uncertainCount: [1], overflow: [0]}
    });
    for (const values of [
      [32768, 1 / 1024, -32768],
      [-32768, 32768, 1 / 1024]
    ])
      for (const statistic of ['sum', 'mean'] as const) {
        const inputs = {
          ...adjacentPolygons,
          points: positions([1, 1, 1, 1, 1, 1]),
          pointOffsets: integers([0, 1, 2, 3]),
          values: floats(values)
        };
        const props = {
          ...choroplethOperationProps,
          points: reviewGeometry(inputs, 'points'),
          polygons: reviewGeometry(inputs, 'polygons'),
          values: {value: 'values', length: 3},
          backend,
          statistic
        };
        // Exact binary arithmetic: 2^15 + 2^-10 - 2^15 = 2^-10; mean rounds once to f32.
        cases.push({
          name: `R4 ${backend} ${statistic} cancellation order ${values[0]}`,
          inputs,
          gpu: () => createPointsInPolygonsChoroplethOperation(props),
          cpu: () => createCPUPointsInPolygonsChoroplethOperation(props),
          expected: {
            counts: [3, 0],
            sums: statistic === 'sum' ? [0.0009765625, 0] : [0.00032552084303461015, NaN],
            overflow: [0],
            uncertainCount: [0]
          }
        });
      }
  }
  return cases;
}
