// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {multipartInputs} from './multipart-point-joins-fixture';
import type {BufferSelectionOperationProps} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import type {PointInPolygonJoinOperationProps} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join-operation';
import type {LineLengthPerPolygonOperationProps} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon-operation';
import type {PointsInPolygonsChoroplethOperationProps} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-operation';
import type {FeatureJoinGeometryValues} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';

export const value = (name: string) => ({
  value: name,
  length:
    multipartInputs[name].values.length / (multipartInputs[name].format === 'float32x2' ? 2 : 1)
});
export const pointGeometry: FeatureJoinGeometryValues = {
  kind: 'points',
  positions: value('points'),
  featureOffsets: value('pointOffsets'),
  sourceIds: value('sourceIds')
};
export const polygonGeometry: FeatureJoinGeometryValues = {
  kind: 'polygons',
  positions: value('polygonPositions'),
  featureOffsets: value('polygonFeatures'),
  polygonOffsets: value('polygonOffsets'),
  ringOffsets: value('ringOffsets'),
  sourceIds: value('polygonIds')
};
export const lineOperationProps: LineLengthPerPolygonOperationProps = {
  lines: {
    kind: 'lines',
    positions: value('lines'),
    featureOffsets: value('lineFeatures'),
    lineOffsets: value('lineOffsets')
  },
  polygons: polygonGeometry,
  lineWeights: value('weights'),
  pairCapacity: 128,
  maximumCrossingCount: 64,
  output: {
    clippedLengths: 'lengths',
    weightedLengths: 'weighted',
    segmentCounts: 'segments',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow'
  }
};
export const choroplethOperationProps: PointsInPolygonsChoroplethOperationProps = {
  points: pointGeometry,
  polygons: polygonGeometry,
  values: value('values'),
  statistic: 'sum',
  backend: 'group',
  boundaryRule: 'closed',
  candidateCapacity: 128,
  leafCapacity: 8,
  output: {
    counts: 'counts',
    featureValues: 'sums',
    overflow: 'overflow',
    uncertainCount: 'uncertainCount'
  }
};
export const pointJoinOperationProps: PointInPolygonJoinOperationProps = {
  points: pointGeometry,
  polygons: polygonGeometry,
  boundaryRule: 'closed',
  candidateCapacity: 128,
  candidateTileCapacity: 3,
  leafCapacity: 8,
  matchCapacity: 13,
  output: {
    pointFeatureIds: 'assignment',
    polygonMembershipCounts: 'counts',
    featurePolygonMembership: 'membership',
    candidateTruncated: 'truncated',
    ids: 'ids',
    count: 'count',
    requiredCount: 'requiredCount',
    matchOverflow: 'matchOverflow',
    matchCandidateOverflow: 'matchCandidateOverflow',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow',
    candidateCount: 'candidateCount',
    uncertainCount: 'uncertainCount'
  }
};
export const bufferOperationProps: BufferSelectionOperationProps = {
  points: pointGeometry,
  features: {kind: 'segments', starts: value('starts'), ends: value('ends')},
  radius: value('radius'),
  candidateCapacity: 128,
  leafCapacity: 2,
  matchCapacity: 13,
  output: {
    outputMask: 'mask',
    distances: 'distances',
    nearestFeatureRows: 'rows',
    candidateTruncated: 'truncated',
    ids: 'ids',
    count: 'count',
    requiredCount: 'requiredCount',
    matchOverflow: 'matchOverflow',
    matchCandidateOverflow: 'matchCandidateOverflow',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow',
    drawInstanceCount: 'draw'
  }
};
export function getOperationCPUInputs(
  inputs: Readonly<Record<string, string>>
): Record<string, GPUCPUVector> {
  return Object.fromEntries(
    Object.entries(inputs).map(([port, name]) => {
      const definition = multipartInputs[name],
        values = definition.values,
        length = value(name).length;
      return [
        port,
        {format: definition.format, length, chunks: [{values, length}], pack: () => values}
      ];
    })
  );
}
export function getOperationResult(result: unknown): Record<string, number[]> {
  return Object.fromEntries(
    Object.entries(result as Record<string, Uint32Array | Float32Array>).map(([port, array]) => [
      port,
      Array.from(array)
    ])
  );
}
