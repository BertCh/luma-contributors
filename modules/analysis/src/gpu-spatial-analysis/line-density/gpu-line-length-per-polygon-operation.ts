// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPULineLengthPerPolygon} from './gpu-line-length-per-polygon';
import type {
  FeatureJoinGeometryValues,
  FeatureJoinValue
} from '../spatial-join/feature-join-operation';
import type {GPUSpatialJoinGeometry} from '../spatial-join/spatial-join-types';
import {
  snapshotPointJoinGeometry,
  getPointJoinFeatureCount,
  getPointJoinGeometryInputs,
  resolvePointJoinGeometry
} from '../spatial-join/point-join-operation-utils';

/** Matches the contributor's bounded private crossing array within WebGPU's local-array limit. @internal */
export const LINE_LENGTH_MAXIMUM_CROSSING_COUNT = 1024;

/** Same packed planar contract for GPU and CPU line-length adapters. */
export type LineLengthPerPolygonOperationProps = {
  id?: string;
  lines: FeatureJoinGeometryValues;
  polygons: FeatureJoinGeometryValues;
  lineWeights?: FeatureJoinValue;
  pairCapacity: number;
  maximumCrossingCount?: number;
  output: {
    clippedLengths: string;
    weightedLengths?: string;
    segmentCounts: string;
    overflow: string;
    candidateOverflow: string;
  };
};
/** Domain operation form, using every linestring part without joining across part boundaries. @internal */
export function createLineLengthPerPolygonOperation(
  input: LineLengthPerPolygonOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'line-length-per-polygon-operation';
  const lines = snapshotPointJoinGeometry(id, input.lines),
    polygons = snapshotPointJoinGeometry(id, input.polygons);
  validateRequirement(
    id,
    lines.kind === 'lines' && polygons.kind === 'polygons',
    'lines and polygons must have their named geometry kinds'
  );
  const lineWeights = input.lineWeights ? Object.freeze({...input.lineWeights}) : undefined;
  const requestedOutput = input.output;
  validateRequirement(
    id,
    !('lengths' in requestedOutput),
    'output.lengths was removed; use output.clippedLengths'
  );
  const output = snapshotConstructionFields(requestedOutput, [
    'clippedLengths',
    'weightedLengths',
    'segmentCounts',
    'overflow',
    'candidateOverflow'
  ]);
  validateRequirement(id, Boolean(output.clippedLengths), 'output.clippedLengths is required');
  validateRequirement(
    id,
    !lineWeights || lineWeights.length === getPointJoinFeatureCount(lines),
    'lineWeights needs one row per line feature'
  );
  validateRequirement(
    id,
    !output.weightedLengths || Boolean(lineWeights),
    'output.weightedLengths requires lineWeights'
  );
  const {pairCapacity} = input,
    maximumCrossingCount = input.maximumCrossingCount ?? 64;
  validateCapacity(id, 'pairCapacity', pairCapacity);
  validateCapacity(
    id,
    'maximumCrossingCount',
    maximumCrossingCount,
    1,
    LINE_LENGTH_MAXIMUM_CROSSING_COUNT
  );
  const polygonCount = getPointJoinFeatureCount(polygons);
  return new GPUContributorOperation({
    id,
    inputs: {
      ...getPointJoinGeometryInputs('lines', lines),
      ...getPointJoinGeometryInputs('polygons', polygons),
      ...(lineWeights ? {lineWeights: lineWeights.value} : {})
    },
    outputs: Object.fromEntries(
      Object.entries(output)
        .filter(([, value]) => value !== undefined)
        .map(([port, value]) => [
          port,
          {
            value: value!,
            format:
              port === 'clippedLengths' || port === 'weightedLengths'
                ? ('float32' as const)
                : ('uint32' as const),
            length: port === 'overflow' || port === 'candidateOverflow' ? 1 : polygonCount
          }
        ])
    ),
    createContributor: getView => {
      const geometry = resolvePointJoinGeometry(id, 'lines', lines, getView) as Extract<
        GPUSpatialJoinGeometry,
        {kind: 'lines'}
      >;
      return new GPULineLengthPerPolygon({
        ...geometry,
        id,
        polygons: resolvePointJoinGeometry(id, 'polygons', polygons, getView) as never,
        lineWeights: lineWeights ? getView('lineWeights') : undefined,
        pairCapacity,
        maximumCrossingCount,
        output: {
          clippedLengths: getView('clippedLengths'),
          weightedLengths: output.weightedLengths ? getView('weightedLengths') : undefined,
          segmentCounts: getView('segmentCounts'),
          status: {overflow: getView('overflow'), candidateOverflow: getView('candidateOverflow')}
        }
      });
    }
  });
}
