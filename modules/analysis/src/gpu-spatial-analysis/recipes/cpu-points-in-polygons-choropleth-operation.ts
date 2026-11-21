import {resolveChoroplethOptions} from './choropleth-options';
import {snapshotRecipeGeometryRows} from './recipe-utils';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {CPUSpatialJoinGeometry} from '../spatial-join/cpu-feature-join';
import {
  computePointInPolygonJoinOnCPU,
  computePointInPolygonCountsOnCPU
} from '../spatial-join/cpu-point-in-polygon-join-operation';
import {
  createPointJoinCPUOperation,
  readPointJoinGeometry,
  snapshotPointJoinGeometry
} from '../spatial-join/point-join-operation-utils';
import {
  createPointsInPolygonsChoroplethOperation,
  type PointsInPolygonsChoroplethOperationProps
} from './points-in-polygons-choropleth-operation';

/** Aggregation and completeness options, shared by both backend choices. */
export type CPUPointsInPolygonsChoroplethOptions = Pick<
  PointsInPolygonsChoroplethOperationProps,
  'statistic' | 'backend' | 'boundaryRule' | 'candidateCapacity' | 'leafCapacity'
>;
/**
 * Aggregates one value per feature in every matched polygon for multipart input, and uses the
 * lowest polygon row for ordinary single-point assignments, preserving recipe behavior.
 * What differs from the GPU recipe:
 * - Packed planar aggregation only, before optional class-break/color and longitude-latitude
 *   preparation stages; those remain composable GPU contributors, not silently omitted options.
 * - Uses the exact CPU point join and compensated float64 sums/means, rounded once at output. CPU orientation
 *   has no dynamic-range uncertainty; malformed polygon hierarchies throw (see the CPU join).
 * - On candidate loss the retained prefix uses unsorted polygon rows. The recipe may spatially
 *   sort larger polygon sets; either form flags incomplete counts/statistics with overflow=1.
 * - Single-point zonal and group backends share the same CPU aggregation arithmetic.
 */
export function computePointsInPolygonsChoroplethOnCPU(
  input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; values?: Float32Array},
  options: CPUPointsInPolygonsChoroplethOptions
) {
  const id = 'CPUPointsInPolygonsChoropleth';
  options = {
    statistic: options.statistic,
    backend: options.backend,
    boundaryRule: options.boundaryRule,
    candidateCapacity: options.candidateCapacity,
    leafCapacity: options.leafCapacity
  };
  const points = snapshotChoroplethGeometry(input.points);
  const polygons = snapshotChoroplethGeometry(input.polygons);
  const {statistic} = resolveChoroplethOptions(id, options);
  const queryCount = points.featureOffsets
    ? points.featureOffsets.length - 1
    : points.positions.length / 2;
  validateRequirement(
    id,
    statistic === 'count' || Boolean(input.values),
    'non-count statistic requires values'
  );
  validateRequirement(
    id,
    !input.values || input.values.length === queryCount,
    'values needs one row per point feature'
  );
  if (statistic === 'count') {
    const join = computePointInPolygonCountsOnCPU(
      {points, polygons},
      {...options, matchCapacity: queryCount}
    );
    const counts = points.featureOffsets
      ? join.polygonMembershipCounts
      : join.polygonAssignmentCounts;
    return {
      counts,
      featureValues: counts,
      overflow: join.candidateOverflow,
      uncertainCount: join.uncertainCount
    };
  }
  const multipartJoin = points.featureOffsets
    ? computePointInPolygonJoinOnCPU({points, polygons}, {...options, matchCapacity: queryCount})
    : undefined;
  const join =
    multipartJoin ??
    computePointInPolygonCountsOnCPU({points, polygons}, {...options, matchCapacity: queryCount});
  const counts = multipartJoin ? join.polygonMembershipCounts : join.polygonAssignmentCounts;
  const polygonCount = counts.length;
  const featureValues = new Float32Array(polygonCount);
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    let total = 0,
      compensation = 0,
      validCount = 0,
      minimum = Infinity,
      maximum = -Infinity;
    for (let feature = 0; feature < queryCount; feature++) {
      const matches = multipartJoin
        ? multipartJoin.featurePolygonMembership[feature * polygonCount + polygon] !== 0
        : join.pointFeatureRows[feature] === polygon;
      const value = input.values![feature];
      if (matches && Number.isFinite(value)) {
        const next = total + value;
        compensation +=
          Math.abs(total) >= Math.abs(value) ? total - next + value : value - next + total;
        total = next;
        validCount++;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
    }
    total += compensation;
    featureValues[polygon] =
      statistic === 'sum'
        ? total
        : validCount === 0
          ? NaN
          : statistic === 'mean'
            ? total / validCount
            : statistic === 'minimum'
              ? minimum
              : maximum;
  }
  return {
    counts,
    featureValues,
    overflow: join.candidateOverflow,
    uncertainCount: join.uncertainCount
  };
}
/** Aggregate rows are feature rows, independently of caller identities or property ownership. */
function snapshotChoroplethGeometry(geometry: CPUSpatialJoinGeometry): CPUSpatialJoinGeometry {
  return snapshotConstructionFields(geometry, [
    'kind',
    'positions',
    'featureOffsets',
    'lineOffsets',
    'polygonOffsets',
    'ringOffsets'
  ]);
}

/** CPU operation matching the GPU recipe aggregation adapter. */
export function createCPUPointsInPolygonsChoroplethOperation(
  input: PointsInPolygonsChoroplethOperationProps
) {
  const id = input.id ?? 'points-in-polygons-choropleth-operation';
  const points = snapshotPointJoinGeometry(id, snapshotRecipeGeometryRows(input.points)),
    polygons = snapshotPointJoinGeometry(id, snapshotRecipeGeometryRows(input.polygons));
  const valueInput = input.values;
  const values = valueInput ? Object.freeze({...valueInput}) : undefined;
  const hasValues = Boolean(values);
  const options = Object.freeze({
    statistic: input.statistic,
    backend: input.backend,
    boundaryRule: input.boundaryRule,
    candidateCapacity: input.candidateCapacity,
    leafCapacity: input.leafCapacity
  });
  const operation = createPointsInPolygonsChoroplethOperation({
    id,
    points,
    polygons,
    values,
    ...options,
    output: Object.freeze({...input.output})
  });
  return createPointJoinCPUOperation(operation, inputs =>
    computePointsInPolygonsChoroplethOnCPU(
      {
        points: readPointJoinGeometry(operation.id, 'points', points, inputs),
        polygons: readPointJoinGeometry(operation.id, 'polygons', polygons, inputs),
        values: hasValues ? getCPUOperationFloat32Rows(operation.id, inputs, 'values') : undefined
      },
      options
    )
  );
}
