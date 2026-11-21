import {resolveChoroplethOptions} from './choropleth-options';
import {snapshotRecipeGeometryRows} from './recipe-utils';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  addPointsInPolygonsChoroplethRecipe,
  type GPUChoroplethStatistic
} from './points-in-polygons-choropleth-recipe';
import type {
  FeatureJoinGeometryValues,
  FeatureJoinValue
} from '../spatial-join/feature-join-operation';
import type {GPUPointInPolygonBoundaryRule} from '../spatial-join/gpu-point-in-polygon-join';
import type {GPUSpatialJoinGeometry} from '../spatial-join/spatial-join-types';
import {
  snapshotPointJoinGeometry,
  getPointJoinFeatureCount,
  getPointJoinGeometryInputs,
  resolvePointJoinGeometry
} from '../spatial-join/point-join-operation-utils';

/** Packed planar aggregate stage, before optional class breaks/colors and polygon preparation. */
export type PointsInPolygonsChoroplethOperationProps = {
  id?: string;
  points: FeatureJoinGeometryValues;
  polygons: FeatureJoinGeometryValues;
  values?: FeatureJoinValue;
  statistic?: GPUChoroplethStatistic;
  backend?: 'zonal' | 'group';
  boundaryRule?: GPUPointInPolygonBoundaryRule;
  candidateCapacity: number;
  leafCapacity?: number;
  output: {counts: string; featureValues?: string; overflow: string; uncertainCount: string};
};
/** Same aggregation stages as the recipe, expressed as logical graph values. */
export function createPointsInPolygonsChoroplethOperation(
  input: PointsInPolygonsChoroplethOperationProps
): GPUContributorOperation {
  const id = input.id ?? 'points-in-polygons-choropleth-operation';
  const points = snapshotPointJoinGeometry(id, snapshotRecipeGeometryRows(input.points)),
    polygons = snapshotPointJoinGeometry(id, snapshotRecipeGeometryRows(input.polygons));
  validateRequirement(
    id,
    points.kind === 'points' && polygons.kind === 'polygons',
    'points and polygons must have their named geometry kinds'
  );
  const {statistic, backend, boundaryRule} = resolveChoroplethOptions(id, input);
  const values = input.values ? Object.freeze({...input.values}) : undefined,
    output = Object.freeze({...input.output});
  validateRequirement(
    id,
    statistic === 'count' || Boolean(values && output.featureValues),
    'non-count statistic requires values and output.featureValues'
  );
  validateRequirement(
    id,
    statistic !== 'count' || !output.featureValues,
    'count publishes only output.counts; use it as the count feature values'
  );
  validateRequirement(
    id,
    !values || values.length === getPointJoinFeatureCount(points),
    'values needs one row per point feature'
  );
  const {candidateCapacity, leafCapacity} = input;
  validateCapacity(id, 'candidateCapacity', candidateCapacity);
  if (leafCapacity !== undefined) validatePowerOfTwo(id, 'leafCapacity', leafCapacity);
  const polygonCount = getPointJoinFeatureCount(polygons);
  return new GPUContributorOperation({
    id,
    inputs: {
      ...getPointJoinGeometryInputs('points', points),
      ...getPointJoinGeometryInputs('polygons', polygons),
      ...(values ? {values: values.value} : {})
    },
    outputs: Object.fromEntries(
      Object.entries(output)
        .filter(([, value]) => value !== undefined)
        .map(([port, value]) => [
          port,
          {
            value: value!,
            format: port === 'featureValues' ? ('float32' as const) : ('uint32' as const),
            length: port === 'counts' || port === 'featureValues' ? polygonCount : 1
          }
        ])
    ),
    createContributor: getView => ({
      getCommandNodes: graph => {
        const polygonGeometry = resolvePointJoinGeometry(
          id,
          'polygons',
          polygons,
          getView
        ) as Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>;
        addPointsInPolygonsChoroplethRecipe(graph, {
          id,
          points: resolvePointJoinGeometry(id, 'points', points, getView) as never,
          polygons: {...polygonGeometry, candidateCapacity, leafCapacity, boundaryRule},
          values: values ? getView<'float32'>('values') : undefined,
          statistic,
          backend,
          output: {
            counts: getView('counts'),
            featureValues: output.featureValues ? getView('featureValues') : undefined,
            status: {overflow: getView('overflow'), uncertainCount: getView('uncertainCount')}
          }
        });
        return [];
      }
    })
  });
}
