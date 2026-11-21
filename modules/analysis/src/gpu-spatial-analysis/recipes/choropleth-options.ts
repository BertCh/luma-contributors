// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUChoroplethStatistic} from './points-in-polygons-choropleth-recipe';
import type {GPUPointInPolygonBoundaryRule} from '../spatial-join/gpu-point-in-polygon-join';

/** Shared construction-time option vocabulary for the packed planar aggregate forms. @internal */
export function resolveChoroplethOptions(
  id: string,
  props: {
    statistic?: GPUChoroplethStatistic;
    backend?: 'zonal' | 'group';
    boundaryRule?: GPUPointInPolygonBoundaryRule;
  }
) {
  const statistic = props.statistic ?? 'count';
  const backend = props.backend ?? 'zonal';
  const boundaryRule = props.boundaryRule ?? 'half-open';
  validateRequirement(
    id,
    ['count', 'sum', 'mean', 'minimum', 'maximum'].includes(statistic),
    `statistic must be count, sum, mean, minimum or maximum; received ${statistic}`
  );
  validateRequirement(
    id,
    ['zonal', 'group'].includes(backend),
    `backend must be zonal or group; received ${backend}`
  );
  validateRequirement(
    id,
    ['closed', 'open', 'half-open'].includes(boundaryRule),
    `boundaryRule must be closed, open or half-open; received ${boundaryRule}`
  );
  return {statistic, backend, boundaryRule};
}
