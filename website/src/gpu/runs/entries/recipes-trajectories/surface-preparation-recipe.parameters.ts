// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumDistance, filter.radius and filter.quantile are compile-time props of
// addSurfacePreparationRecipe (a filter radius of 0 leaves the filter out); the gap slider rewrites
// the input.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 2,
    max: 24,
    step: 1,
    default: 10,
    unit: 'cells'
  },
  {id: 'radius', label: 'filter.radius', min: 0, max: 4, step: 1, default: 2, unit: 'cells'},
  {id: 'quantile', label: 'filter.quantile', min: 0, max: 1, step: 0.1, default: 0.5},
  {id: 'gaps', label: 'Data: gaps', min: 0.5, max: 0.8, step: 0.01, default: 0.62}
];
