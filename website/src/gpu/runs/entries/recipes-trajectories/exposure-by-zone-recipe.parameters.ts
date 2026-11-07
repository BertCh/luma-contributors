// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// threshold is the per-frame hazard cutoff of addExposureByZoneRecipe and weightMeaning a
// compile-time prop; the flood spread reshapes the hazard raster.
export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 0.05, max: 0.95, step: 0.05, default: 0.5},
  {
    id: 'weightMeaning',
    label: 'weightMeaning',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['count', 'density']
  },
  {id: 'spread', label: 'Data: hazardSpread', min: 0.1, max: 1, step: 0.05, default: 0.5}
];
