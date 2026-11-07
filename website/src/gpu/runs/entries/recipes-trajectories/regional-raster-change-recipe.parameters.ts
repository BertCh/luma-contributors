// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// beforeSlice and afterSlice are the per-frame change-detection parameters of
// addRegionalRasterChangeRecipe, threshold its per-frame area cutoff.
export const parameters: GpuSceneParameter[] = [
  {id: 'beforeSlice', label: 'beforeSlice', min: 0, max: 2, step: 1, default: 0},
  {id: 'afterSlice', label: 'afterSlice', min: 1, max: 3, step: 1, default: 3},
  {id: 'threshold', label: 'threshold', min: 0.5, max: 12, step: 0.5, default: 4}
];
