// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// beforeSlice and afterSlice are the per-frame change-detection parameters of
// addRegionalRasterChangeRecipe, threshold its per-frame area cutoff.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'beforeSlice',
    label: 'Before',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['Jul 2021', 'Sep 2021', 'Jul 2023', 'Aug 2024']
  },
  {
    id: 'afterSlice',
    label: 'After',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['Jul 2021', 'Sep 2021', 'Jul 2023', 'Aug 2024']
  },
  {id: 'threshold', label: 'threshold', min: 0, max: 0.5, step: 0.02, default: 0.12}
];
