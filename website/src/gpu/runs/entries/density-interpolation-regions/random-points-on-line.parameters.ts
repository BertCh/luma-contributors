// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `pointsPerLine` is the per-line `counts` input; `seed` is the sampling seed parameter.
export const parameters: GpuSceneParameter[] = [
  {id: 'pointsPerLine', label: 'counts', min: 1, max: 60, step: 1, default: 25},
  {id: 'seed', label: 'seed', min: 1, max: 20, step: 1, default: 5}
];
