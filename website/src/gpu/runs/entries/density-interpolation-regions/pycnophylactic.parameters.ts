// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPycnophylacticInterpolation's `maximumIterations` and `kernel`; zone count varies the data.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumIterations', label: 'maximumIterations', min: 0, max: 80, step: 2, default: 40},
  {id: 'kernel', label: 'kernel', min: 0, max: 1, step: 1, default: 0, options: ['rook', 'box']},
  {id: 'zoneCount', label: 'Data: zoneCount', min: 6, max: 30, step: 1, default: 14}
];
