// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// permutationCount is written into the test's per-frame parameters, bandwidth into the regression's;
// drift is how strongly the true slope of the first predictor changes from west to east.
export const parameters: GpuSceneParameter[] = [
  {id: 'permutationCount', label: 'permutationCount', min: 4, max: 32, step: 1, default: 24},
  {id: 'bandwidth', label: 'bandwidth', min: 0.3, max: 1.2, step: 0.05, default: 0.6},
  {id: 'drift', label: 'Data: slope drift', min: 0, max: 3, step: 0.1, default: 2}
];
