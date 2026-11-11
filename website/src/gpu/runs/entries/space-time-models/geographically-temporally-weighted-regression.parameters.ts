// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The two per-frame settings of GPUGTWRSettings. bandwidth is the kernel bandwidth in space-time
// distance units (the picture's tile height is 0.5); timeScale is tau, the weight of a squared
// period difference in sqrt(ds^2 + tau dt^2). Zero pools all periods into one spatial fit.
export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 0.08, max: 0.8, step: 0.01, default: 0.2},
  {id: 'timeScale', label: 'timeScale', min: 0, max: 0.3, step: 0.005, default: 0.08}
];
