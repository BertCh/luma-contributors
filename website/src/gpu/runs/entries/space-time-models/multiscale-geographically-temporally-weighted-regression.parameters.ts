// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The per-frame settings of GPUMGTWRSettings: one bandwidth for each component (the intercept first,
// then the slope) and a time scale shared by the initial fit and both components. Bandwidths are in
// space-time distance units (the picture's tile height is 0.5). The intercept truly varies slowly,
// the slope has a sharp front that sweeps west to east.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'interceptBandwidth',
    label: 'intercept bandwidth',
    min: 0.1,
    max: 1.2,
    step: 0.02,
    default: 0.7
  },
  {id: 'slopeBandwidth', label: 'slope bandwidth', min: 0.06, max: 0.8, step: 0.01, default: 0.14},
  {id: 'timeScale', label: 'timeScale', min: 0, max: 0.3, step: 0.005, default: 0.08}
];
