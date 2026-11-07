// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumPower and traceMode are GPUSpatialLagMaximumLikelihood props; strength is the true rho.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumPower', label: 'maximumPower', min: 4, max: 64, step: 1, default: 24},
  {
    id: 'traceMode',
    label: 'traceMode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['exact', 'stochastic']
  },
  {id: 'strength', label: 'Data: spatial lag rho', min: 0, max: 0.8, step: 0.05, default: 0.5}
];
