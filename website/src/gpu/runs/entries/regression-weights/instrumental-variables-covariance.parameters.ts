// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// method is a GPUInstrumentalVariablesCovariance prop; heteroskedasticity makes the errors grow with x1.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['HC0', 'HC1', 'HAC']
  },
  {
    id: 'heteroskedasticity',
    label: 'Data: heteroskedasticity',
    min: 0,
    max: 3,
    step: 0.1,
    default: 1.5
  },
  {id: 'kernelWeight', label: 'Data: kernel weight', min: 0.02, max: 0.4, step: 0.02, default: 0.2}
];
