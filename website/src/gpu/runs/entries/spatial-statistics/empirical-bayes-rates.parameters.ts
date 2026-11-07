// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUEmpiricalBayesRates's `clampPriorVariance`, plus the data the rates come from.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'clampPriorVariance',
    label: 'clampPriorVariance',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {id: 'riskContrast', label: 'Data: riskContrast', min: 0, max: 1, step: 0.05, default: 1},
  {id: 'populationScale', label: 'Data: populationScale', min: 0.2, max: 5, step: 0.2, default: 3}
];
