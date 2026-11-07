// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumPower and traceMode are GPUSpatialPanelErrorMaximumLikelihood props (it extends the error ML).
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumPower', label: 'maximumPower', min: 4, max: 48, step: 1, default: 24},
  {
    id: 'traceMode',
    label: 'traceMode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['exact', 'stochastic']
  },
  {id: 'unitEffects', label: 'Data: entity effects', min: 0, max: 1.5, step: 0.1, default: 0.8},
  {id: 'strength', label: 'Data: spatial error lambda', min: 0, max: 0.7, step: 0.05, default: 0.5}
];
