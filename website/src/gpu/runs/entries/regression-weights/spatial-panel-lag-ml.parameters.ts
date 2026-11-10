// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// effects and maximumIterations are GPUSpatialPanelLagMaximumLikelihood props.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'effects',
    label: 'effects',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['random', 'unit']
  },
  {id: 'maximumIterations', label: 'maximumIterations', min: 1, max: 20, step: 1, default: 12},
  {id: 'unitEffects', label: 'Data: entity effects', min: 0, max: 1.5, step: 0.1, default: 0.8},
  {id: 'strength', label: 'Data: spatial lag rho', min: 0, max: 0.7, step: 0.05, default: 0.4},
  {id: 'period', label: 'period shown', min: 1, max: 6, step: 1, default: 1}
];
