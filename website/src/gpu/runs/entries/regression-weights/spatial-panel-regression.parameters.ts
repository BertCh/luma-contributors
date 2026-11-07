// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// effects and instrumentOrder are GPUSpatialPanelRegression props; the others shape the panel data.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'effects',
    label: 'effects',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['pooled', 'unit']
  },
  {id: 'instrumentOrder', label: 'instrumentOrder', min: 1, max: 2, step: 1, default: 2},
  {id: 'unitEffects', label: 'Data: entity effects', min: 0, max: 1.5, step: 0.1, default: 0.8},
  {id: 'strength', label: 'Data: spatial lag rho', min: 0, max: 0.7, step: 0.05, default: 0.4}
];
