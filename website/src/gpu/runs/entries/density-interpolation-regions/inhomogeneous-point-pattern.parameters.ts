// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUInhomogeneousPointPattern's `bandwidth`, `edgeCorrection` and `selfContribution`.
export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 0.04, max: 0.4, step: 0.02, default: 0.16},
  {
    id: 'edgeCorrection',
    label: 'edgeCorrection',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['border', 'translation']
  },
  {
    id: 'selfContribution',
    label: 'selfContribution',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['include', 'exclude']
  }
];
