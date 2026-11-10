// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `level` and `mode` are GPUS2CapCovering props (compile time); `radius` rewrites the cap view per
// frame, in degrees of arc.
export const parameters: GpuSceneParameter[] = [
  {id: 'level', label: 'level', min: 6, max: 8, step: 1, default: 8},
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['exterior', 'interior']
  },
  {id: 'radius', label: 'cap radius', min: 0.15, max: 0.8, step: 0.05, default: 0.35, unit: '°'}
];
