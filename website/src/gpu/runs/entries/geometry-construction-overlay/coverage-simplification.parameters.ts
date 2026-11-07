// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0, max: 0.08, step: 0.001, default: 0.03},
  {
    id: 'simplifyBoundary',
    label: 'simplifyBoundary',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  }
];
