// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'mode', label: 'mode', min: 0, max: 1, step: 1, default: 0, options: ['inside', 'outside']},
  {id: 'shift', label: 'Data: shift', min: -0.5, max: 0.5, step: 0.02, default: 0, unit: 'cells'}
];
