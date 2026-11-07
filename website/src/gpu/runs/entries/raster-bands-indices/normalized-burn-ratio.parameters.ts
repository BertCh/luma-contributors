// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'epsilon', label: 'epsilon', min: 0, max: 0.4, step: 0.02, default: 0},
  {id: 'clampMaximum', label: 'clamp', min: 0.1, max: 1, step: 0.05, default: 1}
];
