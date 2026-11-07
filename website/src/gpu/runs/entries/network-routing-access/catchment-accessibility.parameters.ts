// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// method is compile-time; costLimit and beta shape the facility weights each frame.
export const parameters: GpuSceneParameter[] = [
  {id: 'method', label: 'method', min: 0, max: 1, step: 1, default: 0, options: ['2sfca', '3sfca']},
  {id: 'costLimit', label: 'costLimit', min: 0.2, max: 1.5, step: 0.05, default: 0.7},
  {id: 'decay', label: 'decay', min: 0, max: 1, step: 1, default: 1, options: ['none', 'gaussian']}
];
