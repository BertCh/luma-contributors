// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// stepCount is the operation's compile-time prop; stepLength is a per-frame setting of its packer.
export const parameters: GpuSceneParameter[] = [
  {id: 'stepCount', label: 'stepCount', min: 2, max: 40, step: 2, default: 20},
  {id: 'stepLength', label: 'stepLength', min: 0.25, max: 1.5, step: 0.25, default: 0.5},
  {id: 'seed', label: 'seed', min: 0, max: 12, step: 1, default: 2}
];
