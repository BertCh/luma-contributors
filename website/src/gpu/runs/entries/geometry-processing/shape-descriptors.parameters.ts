// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// sliverThreshold is the operation's per-frame setting; stretch changes the input shapes.
export const parameters: GpuSceneParameter[] = [
  {id: 'sliverThreshold', label: 'sliverThreshold', min: 0.3, max: 0.95, step: 0.05, default: 0.65},
  {id: 'stretch', label: 'Data: stretch', min: 0, max: 1, step: 0.05, default: 0.8}
];
