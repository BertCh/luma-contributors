// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// numberOfSigmas is the per-frame setting; windowSize is compile-time (full window in rows).
export const parameters: GpuSceneParameter[] = [
  {id: 'numberOfSigmas', label: 'numberOfSigmas', min: 0.5, max: 6, step: 0.25, default: 3},
  {id: 'windowSize', label: 'windowSize', min: 3, max: 15, step: 2, default: 7, unit: 'rows'}
];
