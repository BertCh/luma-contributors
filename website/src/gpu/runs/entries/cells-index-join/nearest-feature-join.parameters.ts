// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `k` is GPUNearestFeatureJoin's neighbor count (compile time); `radius` is its per-frame search
// radius view. Queries that find no feature within the radius draw no line.
export const parameters: GpuSceneParameter[] = [
  {id: 'k', label: 'k', min: 1, max: 4, step: 1, default: 1},
  {id: 'radius', label: 'radius', min: 0.05, max: 0.6, step: 0.01, default: 0.3}
];
