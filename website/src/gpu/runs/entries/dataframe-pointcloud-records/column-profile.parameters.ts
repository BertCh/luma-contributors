// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUColumnProfile's `histogramBinCount` (compile-time), its `mask` (share of rows kept)
// and the per-frame histogram domain in `parameters`.
export const parameters: GpuSceneParameter[] = [
  {id: 'histogramBinCount', label: 'histogramBinCount', min: 6, max: 40, step: 2, default: 20},
  {id: 'mask', label: 'mask', min: 10, max: 100, step: 5, default: 60, unit: '% of map'},
  {id: 'domainMaximum', label: 'domain maximum', min: 20, max: 60, step: 2, default: 50}
];
