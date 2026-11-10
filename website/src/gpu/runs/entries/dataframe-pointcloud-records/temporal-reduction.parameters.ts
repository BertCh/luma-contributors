// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame bucketing `parameters` (origin and width) of GPUTemporalReduction and the
// regional lightning-flash counts.
export const parameters: GpuSceneParameter[] = [
  {id: 'origin', label: 'origin', min: 0, max: 12, step: 0.5, default: 0, unit: 'h'},
  {id: 'width', label: 'width', min: 0.5, max: 2, step: 0.25, default: 1, unit: 'h'}
];
