// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointPatternIndices' compile-time `quadratGrid`, its `mask`, and the input clustering.
export const parameters: GpuSceneParameter[] = [
  {id: 'quadratGrid', label: 'quadratGrid', min: 3, max: 12, step: 1, default: 8, unit: '× same'},
  {id: 'mask', label: 'mask', min: 20, max: 100, step: 10, default: 100, unit: '% of map'},
  {id: 'clustering', label: 'Data: observed flash share', min: 0, max: 1, step: 0.05, default: 1}
];
