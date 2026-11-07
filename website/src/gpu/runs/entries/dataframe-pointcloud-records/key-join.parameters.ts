// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUKeyJoin's `rightMask` (share of right-table rows present) and `leftMask`.
export const parameters: GpuSceneParameter[] = [
  {id: 'rightMask', label: 'rightMask', min: 20, max: 100, step: 5, default: 70, unit: '% of rows'},
  {id: 'leftMask', label: 'leftMask', min: 20, max: 100, step: 5, default: 100, unit: '% of rows'}
];
