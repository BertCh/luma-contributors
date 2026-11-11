// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `order` is the compile-time HEALPix depth of the keys and of GPUHealpixCellAggregation. `measure`
// chooses which output column colors the cells.
export const parameters: GpuSceneParameter[] = [
  {id: 'order', label: 'order', min: 1, max: 5, step: 1, default: 3},
  {
    id: 'measure',
    label: 'measure',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['counts', 'sums', 'minimums', 'maximums']
  }
];
