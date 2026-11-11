// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The three fields of GPUPeriodIndex's fixed `periods` axis: width in hours, count of periods and
// the origin (start of period 0) in hours after the first flash. All three are compile-time.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'width',
    label: 'periods.width',
    min: 0,
    max: 4,
    step: 1,
    default: 2,
    options: ['1 h', '2 h', '3 h', '4 h', '6 h']
  },
  {id: 'count', label: 'periods.count', min: 2, max: 19, step: 1, default: 5},
  {id: 'origin', label: 'periods.origin', min: 0, max: 4, step: 0.5, default: 0, unit: 'h'}
];
