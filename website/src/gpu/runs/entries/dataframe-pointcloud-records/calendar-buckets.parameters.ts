// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame `parameters` of GPUCalendarBuckets: the UTC offset and first day of week.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'utcOffsetHours',
    label: 'utcOffset',
    min: -12,
    max: 12,
    step: 1,
    default: -4,
    unit: 'h'
  },
  {
    id: 'firstDayOfWeek',
    label: 'firstDayOfWeek',
    min: 0,
    max: 6,
    step: 1,
    default: 0,
    options: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  }
];
