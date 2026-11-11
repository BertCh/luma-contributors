// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `labelCapacity` is a compile-time prop of GPUGeographicalDetector: strata with a label at or above it
// are left out. `stratifier` chooses the stratification (the pointer cross, or equal-count quintiles of
// a tract column); `variable` is the response.
export const parameters: GpuSceneParameter[] = [
  {id: 'labelCapacity', label: 'labelCapacity', min: 2, max: 5, step: 1, default: 5},
  {
    id: 'stratifier',
    label: 'Data: stratifier',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['cross', 'age 65+ quintiles', 'unemployment quintiles']
  },
  {
    id: 'variable',
    label: 'Data: response',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['poverty', 'unemployment', 'age 65+']
  }
];
