// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'streamedSliceCount', label: 'streamedSliceCount', min: 1, max: 16, step: 1, default: 16},
  {
    id: 'maximumValueMagnitude',
    label: 'maximumValueMagnitude',
    min: 0.3,
    max: 1,
    step: 0.05,
    default: 1
  }
];
