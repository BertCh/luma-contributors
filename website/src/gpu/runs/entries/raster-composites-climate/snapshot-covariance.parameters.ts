// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'sliceCount', label: 'sliceCount', min: 4, max: 16, step: 1, default: 12},
  {
    id: 'maximumValueMagnitude',
    label: 'maximumValueMagnitude',
    min: 0.5,
    max: 1,
    step: 0.05,
    default: 1
  }
];
