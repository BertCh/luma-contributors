// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'gridStart',
    label: 'gridStart',
    min: -3.7,
    max: 3.7,
    step: 0.02,
    default: 0.48,
    unit: 'years'
  },
  {
    id: 'maximumGap',
    label: 'maximumGap',
    min: 0.1,
    max: 1.3,
    step: 0.05,
    default: 0.9,
    unit: 'years'
  }
];
