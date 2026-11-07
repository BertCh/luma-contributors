// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumSegmentLength',
    label: 'maximumSegmentLength',
    min: 0.04,
    max: 0.5,
    step: 0.01,
    default: 0.14
  },
  {
    id: 'closedLines',
    label: 'closedLines',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  }
];
