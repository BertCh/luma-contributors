// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'matchThreshold', label: 'matchThreshold', min: 0.5, max: 0.99, step: 0.01, default: 0.9},
  {
    id: 'shown',
    label: 'Output shown',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['maximum and matches', 'matches only']
  }
];
