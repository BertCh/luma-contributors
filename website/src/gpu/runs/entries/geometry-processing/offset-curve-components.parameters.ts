// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// distance and the join style are constructor props (the raw offset is built with the same ones).
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0.01, max: 0.08, step: 0.005, default: 0.04},
  {
    id: 'joinStyle',
    label: 'joinStyle',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['round', 'mitre', 'bevel']
  }
];
