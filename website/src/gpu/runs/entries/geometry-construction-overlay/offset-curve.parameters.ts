// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: -0.06, max: 0.06, step: 0.0025, default: 0.03},
  {
    id: 'joinStyle',
    label: 'joinStyle',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['round', 'mitre', 'bevel']
  },
  {id: 'mitreLimit', label: 'mitreLimit', min: 1, max: 8, step: 0.25, default: 2}
];
