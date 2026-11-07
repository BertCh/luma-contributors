// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// correction is compile-time; radii (maximum radius) and intensity are per-frame views.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'correction',
    label: 'correction',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['none', 'ang']
  },
  {id: 'radii', label: 'radii (maximum)', min: 0.2, max: 1.5, step: 0.05, default: 0.9},
  {id: 'gradient', label: 'Data: intensity gradient', min: 0, max: 4, step: 0.2, default: 1.5}
];
