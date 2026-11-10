// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// mode is a constructor prop; the box is the per-frame parameter buffer (center and size in world units).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['intersects', 'within', 'contains']
  },
  {id: 'centerX', label: 'box center x', min: 0.2, max: 1.6, step: 0.05, default: 0.9},
  {id: 'centerY', label: 'box center y', min: 0.2, max: 0.8, step: 0.05, default: 0.5},
  {id: 'size', label: 'box size', min: 0.1, max: 1.2, step: 0.05, default: 0.55}
];
