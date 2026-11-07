// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// With the right side omitted, GPUSegmentIntersection reports self-intersections of the left side.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'right',
    label: 'right',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['omitted', 'given']
  },
  {id: 'wobble', label: 'Data: wobble', min: 0.02, max: 0.14, step: 0.01, default: 0.07}
];
