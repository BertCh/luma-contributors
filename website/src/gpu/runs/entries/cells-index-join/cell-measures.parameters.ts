// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `measure` picks which output column is drawn; the resolution
// sets how fine the H3 cells are.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'measure',
    label: 'measure',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['areas', 'perimeters', 'edgeCounts']
  },
  {id: 'resolution', label: 'Data: resolution', min: 3, max: 6, step: 1, default: 5}
];
