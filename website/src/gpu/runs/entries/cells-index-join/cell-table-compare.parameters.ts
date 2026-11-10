// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `measure` is GPUCellTableCompare's compile-time prop; `shift` and `shiftNorth` move the "after"
// points between frames (the points are rewritten and keyed again, no recompile).
export const parameters: GpuSceneParameter[] = [
  {id: 'measure', label: 'measure', min: 0, max: 1, step: 1, default: 0, options: ['count', 'sum']},
  {id: 'resolution', label: 'Data: resolution', min: 3, max: 6, step: 1, default: 5},
  {
    id: 'shift',
    label: 'shift east',
    min: -0.5,
    max: 0.5,
    step: 0.01,
    default: 0.2,
    unit: '°E'
  },
  {
    id: 'shiftNorth',
    label: 'shift north',
    min: -0.3,
    max: 0.3,
    step: 0.01,
    default: 0,
    unit: '°N'
  }
];
