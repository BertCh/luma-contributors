// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `measure` is GPUCellTableCompare's compile-time prop; `shift` moves the "after" points between
// frames (the points are rewritten and keyed again, no recompile).
export const parameters: GpuSceneParameter[] = [
  {id: 'measure', label: 'measure', min: 0, max: 1, step: 1, default: 0, options: ['count', 'sum']},
  {id: 'resolution', label: 'Data: resolution', min: 3, max: 6, step: 1, default: 5},
  {id: 'shift', label: 'Data: shift', min: 0, max: 0.5, step: 0.01, default: 0.2}
];
