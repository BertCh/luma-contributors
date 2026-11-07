// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `activeLevel` is GPUCellLevelSelection's per-frame view: rewriting it switches the level with no
// recompile.
export const parameters: GpuSceneParameter[] = [
  {id: 'activeLevel', label: 'activeLevel', min: 0, max: 3, step: 1, default: 1},
  {
    id: 'family',
    label: 'Data: family',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['h3', 'quadbin']
  }
];
