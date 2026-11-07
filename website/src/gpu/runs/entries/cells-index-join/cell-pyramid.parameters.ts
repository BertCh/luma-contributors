// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `family` is a GPUCellPyramid prop (compile time). `level` picks which of the pyramid's four
// tables (finest first) is drawn; the whole pyramid is computed once.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'level', label: 'level', min: 0, max: 3, step: 1, default: 1}
];
