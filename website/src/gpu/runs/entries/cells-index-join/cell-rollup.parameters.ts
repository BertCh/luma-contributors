// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `family` and the target `resolution` of GPUCellRollup (compile time). The source table is
// aggregated two levels finer, so the rolled-up cells are drawn over their children.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'resolution', label: 'resolution', min: 3, max: 5, step: 1, default: 4}
];
