// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// addPopulationSupportAlignmentRecipe takes populations with source ids and a list of target ids;
// the sliders make some target ids absent from the source or repeated.
export const parameters: GpuSceneParameter[] = [
  {id: 'absent', label: 'Data: absentTargetIds', min: 0, max: 6, step: 1, default: 2},
  {id: 'duplicated', label: 'Data: repeatedTargetIds', min: 0, max: 6, step: 1, default: 1}
];
