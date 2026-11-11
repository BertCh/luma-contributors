// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPointToCell's `family` and `resolution` props (fixed at compile time). Quadbin
// uses zoom `resolution + 5` so both families draw cells of a similar size.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'resolution', label: 'resolution', min: 3, max: 6, step: 1, default: 4}
];
