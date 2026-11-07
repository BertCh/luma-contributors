// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `family` is GPUCellSetOutline's compile-time prop; `resolution` sets the cells of the input set
// (written through a new aggregation); `spread` scales the point clusters per frame.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'resolution', label: 'Data: resolution', min: 3, max: 6, step: 1, default: 5},
  {id: 'spread', label: 'Data: spread', min: 0.4, max: 1.4, step: 0.05, default: 1}
];
