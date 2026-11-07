// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `operation` is GPUH3CellIdentities' compile-time prop; `resolution` sets the size of the cells
// whose identifiers are listed.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['directed-edges', 'vertices']
  },
  {id: 'resolution', label: 'Data: resolution', min: 3, max: 6, step: 1, default: 4}
];
