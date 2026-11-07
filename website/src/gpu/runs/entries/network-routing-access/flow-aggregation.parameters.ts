// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// gridSize and excludeSelfFlows are compile-time props; spread moves the origins and destinations.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'gridSize',
    label: 'zones (gridSize)',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['8 columns', '12 columns', '16 columns']
  },
  {
    id: 'excludeSelfFlows',
    label: 'excludeSelfFlows',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {id: 'spread', label: 'Data: spread', min: 0.3, max: 2, step: 0.1, default: 1}
];
