// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// order and resolution are compile-time props of GPUAdjacencyMatrix.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'order',
    label: 'order',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['none (input order)', 'grouped']
  },
  {
    id: 'resolution',
    label: 'resolution',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['32', '64', '128']
  },
  {id: 'communities', label: 'North–south bands', min: 2, max: 8, step: 1, default: 4}
];
