// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'gridType',
    label: 'gridType',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['square', 'hex', 'triangle']
  },
  {id: 'cellWidth', label: 'cellWidth', min: 0.03, max: 0.15, step: 0.005, default: 0.08},
  {
    id: 'hexOrientation',
    label: 'hexOrientation',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['pointy', 'flat']
  }
];
