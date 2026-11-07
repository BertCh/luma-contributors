// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'factor',
    label: 'destination.spacing (source cells)',
    min: 1.5,
    max: 12,
    step: 0.5,
    default: 5.5
  },
  {
    id: 'normalization',
    label: 'normalization',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['destination-area', 'covered-area', 'extensive']
  },
  {
    id: 'shift',
    label: 'destination.origin shift',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.5,
    unit: 'cell'
  }
];
