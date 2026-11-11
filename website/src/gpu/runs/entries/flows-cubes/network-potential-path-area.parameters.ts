// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// spare sets the per-frame `budget` view: the fastest trip from the origin to the destination plus
// this share. origin picks which intersection the traveller leaves from.
export const parameters: GpuSceneParameter[] = [
  {id: 'spare', label: 'budget: spare time', min: 0, max: 0.8, step: 0.02, default: 0.2},
  {
    id: 'origin',
    label: 'Data: origin',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['west', 'north', 'south']
  }
];
