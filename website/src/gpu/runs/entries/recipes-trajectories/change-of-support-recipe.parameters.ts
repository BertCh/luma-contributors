// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// denominator is a compile-time prop of addChangeOfSupportRecipe (how extensive mass is divided);
// "show" picks which transferred variable is drawn; the contrast shapes the source values.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'denominator',
    label: 'denominator',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['zone', 'overlap']
  },
  {
    id: 'show',
    label: 'show',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['extensiveValues', 'intensiveValues']
  },
  {id: 'contrast', label: 'Data: sourceContrast', min: 0, max: 2, step: 0.1, default: 1}
];
