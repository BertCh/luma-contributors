// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'threshold', min: 0, max: 20, step: 0.5, default: 2, unit: ' mm/day'},
  {
    id: 'condition',
    label: 'condition',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['above', 'at-or-above', 'below', 'at-or-below']
  },
  {
    id: 'minimumSpellLength',
    label: 'minimumSpellLength',
    min: 1,
    max: 7,
    step: 1,
    default: 3,
    unit: ' days'
  },
  {
    id: 'statistic',
    label: 'state shown',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['longestRun', 'spellCount', 'matchingSampleCount', 'spellSampleCount']
  }
];
