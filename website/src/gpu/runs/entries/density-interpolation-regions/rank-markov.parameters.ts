// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURankMarkov's `mode`; the period only chooses which map is shown.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['full-rank', 'geographic-rank']
  },
  {id: 'period', label: 'Show: period', min: 0, max: 5, step: 1, default: 0}
];
