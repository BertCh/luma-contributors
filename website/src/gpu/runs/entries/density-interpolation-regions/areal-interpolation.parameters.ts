// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUArealInterpolation's `mode` and `denominator`; zone counts vary the data.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['extensive', 'intensive']
  },
  {
    id: 'denominator',
    label: 'denominator',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['zone', 'overlap']
  },
  {id: 'targetZoneCount', label: 'targetZoneCount', min: 4, max: 24, step: 1, default: 9},
  {
    id: 'show',
    label: 'Show: source / result',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['source', 'result']
  }
];
