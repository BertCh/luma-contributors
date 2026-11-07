// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame `parameters` (scale), `domain` and `range` views of GPUScalarScale.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'scale',
    label: 'scale',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['linear', 'sqrt', 'log']
  },
  {id: 'domainMaximum', label: 'domain max', min: 20, max: 80, step: 2, default: 50},
  {id: 'rangeMaximum', label: 'range max', min: 4, max: 30, step: 1, default: 20}
];
