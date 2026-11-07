// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The transpose has no tunable props; the sliders shape the directed k-nearest-neighbor input.
export const parameters: GpuSceneParameter[] = [
  {id: 'k', label: 'Data: k', min: 2, max: 8, step: 1, default: 4},
  {
    id: 'scatter',
    label: 'Data: noise share',
    min: 0,
    max: 0.4,
    step: 0.02,
    default: 0.2
  }
];
