// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The system has no tunable props; the sliders set the error correlation between the two equations.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'correlation',
    label: 'Data: error correlation',
    min: -0.9,
    max: 0.9,
    step: 0.1,
    default: 0.7
  },
  {id: 'noise', label: 'Data: noise', min: 0.1, max: 1.2, step: 0.05, default: 0.5}
];
