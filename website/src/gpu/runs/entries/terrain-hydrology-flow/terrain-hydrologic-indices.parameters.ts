// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// minimumSlope is the operation's own setting; output picks which of its four index views is drawn.
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumSlope', label: 'minimumSlope', min: 0.001, max: 0.2, step: 0.001, default: 0.02},
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['wetnessIndex', 'streamPowerIndex', 'slope', 'specificCatchmentArea']
  }
];
