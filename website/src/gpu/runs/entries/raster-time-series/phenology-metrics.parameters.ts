// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'startFraction', label: 'startFraction', min: 0.05, max: 0.6, step: 0.05, default: 0.25},
  {id: 'endFraction', label: 'endFraction', min: 0.05, max: 0.6, step: 0.05, default: 0.15},
  {id: 'minimumAmplitude', label: 'minimumAmplitude', min: 0, max: 0.4, step: 0.02, default: 0.2},
  {
    id: 'secondSeasonMinimumFraction',
    label: 'secondSeasonMinimumFraction',
    min: 0.3,
    max: 0.95,
    step: 0.05,
    default: 0.5
  }
];
