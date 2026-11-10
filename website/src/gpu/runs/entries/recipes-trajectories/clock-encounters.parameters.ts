// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// distance, and the shared clock's start and step, are the per-frame props of addClockEncounters.
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0.01, max: 0.15, step: 0.005, default: 0.06},
  {id: 'start', label: 'clock start', min: 0, max: 400, step: 10, default: 100, unit: 'playback s'},
  {id: 'step', label: 'clock step', min: 5, max: 40, step: 1, default: 12, unit: 'playback s'}
];
