// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame stop settings of addFleetDwellRecipe (speeds are world units per second);
// the zone size only moves the zone polygons.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'stopSpeedThreshold',
    label: 'stopSpeedThreshold',
    min: 0,
    max: 0.003,
    step: 0.0001,
    default: 0.0006
  },
  {
    id: 'stopMinimumDuration',
    label: 'stopMinimumDuration',
    min: 0,
    max: 150,
    step: 5,
    default: 30,
    unit: 'playback s'
  },
  {id: 'zoneSize', label: 'zoneSize', min: 0.5, max: 1.6, step: 0.05, default: 1}
];
