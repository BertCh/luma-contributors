// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainContours({levels}): ten levels at firstLevel + k * interval, written per frame.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'interval',
    label: 'levels: interval',
    min: 30,
    max: 250,
    step: 10,
    default: 150,
    unit: ' m'
  },
  {
    id: 'firstLevel',
    label: 'levels: first',
    min: 700,
    max: 2100,
    step: 25,
    default: 750,
    unit: ' m'
  }
];
