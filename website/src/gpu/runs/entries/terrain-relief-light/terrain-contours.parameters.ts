// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainContours({levels}): ten levels at firstLevel + k * interval, written per frame.
export const parameters: GpuSceneParameter[] = [
  {id: 'interval', label: 'levels: interval', min: 30, max: 140, step: 5, default: 70, unit: ' m'},
  {id: 'firstLevel', label: 'levels: first', min: 0, max: 150, step: 5, default: 40, unit: ' m'}
];
