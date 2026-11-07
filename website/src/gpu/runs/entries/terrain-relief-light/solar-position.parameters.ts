// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUSolarPositionParameterValues({timestamp}) (hour and month of 2026).
export const parameters: GpuSceneParameter[] = [
  {id: 'hour', label: 'timestamp: hour (UTC)', min: 0, max: 24, step: 0.5, default: 12, unit: ' h'},
  {id: 'month', label: 'timestamp: month', min: 1, max: 12, step: 1, default: 6}
];
