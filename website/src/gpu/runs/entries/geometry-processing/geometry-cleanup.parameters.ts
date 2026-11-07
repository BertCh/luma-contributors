// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The settings of getGPUGeometryCleanupParameterValues.
export const parameters: GpuSceneParameter[] = [
  {id: 'gridSize', label: 'gridSize', min: 0, max: 0.06, step: 0.005, default: 0.03},
  {id: 'tolerance', label: 'tolerance', min: 0, max: 0.03, step: 0.002, default: 0}
];
