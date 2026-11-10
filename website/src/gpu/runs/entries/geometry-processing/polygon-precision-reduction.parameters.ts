// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// gridSize is a constructor prop of GPUPolygonPrecisionReduction.
export const parameters: GpuSceneParameter[] = [
  {id: 'gridSize', label: 'gridSize', min: 0.005, max: 0.12, step: 0.005, default: 0.06}
];
