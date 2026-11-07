// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// gridSize is a constructor prop of GPULinePrecisionReduction.
export const parameters: GpuSceneParameter[] = [
  {id: 'gridSize', label: 'gridSize', min: 0.01, max: 0.2, step: 0.01, default: 0.06}
];
