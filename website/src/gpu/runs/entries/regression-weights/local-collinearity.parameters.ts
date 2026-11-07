// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The operation has no tunable props. Neighbors sets the local neighborhood (the weights), and
// collinearity is how strongly the second predictor follows the first toward the east.
export const parameters: GpuSceneParameter[] = [
  {id: 'collinearity', label: 'Data: collinearity', min: 0, max: 0.98, step: 0.02, default: 0.9},
  {id: 'neighbors', label: 'Data: neighbors', min: 4, max: 16, step: 1, default: 8}
];
