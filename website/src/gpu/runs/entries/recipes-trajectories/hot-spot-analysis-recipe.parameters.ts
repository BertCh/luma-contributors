// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 4, step: 1, default: 1, unit: 'cells'},
  {id: 'minimumNeighbors', label: 'minimumNeighbors', min: 0, max: 8, step: 1, default: 0},
  {id: 'minimumMagnitude', label: 'Data: minimumMagnitude', min: 4, max: 5, step: 0.1, default: 4}
];
