// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'gridWidth', label: 'grid.width', min: 16, max: 120, step: 4, default: 48, unit: 'cells'},
  {id: 'holeScale', label: 'Data: hole size', min: 0.1, max: 0.8, step: 0.05, default: 0.4}
];
