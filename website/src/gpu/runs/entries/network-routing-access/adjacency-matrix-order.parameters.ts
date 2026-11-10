// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tieKeys is an optional compile-time input of GPUAdjacencyMatrixOrder.
export const parameters: GpuSceneParameter[] = [
  {id: 'resolution', label: 'resolution', min: 16, max: 128, step: 16, default: 64},
  {id: 'communities', label: 'North–south bands', min: 2, max: 8, step: 1, default: 5}
];
