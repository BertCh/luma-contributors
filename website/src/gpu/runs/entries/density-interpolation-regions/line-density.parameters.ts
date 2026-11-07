// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPULineDensity's `rows` prop (the grid resolution; cell size follows) plus data size.
export const parameters: GpuSceneParameter[] = [
  {id: 'rows', label: 'rows', min: 8, max: 64, step: 4, default: 32},
  {id: 'lineCount', label: 'Data: lineCount', min: 4, max: 40, step: 2, default: 18}
];
