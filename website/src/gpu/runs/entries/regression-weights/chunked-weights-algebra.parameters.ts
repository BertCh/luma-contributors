// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUChunkedSpatialWeightsAlgebra's `operation`; chunks is how the rows are split.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 3,
    step: 1,
    default: 1,
    options: ['union', 'intersection', 'difference', 'symmetricDifference']
  },
  {id: 'chunkCount', label: 'Data: chunks', min: 1, max: 4, step: 1, default: 3}
];
