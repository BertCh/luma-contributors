// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUChunkedSpatialWeightsTransform's `operation` and `bandwidth`; chunks splits the rows.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 4,
    step: 1,
    default: 2,
    options: ['row', 'binary', 'kernel', 'double', 'variance']
  },
  {id: 'bandwidth', label: 'bandwidth', min: 0.04, max: 0.16, step: 0.01, default: 0.08},
  {id: 'chunkCount', label: 'Data: chunks', min: 1, max: 4, step: 1, default: 3}
];
