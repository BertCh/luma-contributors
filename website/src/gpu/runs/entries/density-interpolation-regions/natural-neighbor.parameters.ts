// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUNaturalNeighbor's `maximumVertexCount`; the sample count varies the data.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumVertexCount', label: 'maximumVertexCount', min: 8, max: 64, step: 4, default: 32},
  {id: 'sampleCount', label: 'Data: sampleCount', min: 8, max: 60, step: 4, default: 40}
];
