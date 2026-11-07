// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUPairGather's `how` and the slot capacity of its `neighbors` input; the radius limits
// which candidate pairs exist in the first place.
export const parameters: GpuSceneParameter[] = [
  {id: 'how', label: 'how', min: 0, max: 1, step: 1, default: 1, options: ['inner', 'left']},
  {id: 'capacity', label: 'neighbors.capacity', min: 1, max: 4, step: 1, default: 2},
  {id: 'radius', label: 'Data: match radius', min: 0.03, max: 0.2, step: 0.01, default: 0.1}
];
