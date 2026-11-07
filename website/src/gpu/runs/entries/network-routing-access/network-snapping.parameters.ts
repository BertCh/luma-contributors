// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumSnapDistance is GPUNetworkSnapping's per-frame view.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumSnapDistance',
    label: 'maximumSnapDistance',
    min: 0.004,
    max: 0.05,
    step: 0.002,
    default: 0.03
  },
  {id: 'spread', label: 'Data: spread', min: 0.2, max: 2, step: 0.1, default: 1}
];
