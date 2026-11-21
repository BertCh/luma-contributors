// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUHDBSCAN's `minimumClusterSize` and `minimumSamples`.
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumClusterSize', label: 'minimumClusterSize', min: 10, max: 160, step: 10, default: 30},
  {id: 'minimumSamples', label: 'minimumSamples', min: 2, max: 24, step: 1, default: 8}
];
