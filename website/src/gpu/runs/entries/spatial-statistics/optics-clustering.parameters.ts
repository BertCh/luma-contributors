// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUOPTICS's `minimumSamples`, `maximumDistance` and `xi`.
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumSamples', label: 'minimumSamples', min: 3, max: 24, step: 1, default: 8},
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.04, max: 0.4, step: 0.02, default: 0.16},
  {id: 'xi', label: 'xi', min: 0.02, max: 0.4, step: 0.02, default: 0.1}
];
