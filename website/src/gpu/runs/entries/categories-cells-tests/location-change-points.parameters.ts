// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `cost` is compile-time; `penalty` and `minimumSegmentLength` are the per-frame settings packed by
// GPULocationChangePoints.getParameterValues. The NDVI series has unit-free variance well under 1, so
// the penalty is small: low keeps every wiggle, high keeps only the fire.
export const parameters: GpuSceneParameter[] = [
  {id: 'penalty', label: 'penalty', min: 0.02, max: 1.5, step: 0.02, default: 0.2},
  {id: 'minimumSegmentLength', label: 'minimumSegmentLength', min: 1, max: 5, step: 1, default: 1},
  {id: 'cost', label: 'cost', min: 0, max: 1, step: 1, default: 0, options: ['mean', 'slope']}
];
