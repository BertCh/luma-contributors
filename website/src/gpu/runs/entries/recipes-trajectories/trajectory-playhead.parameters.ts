// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame playhead and maximumGap settings of GPUTrajectoryPlayhead.
export const parameters: GpuSceneParameter[] = [
  {id: 'playhead', label: 'playhead', min: 0, max: 600, step: 5, default: 120, unit: 's'},
  {id: 'maximumGap', label: 'maximumGap', min: 0, max: 12, step: 1, default: 0, unit: 's'}
];
