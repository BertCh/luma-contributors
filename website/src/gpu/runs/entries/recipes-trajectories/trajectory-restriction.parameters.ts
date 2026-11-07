// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame window of GPUTrajectoryRestriction: parameters[0], [1] and [4]
// (maximumX, as a fraction of the world width).
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumTime', label: 'minimumTime', min: 0, max: 600, step: 10, default: 100, unit: 's'},
  {id: 'maximumTime', label: 'maximumTime', min: 0, max: 600, step: 10, default: 450, unit: 's'},
  {id: 'maximumX', label: 'maximumX', min: 0.3, max: 1, step: 0.01, default: 0.7}
];
