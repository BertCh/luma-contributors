// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The per-frame tolerance of GPUTrajectoryThinning in mode 'minimum-distance', in world units
// (the frame is about 25 km across).
export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0.002, max: 0.15, step: 0.002, default: 0.04}
];
