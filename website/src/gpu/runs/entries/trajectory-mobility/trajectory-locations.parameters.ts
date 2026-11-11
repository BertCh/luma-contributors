// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The per-frame clustering radius of GPUTrajectoryLocations, in world units (the frame is about
// 25 km across). The staypoints it clusters come from GPUTrajectoryStaypoints with fixed settings.
export const parameters: GpuSceneParameter[] = [
  {id: 'epsilon', label: 'epsilon', min: 0.002, max: 0.04, step: 0.002, default: 0.01}
];
