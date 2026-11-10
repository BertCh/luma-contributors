// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUNetworkServiceAreas' `facilityCount` view and `costLimit` view, plus data congestion.
export const parameters: GpuSceneParameter[] = [
  {id: 'facilityCount', label: 'facilityCount', min: 1, max: 8, step: 1, default: 4},
  {id: 'costLimit', label: 'costLimit', min: 0.1, max: 2, step: 0.05, default: 0.8},
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 3, step: 0.1, default: 0.8}
];
