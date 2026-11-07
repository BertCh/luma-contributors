// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// routeCount is compile-time; congestion rewrites the topology weights each frame.
export const parameters: GpuSceneParameter[] = [
  {id: 'routeCount', label: 'routeCount', min: 1, max: 4, step: 1, default: 3},
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 4, step: 0.1, default: 1.2}
];
