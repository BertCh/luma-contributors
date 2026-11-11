// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'changeThreshold', label: 'changeThreshold', min: 3, max: 16, step: 0.5, default: 6.5},
  {id: 'consecutiveCount', label: 'consecutiveCount', min: 3, max: 9, step: 1, default: 6},
  {id: 'cloudFraction', label: 'Data: cloud fraction', min: 0, max: 0.12, step: 0.01, default: 0.03}
];
