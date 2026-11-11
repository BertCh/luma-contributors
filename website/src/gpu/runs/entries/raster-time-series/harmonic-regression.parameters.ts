// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'period', label: 'period', min: 0.6, max: 1.4, step: 0.05, default: 1, unit: 'years'},
  {id: 'harmonicCount', label: 'harmonicCount', min: 1, max: 3, step: 1, default: 2},
  {id: 'cloudFraction', label: 'Data: cloud fraction', min: 0, max: 0.4, step: 0.02, default: 0.14}
];
