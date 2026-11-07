// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// operation and maximumFacilityCount are compile-time props; coverageDistance is a per-frame view.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'operation',
    label: 'operation',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['p-median', 'maximum-coverage']
  },
  {id: 'maximumFacilityCount', label: 'maximumFacilityCount', min: 1, max: 8, step: 1, default: 4},
  {id: 'coverageDistance', label: 'coverageDistance', min: 0.1, max: 1, step: 0.05, default: 0.4}
];
