// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 3, step: 0.5, default: 1.5, unit: 'cells'},
  {
    id: 'temporalWindow',
    label: 'temporalWindow',
    min: 1,
    max: 3,
    step: 1,
    default: 1,
    unit: 'slices'
  },
  {id: 'minimumMagnitude', label: 'Data: minimumMagnitude', min: 4, max: 5, step: 0.1, default: 4}
];
