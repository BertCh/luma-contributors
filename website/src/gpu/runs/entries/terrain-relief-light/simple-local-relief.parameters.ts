// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUSimpleLocalReliefParameterValues({radius, verticalExaggeration}).
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 20, step: 1, default: 8, unit: ' cells'},
  {
    id: 'verticalExaggeration',
    label: 'verticalExaggeration',
    min: 0.5,
    max: 4,
    step: 0.25,
    default: 1
  }
];
