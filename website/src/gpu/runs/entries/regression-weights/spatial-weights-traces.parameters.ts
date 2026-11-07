// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialWeightsTraces' `mode`, `maximumPower` and `probeCount` props.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['exact', 'stochastic']
  },
  {id: 'maximumPower', label: 'maximumPower', min: 4, max: 32, step: 1, default: 20},
  {id: 'probeCount', label: 'probeCount', min: 2, max: 64, step: 1, default: 6}
];
