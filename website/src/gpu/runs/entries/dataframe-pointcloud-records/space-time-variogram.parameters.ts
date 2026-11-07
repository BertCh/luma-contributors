// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame `parameters` of GPUSpaceTimeVariogram (maximum distance and maximum time
// lag) and the temporal drift of the input values.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 0.1,
    max: 0.7,
    step: 0.05,
    default: 0.4
  },
  {
    id: 'maximumTimeLag',
    label: 'maximumTimeLag',
    min: 2,
    max: 24,
    step: 1,
    default: 12,
    unit: 'h'
  },
  {id: 'drift', label: 'Data: temporal drift', min: 0, max: 4, step: 0.2, default: 1.5}
];
