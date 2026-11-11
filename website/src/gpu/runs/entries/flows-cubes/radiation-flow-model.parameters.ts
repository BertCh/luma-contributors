// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// normalization is GPURadiationFlowModel's compile-time prop. minimumJobs filters the observed flows,
// which changes the outflows and populations the model reads on the next run.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'normalization',
    label: 'normalization',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['none', 'finite-size']
  },
  {
    id: 'minimumJobs',
    label: 'Data: minimum workers per flow',
    min: 25,
    max: 400,
    step: 5,
    default: 25
  }
];
