// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// streamThreshold is the per-frame stream threshold of the flow stage of addDrainageNetworkRecipe.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'streamThreshold',
    label: 'streamThreshold',
    min: 10,
    max: 300,
    step: 10,
    default: 60,
    unit: 'cells'
  }
];
