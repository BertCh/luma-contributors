// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// replicates is a compile-time prop of addSegregationDecompositionRecipe (paired randomization);
// the other sliders change the final-period populations.
export const parameters: GpuSceneParameter[] = [
  {id: 'replicates', label: 'randomization.replicates', min: 8, max: 128, step: 8, default: 64},
  {id: 'sorting', label: 'Data: sorting', min: 0, max: 1, step: 0.05, default: 0.6},
  {id: 'growth', label: 'Data: groupGrowth', min: 0, max: 1, step: 0.05, default: 0.4}
];
