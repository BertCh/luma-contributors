// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// epsilon and minimumPoints are the per-frame clustering parameters of addClusterAndOutlineRecipe.
export const parameters: GpuSceneParameter[] = [
  {id: 'epsilon', label: 'epsilon', min: 1, max: 6, step: 0.25, default: 2.5, unit: '%'},
  {id: 'minimumPoints', label: 'minimumPoints', min: 2, max: 20, step: 1, default: 5}
];
