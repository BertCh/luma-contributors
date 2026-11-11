// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `sliceCount` is a compile-time prop of GPUPettittTest: how many of the 16 scenes, from the first,
// enter the test (the fire is scene 9, so 9 or fewer scenes see no change). `alpha` is the p-value
// cutoff the picture applies to the output (not a prop).
export const parameters: GpuSceneParameter[] = [
  {id: 'sliceCount', label: 'sliceCount', min: 8, max: 16, step: 1, default: 16},
  {id: 'alpha', label: 'p-value cutoff', min: 0.001, max: 0.1, step: 0.001, default: 0.01}
];
