// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Both change the second epoch. The fitted vertical shift removes the bias (try it: the picture does
// not move), so only the real change on unstable ground is left in the corrected difference.
export const parameters: GpuSceneParameter[] = [
  {id: 'bias', label: 'Data: vertical bias', min: -0.1, max: 0.1, step: 0.005, default: 0.05},
  {id: 'thinning', label: 'Data: glacier thinning', min: 0, max: 0.1, step: 0.005, default: 0.07}
];
