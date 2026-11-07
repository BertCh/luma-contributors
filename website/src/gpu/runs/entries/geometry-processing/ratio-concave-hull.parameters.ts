// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// ratio is a constructor prop of GPURatioConcaveHull; opening sets how wide the C is open.
export const parameters: GpuSceneParameter[] = [
  {id: 'ratio', label: 'ratio', min: 0.02, max: 1, step: 0.02, default: 0.2},
  {id: 'opening', label: 'Data: opening', min: 0.1, max: 0.7, step: 0.05, default: 0.3}
];
