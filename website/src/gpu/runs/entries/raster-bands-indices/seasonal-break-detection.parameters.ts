// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'minimumBICGain', label: 'minimumBICGain', min: 0, max: 60, step: 2, default: 10},
  {id: 'shift', label: 'Data: post-fire change multiplier', min: 0, max: 2, step: 0.1, default: 1},
  {id: 'noise', label: 'Data: noise', min: 0, max: 0.3, step: 0.01, default: 0.08}
];
