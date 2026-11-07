// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTransitionMatrix's `classCount`, `periodLag` and `minimumTransitions`.
export const parameters: GpuSceneParameter[] = [
  {id: 'classCount', label: 'classCount', min: 3, max: 8, step: 1, default: 5},
  {id: 'periodLag', label: 'periodLag', min: 1, max: 4, step: 1, default: 1},
  {id: 'minimumTransitions', label: 'minimumTransitions', min: 1, max: 120, step: 1, default: 30},
  {id: 'period', label: 'Show: period', min: 0, max: 5, step: 1, default: 0}
];
