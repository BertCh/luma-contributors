// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTransitionDiagnostics' `classCount` and `minimumTransitions`; periodLag varies the counts.
export const parameters: GpuSceneParameter[] = [
  {id: 'classCount', label: 'classCount', min: 3, max: 8, step: 1, default: 5},
  {id: 'minimumTransitions', label: 'minimumTransitions', min: 1, max: 120, step: 1, default: 60},
  {id: 'periodLag', label: 'Data: periodLag', min: 1, max: 4, step: 1, default: 1}
];
