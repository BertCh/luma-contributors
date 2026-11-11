// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// memberCount and fair are compile-time props of GPUEnsembleScores; the spread only rewrites the
// ensemble buffer.
export const parameters: GpuSceneParameter[] = [
  {id: 'memberCount', label: 'memberCount', min: 2, max: 32, step: 2, default: 4},
  {id: 'fair', label: 'fair', min: 0, max: 1, step: 1, default: 0, options: ['false', 'true']},
  {id: 'spread', label: 'Data: ensemble spread', min: 0, max: 3, step: 0.05, default: 1}
];
