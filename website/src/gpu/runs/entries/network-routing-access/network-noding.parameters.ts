// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tolerance is GPUNetworkNoding's per-frame `tolerance` view; lineCount sizes its input.
export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0, max: 0.3, step: 0.01, default: 0.01},
  {id: 'lineCount', label: 'Data: lines', min: 4, max: 16, step: 1, default: 10}
];
