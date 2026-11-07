// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `minimumResolution` is a GPUCellCompaction operation prop (compile time). `keep` is the share of
// resolution-6 cells present in the input; it rewrites the per-row `mask` view, no recompile.
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumResolution', label: 'minimumResolution', min: 4, max: 6, step: 1, default: 4},
  {id: 'keep', label: 'Data: keep', min: 0.6, max: 1, step: 0.01, default: 0.93}
];
