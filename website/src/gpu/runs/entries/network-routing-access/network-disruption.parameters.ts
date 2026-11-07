// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// closures rewrites the recipe's edgeMask each frame; threshold is its accessibility parameter word.
export const parameters: GpuSceneParameter[] = [
  {id: 'closures', label: 'edgeMask (closed)', min: 0, max: 0.5, step: 0.02, default: 0.2},
  {id: 'threshold', label: 'threshold', min: 0.2, max: 1.5, step: 0.05, default: 0.7}
];
