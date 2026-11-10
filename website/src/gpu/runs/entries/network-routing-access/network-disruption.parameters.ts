// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// closures rewrites the recipe's edgeMask each frame; threshold is its accessibility parameter word.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'closureRadius',
    label: 'edgeMask (closure radius)',
    min: 0.03,
    max: 0.3,
    step: 0.01,
    default: 0.1
  },
  {id: 'threshold', label: 'threshold', min: 0.2, max: 1.5, step: 0.05, default: 0.7}
];
