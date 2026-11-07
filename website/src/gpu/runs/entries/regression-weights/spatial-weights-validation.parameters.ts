// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Validation has no tunable props; the slider decides how many rows of the uploaded CSR are broken.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'defects',
    label: 'Data: broken rows',
    min: 0,
    max: 0.4,
    step: 0.02,
    default: 0.12
  }
];
