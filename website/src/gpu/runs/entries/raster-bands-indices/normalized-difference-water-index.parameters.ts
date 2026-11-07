// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['mcfeeters', 'gao']
  },
  {id: 'epsilon', label: 'epsilon', min: 0, max: 0.4, step: 0.02, default: 0}
];
