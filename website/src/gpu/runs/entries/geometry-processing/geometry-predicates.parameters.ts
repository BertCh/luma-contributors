// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// output picks which written column colors the lines.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['isSimple', 'isRing', 'isClosed', 'isCcw']
  }
];
