// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode shown',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['EOF 1', 'EOF 2', 'EOF 3', 'EOF 4']
  },
  {id: 'date', label: 'date shown (scene index)', min: 0, max: 15, step: 1, default: 8},
  {id: 'sliceCount', label: 'sliceCount', min: 6, max: 16, step: 1, default: 12}
];
