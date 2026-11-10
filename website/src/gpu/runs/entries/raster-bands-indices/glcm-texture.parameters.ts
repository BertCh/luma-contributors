// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'feature',
    label: 'Feature shown',
    min: 0,
    max: 6,
    step: 1,
    default: 0,
    options: ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy']
  },
  {id: 'levels', label: 'levels', min: 2, max: 16, step: 1, default: 8},
  {id: 'radius', label: 'radius', min: 1, max: 6, step: 1, default: 2, unit: 'cells'},
  {
    id: 'direction',
    label: 'offset',
    min: 0,
    max: 3,
    step: 1,
    default: 0,
    options: ['[1, 0]', '[0, 1]', '[1, 1]', '[-1, 1]']
  },
  {id: 'roughness', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.6}
];
