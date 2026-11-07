// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'probability', label: 'probabilities[0]', min: 0, max: 1, step: 0.05, default: 0.5},
  {
    id: 'quantileMethod',
    label: 'quantileMethod',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['exactextract', 'inverse-cdf']
  },
  {id: 'gridWidth', label: 'grid.width', min: 6, max: 60, step: 2, default: 12, unit: 'cells'}
];
