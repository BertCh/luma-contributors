// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'method',
    label: 'method',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['nearest', 'inverse-distance', 'gdal-four-quadrant']
  },
  {
    id: 'maximumDistance',
    label: 'maximumDistance',
    min: 1,
    max: 24,
    step: 1,
    default: 12,
    unit: 'cells'
  },
  {id: 'gaps', label: 'Data: gaps', min: 0.5, max: 0.8, step: 0.01, default: 0.62}
];
