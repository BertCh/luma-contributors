// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'cancelOpposingSegments',
    label: 'cancelOpposingSegments',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {
    id: 'vertexTolerance',
    label: 'vertexTolerance',
    min: 0.0001,
    max: 0.02,
    step: 0.0005,
    default: 0.006
  },
  {id: 'jitter', label: 'Data: jitter', min: 0, max: 0.012, step: 0.0005, default: 0.003}
];
