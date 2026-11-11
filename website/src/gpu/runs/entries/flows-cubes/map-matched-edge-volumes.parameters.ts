// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// sigma and searchRadius are GPUMapMatching's per-frame parameters, which decide the routes that
// GPUMapMatchedEdgeVolumes counts. trackCount is the compile-time shape of both operations.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'trackCount',
    label: 'trackCount',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['24', '48', '96']
  },
  {
    id: 'searchRadius',
    label: 'matching searchRadius',
    min: 0.01,
    max: 0.12,
    step: 0.005,
    default: 0.05
  },
  {id: 'sigma', label: 'matching sigma', min: 0.003, max: 0.04, step: 0.001, default: 0.015},
  {id: 'noise', label: 'Data: GPS noise', min: 0, max: 0.04, step: 0.002, default: 0.012}
];
