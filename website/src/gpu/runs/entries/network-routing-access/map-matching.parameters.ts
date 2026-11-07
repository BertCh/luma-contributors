// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// sigma, beta and searchRadius are GPUMapMatching's per-frame `parameters` words.
export const parameters: GpuSceneParameter[] = [
  {id: 'sigma', label: 'sigma', min: 0.003, max: 0.04, step: 0.001, default: 0.015},
  {id: 'searchRadius', label: 'searchRadius', min: 0.01, max: 0.12, step: 0.005, default: 0.06},
  {id: 'noise', label: 'Data: GPS noise', min: 0, max: 0.03, step: 0.002, default: 0.012}
];
