// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumIterations and tolerance are GPUProbitRegression props; noise is the latent error scale.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumIterations', label: 'maximumIterations', min: 1, max: 12, step: 1, default: 8},
  {id: 'noise', label: 'Data: latent noise', min: 0.5, max: 2.5, step: 0.1, default: 1}
];
