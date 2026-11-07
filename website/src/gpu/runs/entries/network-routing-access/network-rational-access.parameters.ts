// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tau and rho are the travel and supply normalization words of GPUNetworkRationalAccess.parameters.
export const parameters: GpuSceneParameter[] = [
  {id: 'tau', label: 'tau', min: 0.2, max: 6, step: 0.2, default: 2},
  {id: 'rho', label: 'rho', min: 0.2, max: 6, step: 0.2, default: 1},
  {id: 'maximumCycles', label: 'maximumCycles', min: 1, max: 64, step: 1, default: 24}
];
