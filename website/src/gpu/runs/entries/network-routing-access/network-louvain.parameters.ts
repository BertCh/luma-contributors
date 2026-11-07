// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// resolution is the first word of GPUNetworkLouvain.parameters; sigma shapes the weights data.
export const parameters: GpuSceneParameter[] = [
  {id: 'resolution', label: 'parameters (resolution)', min: 0.2, max: 3, step: 0.1, default: 1},
  {id: 'sigma', label: 'Data: tie reach', min: 0.05, max: 0.3, step: 0.01, default: 0.12}
];
