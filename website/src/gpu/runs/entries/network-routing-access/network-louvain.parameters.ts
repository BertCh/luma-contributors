// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// resolution is the first word of GPUNetworkLouvain.parameters; minimumJobs filters the job flow weights.
export const parameters: GpuSceneParameter[] = [
  {id: 'resolution', label: 'parameters (resolution)', min: 0.2, max: 3, step: 0.1, default: 1},
  {id: 'minimumJobs', label: 'Minimum workers per link', min: 25, max: 400, step: 5, default: 50}
];
