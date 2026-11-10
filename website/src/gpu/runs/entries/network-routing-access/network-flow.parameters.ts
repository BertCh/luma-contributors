// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// requiredFlow is the third word of GPUNetworkFlow.parameters; capacity scales the capacities view.
export const parameters: GpuSceneParameter[] = [
  {id: 'requiredFlow', label: 'parameters (required flow)', min: 1, max: 30, step: 1, default: 12},
  {id: 'capacity', label: 'capacities', min: 1, max: 8, step: 1, default: 4}
];
