// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// angleCost, leftTurnCost and rightTurnCost are GPUNetworkLineGraph's per-frame `parameters` words.
export const parameters: GpuSceneParameter[] = [
  {id: 'angleCost', label: 'angleCost', min: 0, max: 0.4, step: 0.02, default: 0.12},
  {id: 'leftTurnCost', label: 'leftTurnCost', min: 0, max: 0.4, step: 0.02, default: 0.15},
  {id: 'rightTurnCost', label: 'rightTurnCost', min: 0, max: 0.4, step: 0.02, default: 0}
];
