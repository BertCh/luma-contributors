// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// ridgeLambda is GPUOrdinaryLeastSquares' per-frame parameter; the other two shape the data.
export const parameters: GpuSceneParameter[] = [
  {id: 'ridgeLambda', label: 'ridgeLambda', min: 0, max: 6, step: 0.1, default: 0},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1.2, step: 0.05, default: 0.4},
  {id: 'rho', label: 'Data: spatial dependence', min: 0, max: 0.9, step: 0.05, default: 0.5}
];
