// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUGaussianPredictiveScores has no scalar props: the predictive mean and variance are buffers.
// The pad writes them per frame, so the parameters are labelled as data. The standard deviation
// is in log-rain units (the unit of the observations).
export const parameters: GpuSceneParameter[] = [
  {id: 'spread', label: 'Data: predictive sd', min: 0.1, max: 4, step: 0.05, default: 1.5},
  {id: 'bias', label: 'Data: bias', min: -1, max: 3, step: 0.05, default: 1.5},
  {id: 'wetSpread', label: 'Data: sd rises with rain', min: 0, max: 1, step: 0.05, default: 0}
];
