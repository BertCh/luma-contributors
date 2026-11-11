// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `miscoverage` is the per-frame alpha of getGPUConformalPredictionParameterValues. The neighbour
// lists (`neighbors.offsets`, `indices`, `weights`) are per-frame buffers: `neighborCount` sets how
// many nearest calibration stations each test station lists and `bandwidth` the Gaussian kernel
// scale of their weights, in map units (picture height 1).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'miscoverage',
    label: 'miscoverage',
    min: 0.05,
    max: 0.5,
    step: 0.01,
    default: 0.1
  },
  {
    id: 'neighborCount',
    label: 'neighborCount',
    min: 4,
    max: 64,
    step: 2,
    default: 32
  },
  {
    id: 'bandwidth',
    label: 'bandwidth',
    min: 0.02,
    max: 0.4,
    step: 0.01,
    default: 0.12
  }
];
