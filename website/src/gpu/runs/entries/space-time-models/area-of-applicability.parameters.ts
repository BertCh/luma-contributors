// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `predictorCount` is fixed at construction (the first bands of red, near infrared, shortwave
// infrared, green, blue and shortwave infrared 2). `weights` is the per-frame variable importance of
// the red predictor (the others stay 1). The training box is the pointer-owned `mask`; its radius is a
// data choice.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'predictorCount',
    label: 'predictorCount',
    min: 2,
    max: 6,
    step: 1,
    default: 4
  },
  {id: 'weights', label: 'weights', min: 0, max: 3, step: 0.25, default: 1},
  {
    id: 'discRadius',
    label: 'Data: training radius',
    min: 3,
    max: 25,
    step: 1,
    default: 9,
    unit: '%'
  }
];
