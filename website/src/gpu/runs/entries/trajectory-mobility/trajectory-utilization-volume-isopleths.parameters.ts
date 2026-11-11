// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// level is the per-frame setting of GPUUtilizationVolumeIsopleths. The density it reads comes from
// GPUBrownianBridgeUtilization; its motion variance is the one input knob.
export const parameters: GpuSceneParameter[] = [
  {id: 'level', label: 'level', min: 0.05, max: 0.99, step: 0.01, default: 0.9},
  {
    id: 'motionVariance',
    label: 'Data: motionVariances',
    min: 0.00005,
    max: 0.002,
    step: 0.00005,
    default: 0.00045
  }
];
