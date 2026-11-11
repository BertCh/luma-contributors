// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The per-frame contents of the `motionVariances` view (world units squared per playback second;
// 4.5e-4 is what estimateBrownianMotionVariance finds for this track) and of `locationErrors`
// (world units).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'motionVariance',
    label: 'motionVariances',
    min: 0.00005,
    max: 0.002,
    step: 0.00005,
    default: 0.00045
  },
  {id: 'locationError', label: 'locationErrors', min: 0, max: 0.03, step: 0.001, default: 0.005}
];
