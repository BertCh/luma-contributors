// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUSpatialWeightsSummary's `symmetryTolerance`; the noise makes w_ij differ from w_ji.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'symmetryTolerance',
    label: 'symmetryTolerance',
    min: 0,
    max: 0.3,
    step: 0.01,
    default: 0.1
  },
  {
    id: 'noise',
    label: 'Data: weight noise',
    min: 0,
    max: 0.6,
    step: 0.02,
    default: 0.3
  }
];
