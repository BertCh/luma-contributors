// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tolerance is the coordinate tolerance of GPUCanonicalGeometryEquality; displacement moves leaf k by k steps.
export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0, max: 0.02, step: 0.001, default: 0.006},
  {
    id: 'displacement',
    label: 'Data: displacement',
    min: 0,
    max: 0.002,
    step: 0.0001,
    default: 0.001
  }
];
