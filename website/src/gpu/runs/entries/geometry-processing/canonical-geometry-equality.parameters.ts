// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tolerance is the per-coordinate tolerance of GPUCanonicalGeometryEquality. The shift is how far
// delivery B stores the moved vertex from where delivery A has it; its steps never coincide with a
// tolerance step, so the comparison is never decided by float32 rounding.
export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0, max: 0.028, step: 0.004, default: 0.02},
  {
    id: 'shift',
    label: 'Data: vertex shift',
    min: 0.002,
    max: 0.05,
    step: 0.004,
    default: 0.026
  }
];
