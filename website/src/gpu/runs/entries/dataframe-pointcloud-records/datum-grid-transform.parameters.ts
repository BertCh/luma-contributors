// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUDatumGridTransform's compile-time `direction` and `maximumIterations`, and the size
// and wavelength of the shift grid samples.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'direction',
    label: 'direction',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['forward', 'inverse']
  },
  {id: 'maximumIterations', label: 'maximumIterations', min: 1, max: 12, step: 1, default: 8},
  {id: 'shift', label: 'Data: shift size', min: 0, max: 0.12, step: 0.005, default: 0.05}
];
