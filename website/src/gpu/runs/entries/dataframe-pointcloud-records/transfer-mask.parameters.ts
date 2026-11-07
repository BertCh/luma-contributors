// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTransferMask's compile-time `mode`, and the circular parcel selection that sets the
// `source` flags.
export const parameters: GpuSceneParameter[] = [
  {id: 'mode', label: 'mode', min: 0, max: 1, step: 1, default: 0, options: ['any', 'count']},
  {
    id: 'selectionRadius',
    label: 'Data: selection radius',
    min: 0.05,
    max: 0.5,
    step: 0.01,
    default: 0.22
  },
  {
    id: 'selectionCenter',
    label: 'Data: selection center x',
    min: 0.1,
    max: 1.6,
    step: 0.02,
    default: 0.7
  }
];
