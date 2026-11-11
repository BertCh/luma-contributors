// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `order` is the compile-time HEALPix depth of GPUHealpixPointToCell (`12 * 4^order` equal-area
// cells). `keptShare` is the share of coastline vertices whose per-row `mask` is 1; masked rows get the
// zero key and no cell.
export const parameters: GpuSceneParameter[] = [
  {id: 'order', label: 'order', min: 1, max: 5, step: 1, default: 3},
  {id: 'keptShare', label: 'mask: kept share', min: 0.1, max: 1, step: 0.1, default: 1}
];
