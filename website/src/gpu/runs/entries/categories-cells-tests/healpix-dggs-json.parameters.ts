// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `order` is the HEALPix order of the cells and `ancestorOrder` the order of the ancestral zone that
// createHealpixDGGSJSON packs (clamped to `order`). The pointer chooses which ancestor.
export const parameters: GpuSceneParameter[] = [
  {id: 'order', label: 'order', min: 2, max: 4, step: 1, default: 3},
  {id: 'ancestorOrder', label: 'ancestorOrder', min: 0, max: 2, step: 1, default: 0}
];
