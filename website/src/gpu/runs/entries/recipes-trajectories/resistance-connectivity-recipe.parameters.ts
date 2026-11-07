// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// source and sink (rows on the left and right edge) are compile-time props of
// addResistanceConnectivityRecipe; the barrier strength rewrites the conductances.
export const parameters: GpuSceneParameter[] = [
  {id: 'source', label: 'source', min: 0, max: 15, step: 1, default: 3, unit: 'row'},
  {id: 'sink', label: 'sink', min: 0, max: 15, step: 1, default: 12, unit: 'row'},
  {id: 'barrier', label: 'Data: barrierStrength', min: 0, max: 0.99, step: 0.01, default: 0.9}
];
