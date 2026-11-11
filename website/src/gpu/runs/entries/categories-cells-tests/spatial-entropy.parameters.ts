// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `categoryCount` is a compile-time prop of GPUSpatialEntropy. `scrambled` is data: the share of
// tracts whose class is swapped with a random other tract's, which destroys the spatial pattern.
export const parameters: GpuSceneParameter[] = [
  {id: 'categoryCount', label: 'categoryCount', min: 2, max: 5, step: 1, default: 3},
  {id: 'scrambled', label: 'Data: scrambled', min: 0, max: 1, step: 0.05, default: 0, unit: ''}
];
