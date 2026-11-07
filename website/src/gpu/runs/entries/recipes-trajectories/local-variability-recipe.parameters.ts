// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// residualPower is a compile-time prop of addLocalVariabilityRecipe; the
// noise slider scales the input scatter (stronger on the right).
export const parameters: GpuSceneParameter[] = [
  {id: 'residualPower', label: 'residualPower', min: 1, max: 2, step: 1, default: 2},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1.5, step: 0.05, default: 0.7}
];
