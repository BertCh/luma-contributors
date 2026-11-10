// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to the GPUDotDensity parameter buffer (`dotsPerUnit`, `seed`).
export const parameters: GpuSceneParameter[] = [
  {id: 'dotsPerUnit', label: 'dotsPerUnit', min: 0.02, max: 0.3, step: 0.01, default: 0.1},
  {id: 'seed', label: 'seed', min: 1, max: 20, step: 1, default: 7}
];

export const legend: RunLegendItem[] = [
  {label: 'Age 65 and older', color: '#5aa9ff'},
  {label: 'Under age 65', color: '#ffb454'}
];
