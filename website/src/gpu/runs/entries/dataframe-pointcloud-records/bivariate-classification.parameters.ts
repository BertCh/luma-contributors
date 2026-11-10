// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to the per-frame `classCountX` and `classCountY` of GPUBivariateClassification.
export const parameters: GpuSceneParameter[] = [
  {id: 'classCountX', label: 'classCountX', min: 2, max: 4, step: 1, default: 3},
  {id: 'classCountY', label: 'classCountY', min: 2, max: 4, step: 1, default: 3},
  {id: 'maskShare', label: 'mask', min: 20, max: 100, step: 10, default: 100, unit: '% of map'}
];

export const legend: RunLegendItem[] = [
  {label: 'x class rises: more blue', color: '#4a8fc4'},
  {label: 'y class rises: more orange', color: '#d9873f'},
  {label: 'key squares: joint class, size = rows in it', color: '#7a6a8a'}
];
