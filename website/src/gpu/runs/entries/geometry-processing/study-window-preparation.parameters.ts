// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// GPUStudyWindowPreparation has no per-frame prop; the sliders reshape the window polygon.
export const parameters: GpuSceneParameter[] = [
  {id: 'irregularity', label: 'Data: irregularity', min: 0, max: 0.8, step: 0.05, default: 0.4},
  {id: 'radius', label: 'Data: radius', min: 0.5, max: 1.4, step: 0.05, default: 1}
];

export const legend: RunLegendItem[] = [
  {label: 'polygon: the study window', color: '#9aa0ad'},
  {label: 'orange box: summary bounds', color: '#f2994d'},
  {label: 'left bar: area (x10); right bar: perimeter', color: '#8a6bd1'}
];
