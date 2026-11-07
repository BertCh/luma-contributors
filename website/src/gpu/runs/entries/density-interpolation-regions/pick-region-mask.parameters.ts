// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to GPUPickRegionMask's `batchIndex`; the pick radius varies which objects were picked.
export const parameters: GpuSceneParameter[] = [
  {id: 'batchIndex', label: 'batchIndex', min: 0, max: 2, step: 1, default: 1},
  {id: 'pickRadius', label: 'Data: pick radius', min: 0.05, max: 0.4, step: 0.01, default: 0.22}
];

export const legend: RunLegendItem[] = [
  {label: 'layer batch 0', color: '#2666d9'},
  {label: 'layer batch 1', color: '#e6731a'},
  {label: 'layer batch 2', color: '#1aa68c'},
  {label: 'picked rows of batchIndex', color: '#ffd926'}
];
