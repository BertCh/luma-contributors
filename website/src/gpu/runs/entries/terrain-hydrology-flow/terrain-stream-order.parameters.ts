// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// streamThreshold is the GPUTerrainFlow setting that defines the stream mask the order is ranked on.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'streamThreshold',
    label: 'GPUTerrainFlow: streamThreshold',
    min: 5,
    max: 120,
    step: 5,
    default: 25,
    unit: ' cells'
  }
];

export const legend: RunLegendItem[] = [
  {label: 'order 1', color: '#66bfff'},
  {label: 'order 2', color: '#1a80f2'},
  {label: 'order 3', color: '#7333d9'},
  {label: 'order 4', color: '#d91a80'},
  {label: 'order 5+', color: '#990026'}
];
