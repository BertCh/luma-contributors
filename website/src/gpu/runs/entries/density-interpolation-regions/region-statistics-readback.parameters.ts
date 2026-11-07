// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// `binCount` and `slotCount` are GPURegionStatisticsReadback props. `reads` is how many encodeRead
// calls are made before any is read: with more reads than slots the ring is under backpressure and
// the extra calls return null. The region size moves the GPURegionStatistics region.
export const parameters: GpuSceneParameter[] = [
  {id: 'binCount', label: 'binCount', min: 4, max: 24, step: 2, default: 12},
  {id: 'slotCount', label: 'slotCount', min: 1, max: 6, step: 1, default: 3},
  {id: 'reads', label: 'Data: encodeRead calls', min: 1, max: 8, step: 1, default: 5},
  {id: 'size', label: 'Data: region size', min: 0.05, max: 0.5, step: 0.01, default: 0.24}
];

export const legend: RunLegendItem[] = [
  {label: 'left: points in the region (bright) from the GPU mask', color: '#5aa9ff'},
  {label: 'bars: histogram read back to the CPU', color: '#ffb454'},
  {label: 'lines: mean (long) and minimum, maximum (ticks) read back', color: '#ffffff'},
  {label: 'ring slots: acquired by encodeRead (filled) or free (outline)', color: '#4fd1c5'},
  {label: 'red: encodeRead returned null (backpressure, frame dropped)', color: '#ff6b6b'},
  {label: 'bottom bar: selectedCount of all points', color: '#8bd450'}
];
