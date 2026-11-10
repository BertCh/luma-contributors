// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// `capacity` and `slots` are GPUSnapshotArena props. `iterations` is the length of the relaxation
// being watched: when it exceeds `capacity` the taps decimate (every n-th iteration) so frames fit.
export const parameters: GpuSceneParameter[] = [
  {id: 'capacity', label: 'capacity', min: 1, max: 8, step: 1, default: 5, unit: ' frames'},
  {id: 'slots', label: 'slots', min: 1, max: 3, step: 1, default: 3, unit: ' per frame'},
  {id: 'iterations', label: 'Data: iterations', min: 2, max: 12, step: 1, default: 5}
];

export const legend: RunLegendItem[] = [
  {label: 'slot state: tract values (one column per frame)', color: '#5aa9ff'},
  {label: 'slot residual: change since the previous iteration', color: '#ff9a3c'},
  {label: 'slot deviation: drift from the start (blue below, red above)', color: '#ff6b6b'},
  {label: 'empty outline: capacity not yet used', color: '#5b6580'},
  {label: 'bottom bar: bytes filled of the readback buffer', color: '#4fd1c5'}
];
