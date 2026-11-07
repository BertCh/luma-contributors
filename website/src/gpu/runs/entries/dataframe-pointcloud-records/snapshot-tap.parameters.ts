// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// Bound to GPUSnapshotTap's `afterNode` (the iteration whose last node the tap follows), `views`
// (how many named views it copies) and `iteration` (the frame's own coordinate, free of where the
// tap sits). `iterations` is the length of the relaxation being observed.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'afterNode',
    label: 'afterNode',
    min: 1,
    max: 6,
    step: 1,
    default: 3,
    unit: ' (iteration k)'
  },
  {id: 'views', label: 'views', min: 1, max: 3, step: 1, default: 2, unit: ' views'},
  {id: 'iteration', label: 'iteration', min: 0, max: 9, step: 1, default: 7},
  {id: 'iterations', label: 'Data: iterations', min: 3, max: 6, step: 1, default: 5}
];

export const legend: RunLegendItem[] = [
  {label: 'the tap node, a copy node in the graph', color: '#ff6b6b'},
  {label: 'highlighted edge: its afterNode dependency', color: '#ff6b6b'},
  {label: 'grey: the observed relaxation (node boxes and dependencies)', color: '#5b6580'},
  {label: 'ticks: frame iteration coordinate (bright tick = this frame)', color: '#ffb454'},
  {label: 'maps: the tapped views, one per view (state, residual, deviation)', color: '#5aa9ff'}
];
