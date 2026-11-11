// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The permutation count is per-frame; the distance and time band edges are compile-time, so the
// band widths rebuild. The data control exchanges a share of the origin times between events.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'distanceBand',
    label: 'distanceEdges step',
    min: 10,
    max: 60,
    step: 10,
    default: 20,
    unit: 'km'
  },
  {id: 'timeBand', label: 'timeEdges step', min: 0.25, max: 2, step: 0.25, default: 0.5, unit: 'h'},
  {id: 'observedShare', label: 'Data: observed times', min: 0, max: 1, step: 0.05, default: 1}
];
