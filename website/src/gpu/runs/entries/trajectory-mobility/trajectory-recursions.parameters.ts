// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// radius is the per-frame content of the `radii` view (world units; the frame is about 25 km across).
// metric only chooses which output column is drawn.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radii[0]', min: 0.01, max: 0.25, step: 0.01, default: 0.03},
  {
    id: 'metric',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['revisits', 'residenceTimes', 'firstPassageTimes']
  }
];
