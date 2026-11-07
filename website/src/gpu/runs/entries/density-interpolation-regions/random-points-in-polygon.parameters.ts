// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `pointsPerPolygon` is the per-polygon `counts` input; `seed` is the sampling seed parameter.
export const parameters: GpuSceneParameter[] = [
  {id: 'pointsPerPolygon', label: 'counts', min: 1, max: 60, step: 1, default: 20},
  {id: 'seed', label: 'seed', min: 1, max: 20, step: 1, default: 3}
];
