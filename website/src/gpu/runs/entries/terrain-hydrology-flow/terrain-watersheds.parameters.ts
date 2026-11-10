// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the number of active pour points in GPUTerrainWatersheds' `pourPoints` view.
export const parameters: GpuSceneParameter[] = [
  {id: 'pourPointCount', label: 'pourPoints', min: 1, max: 8, step: 1, default: 6}
];
