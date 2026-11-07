// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUMinimumClearance has no per-frame prop; irregularity reshapes the input polygons.
export const parameters: GpuSceneParameter[] = [
  {id: 'irregularity', label: 'Data: irregularity', min: 0, max: 0.8, step: 0.05, default: 0.5}
];
