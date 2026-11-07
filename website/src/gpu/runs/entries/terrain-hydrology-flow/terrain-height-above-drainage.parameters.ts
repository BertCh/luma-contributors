// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// streamThreshold is the GPUTerrainFlow setting that defines which cells count as drainage.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'streamThreshold',
    label: 'streamThreshold',
    min: 5,
    max: 120,
    step: 5,
    default: 25,
    unit: ' cells'
  }
];
