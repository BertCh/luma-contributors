// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumIterations is a compile-time prop of addSparseResistanceConnectivityRecipe; the gap width
// rewrites the conductances (zero conductance is a barrier) and the sink row moves a compile-time vertex.
export const parameters: GpuSceneParameter[] = [
  {id: 'gapWidth', label: 'Data: gapWidth', min: 1, max: 12, step: 1, default: 4, unit: 'rows'},
  {id: 'sinkRow', label: 'Data: sinkRow', min: 0, max: 31, step: 1, default: 24, unit: 'row'},
  {
    id: 'maximumIterations',
    label: 'maximumIterations',
    min: 0,
    max: 2,
    step: 1,
    default: 2,
    options: ['16', '64', '1024']
  }
];
