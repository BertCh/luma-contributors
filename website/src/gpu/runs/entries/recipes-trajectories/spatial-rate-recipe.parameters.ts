// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// islandPolicy is a compile-time prop of addSpatialRateRecipe; islands and small tracts shape the data.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'islandPolicy',
    label: 'islandPolicy',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['undefined', 'own-rate']
  },
  {id: 'islandCount', label: 'Data: islandCount', min: 0, max: 60, step: 4, default: 24},
  {id: 'smallShare', label: 'Data: smallTractShare', min: 0, max: 0.8, step: 0.05, default: 0.4}
];
