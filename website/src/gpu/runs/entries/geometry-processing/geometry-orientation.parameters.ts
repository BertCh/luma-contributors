// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// exteriorClockwise picks the winding convention (GeoJSON counter-clockwise, shapefile clockwise).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'exteriorClockwise',
    label: 'exteriorClockwise',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  }
];
