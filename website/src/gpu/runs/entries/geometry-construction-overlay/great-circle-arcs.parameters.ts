// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'minimumSegments', label: 'minimumSegments', min: 2, max: 48, step: 1, default: 4},
  {
    id: 'maximumSegmentLength',
    label: 'maximumSegmentLength',
    min: 100,
    max: 3000,
    step: 100,
    default: 1200,
    unit: 'km'
  }
];
