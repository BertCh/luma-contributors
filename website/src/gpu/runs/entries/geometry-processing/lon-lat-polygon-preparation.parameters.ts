// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumSegmentDegrees is the per-frame densification step (great-circle edges); antimeridian and
// poleRings are constructor props.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumSegmentDegrees',
    label: 'maximumSegmentDegrees',
    min: 2,
    max: 40,
    step: 2,
    default: 10,
    unit: '°'
  },
  {
    id: 'antimeridian',
    label: 'antimeridian',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['split', 'reject']
  },
  {
    id: 'poleRings',
    label: 'poleRings',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['cap', 'reject']
  }
];
