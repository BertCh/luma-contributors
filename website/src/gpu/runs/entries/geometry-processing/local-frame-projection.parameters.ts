// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The frame of GPULocalFrameProjection: its origin latitude and its projection kind.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'originLatitude',
    label: 'origin latitude',
    min: -50,
    max: 60,
    step: 5,
    default: 45,
    unit: '°'
  },
  {
    id: 'kind',
    label: 'kind',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['azimuthal-equidistant', 'lambert-equal-area']
  }
];
