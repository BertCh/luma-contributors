// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The survey footprint can translate and rotate. Its vertex count is fixed at compile time.
export const parameters: GpuSceneParameter[] = [
  {id: 'sides', label: 'Data: sides', min: 3, max: 8, step: 1, default: 5},
  {
    id: 'separation',
    label: 'survey longitude',
    min: -91.2,
    max: -85.4,
    step: 0.1,
    default: -88.0,
    unit: '°'
  },
  {
    id: 'surveyLatitude',
    label: 'survey latitude',
    min: 46.6,
    max: 49.0,
    step: 0.05,
    default: 47.75,
    unit: '°'
  },
  {
    id: 'rotation',
    label: 'Data: rotation',
    min: 0,
    max: 1.2,
    step: 0.02,
    default: 0.5,
    unit: ' rad'
  }
];
