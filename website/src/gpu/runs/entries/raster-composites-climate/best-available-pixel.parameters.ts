// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'targetDayOfYear', label: 'targetDayOfYear', min: 150, max: 260, step: 2, default: 215},
  {id: 'targetYear', label: 'targetYear', min: 2017, max: 2024, step: 0.1, default: 2021.5},
  {
    id: 'dayOfYearSigma',
    label: 'dayOfYearSigma',
    min: 5,
    max: 60,
    step: 1,
    default: 30,
    unit: ' days'
  },
  {id: 'cloudWeight', label: 'cloudWeight', min: 0, max: 1, step: 0.05, default: 0.3},
  {
    id: 'shown',
    label: 'map shows',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['winning scene', 'NDVI of the winner']
  }
];
