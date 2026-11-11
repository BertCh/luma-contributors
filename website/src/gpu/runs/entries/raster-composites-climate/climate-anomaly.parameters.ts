// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import {ACQUISITION_DATES} from './support/acquisitions';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'scene',
    label: 'scene evaluated',
    min: 0,
    max: 15,
    step: 1,
    default: 8,
    options: [...ACQUISITION_DATES]
  },
  {
    id: 'activeSliceCount',
    label: 'activeSliceCount',
    min: 3,
    max: 12,
    step: 1,
    default: 8,
    unit: ' scenes'
  },
  {
    id: 'standardized',
    label: 'standardized',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  }
];
