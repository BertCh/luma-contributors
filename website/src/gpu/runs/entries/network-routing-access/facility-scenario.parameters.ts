// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// openFacilities rewrites the recipe's facilityMask each frame; threshold is its parameter word.
export const parameters: GpuSceneParameter[] = [
  {id: 'openFacilities', label: 'facilityMask (open)', min: 1, max: 8, step: 1, default: 4},
  {id: 'threshold', label: 'threshold', min: 0.1, max: 1.5, step: 0.05, default: 0.5},
  {
    id: 'measure',
    label: 'Output: measure',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['nearestFacilities', 'accessibility']
  }
];
