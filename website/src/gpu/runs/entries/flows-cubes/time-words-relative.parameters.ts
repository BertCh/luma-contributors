// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// unit is the temporalContext.unit of the Int64 words and origin the instant the output counts
// from, in hours after the first flash. Both are compile-time props of GPUTimeWordsRelative.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'unit',
    label: 'temporalContext.unit',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['second', 'millisecond', 'microsecond']
  },
  {id: 'origin', label: 'origin', min: 0, max: 18, step: 0.5, default: 0, unit: 'h'}
];
