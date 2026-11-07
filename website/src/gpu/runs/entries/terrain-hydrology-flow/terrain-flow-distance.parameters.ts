// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// streamThreshold sets the stream cells that stop each path; streams switches the optional
// `streams` prop (compile-time), so paths run to the outlet instead.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'streamThreshold',
    label: 'streamThreshold',
    min: 5,
    max: 120,
    step: 5,
    default: 30,
    unit: ' cells'
  },
  {id: 'streams', label: 'streams', min: 0, max: 1, step: 1, default: 1, options: ['false', 'true']}
];
