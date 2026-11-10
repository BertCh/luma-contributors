// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// stepLength and minimumSpeed are per-frame settings of the packer; seed jitters the seed lattice.
export const parameters: GpuSceneParameter[] = [
  {id: 'stepLength', label: 'stepLength', min: 0.001, max: 0.006, step: 0.0005, default: 0.003},
  {id: 'minimumSpeed', label: 'minimumSpeed', min: 0, max: 0.4, step: 0.01, default: 0.03},
  {id: 'seed', label: 'seed', min: 0, max: 12, step: 1, default: 3}
];
