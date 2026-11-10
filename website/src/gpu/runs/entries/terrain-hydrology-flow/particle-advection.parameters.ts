// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// timeStep, dropRate and speedScale are per-frame settings of the packer.
export const parameters: GpuSceneParameter[] = [
  {id: 'timeStep', label: 'timeStep', min: 0.2, max: 4, step: 0.2, default: 0.6},
  {id: 'speedScale', label: 'speedScale', min: 0.5, max: 3, step: 0.25, default: 1},
  {id: 'dropRate', label: 'dropRate', min: 0, max: 1, step: 0.05, default: 0.05}
];
