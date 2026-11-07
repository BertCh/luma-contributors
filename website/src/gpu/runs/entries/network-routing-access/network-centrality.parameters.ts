// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// costLimit is the per-frame radius view; pivotNodes sets the sample size (and betweennessScale).
export const parameters: GpuSceneParameter[] = [
  {id: 'costLimit', label: 'costLimit', min: 0.1, max: 2, step: 0.05, default: 0.6},
  {id: 'pivotNodes', label: 'pivotNodes', min: 8, max: 96, step: 8, default: 48}
];
