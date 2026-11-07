// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tolerance is the snap radius in world units; noise offsets the input vertices from the reference.
export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0, max: 0.05, step: 0.002, default: 0.02},
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.6}
];
