// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUNormalizeGeometry has no per-frame prop; the slider shuffles the input tree.
export const parameters: GpuSceneParameter[] = [
  {id: 'shuffle', label: 'Data: child order', min: 0, max: 9, step: 1, default: 3}
];
