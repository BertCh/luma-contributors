// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// departure is GPUNetworkTimeRouting's per-frame `departure` view; slope scales the affine costs.
export const parameters: GpuSceneParameter[] = [
  {id: 'departure', label: 'departure', min: 0, max: 2, step: 0.05, default: 0.5},
  {id: 'slope', label: 'Data: slope', min: 0, max: 1, step: 0.05, default: 0.4}
];
