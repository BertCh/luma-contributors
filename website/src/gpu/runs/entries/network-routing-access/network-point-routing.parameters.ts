// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The two fractions are GPUNetworkPointRouting's per-query `fractions` (position along the source
// and target roads); congestion rewrites the edge costs.
export const parameters: GpuSceneParameter[] = [
  {id: 'sourceFraction', label: 'fractions (source)', min: 0, max: 1, step: 0.05, default: 0.3},
  {id: 'targetFraction', label: 'fractions (target)', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 3, step: 0.1, default: 0.8}
];
