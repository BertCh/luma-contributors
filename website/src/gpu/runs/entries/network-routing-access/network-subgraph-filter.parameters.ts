// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The edge speed range is GPUNetworkSubgraphFilter's per-frame `parameters` view.
export const parameters: GpuSceneParameter[] = [
  {id: 'minimumSpeed', label: 'edgeRange minimum', min: 0, max: 1, step: 0.05, default: 0.3},
  {id: 'maximumSpeed', label: 'edgeRange maximum', min: 0, max: 1, step: 0.05, default: 0.9}
];
