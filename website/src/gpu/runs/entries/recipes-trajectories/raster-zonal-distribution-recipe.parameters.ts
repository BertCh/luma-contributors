// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// probability is the per-frame `probabilities` view of addRasterZonalDistributionRecipe (the
// quantile drawn per zone); the relief and roughness shape the raster values.
export const parameters: GpuSceneParameter[] = [
  {id: 'probability', label: 'probabilities', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'relief', label: 'Data: relief', min: 0.2, max: 2, step: 0.1, default: 1},
  {id: 'roughness', label: 'Data: roughness', min: 0, max: 1, step: 0.05, default: 0.3}
];
