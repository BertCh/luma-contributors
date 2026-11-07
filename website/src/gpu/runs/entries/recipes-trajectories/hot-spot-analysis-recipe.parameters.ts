// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// radius and minimumNeighbors are compile-time props of addHotSpotAnalysisRecipe (lattice source);
// the cluster strength only changes the input values.
export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 1, max: 4, step: 1, default: 1, unit: 'cells'},
  {id: 'minimumNeighbors', label: 'minimumNeighbors', min: 0, max: 8, step: 1, default: 0},
  {id: 'strength', label: 'Data: clusterStrength', min: 0, max: 3, step: 0.1, default: 1.5}
];
