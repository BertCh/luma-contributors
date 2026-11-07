// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumVertices caps the vertices of each convex piece (a constructor prop).
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumVertices', label: 'maximumVertices', min: 3, max: 12, step: 1, default: 5},
  {id: 'irregularity', label: 'Data: irregularity', min: 0.1, max: 0.7, step: 0.05, default: 0.45}
];
