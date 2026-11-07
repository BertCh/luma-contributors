// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUCellGeometry's `family` prop (compile time) and the resolution of the keys it decodes.
export const parameters: GpuSceneParameter[] = [
  {id: 'family', label: 'family', min: 0, max: 1, step: 1, default: 0, options: ['h3', 'quadbin']},
  {id: 'resolution', label: 'Data: resolution', min: 3, max: 6, step: 1, default: 5}
];
