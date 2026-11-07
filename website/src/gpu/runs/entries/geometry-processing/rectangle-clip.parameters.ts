// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The clip rectangle of getGPURectangleClipParameterValues, as a center and a size in world units.
export const parameters: GpuSceneParameter[] = [
  {id: 'centerX', label: 'center x', min: 0.3, max: 1.4, step: 0.05, default: 0.9},
  {id: 'centerY', label: 'center y', min: 0.25, max: 0.75, step: 0.05, default: 0.5},
  {id: 'size', label: 'size', min: 0.2, max: 0.9, step: 0.05, default: 0.6}
];
