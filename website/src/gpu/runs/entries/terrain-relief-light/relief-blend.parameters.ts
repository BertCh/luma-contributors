// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPUReliefBlendParameterValues(layers): the slope layer's blend mode and the opacities of the slope, openness and sky view layers.
export const parameters: GpuSceneParameter[] = [
  {id: 'slopeOpacity', label: 'layers[1].opacity', min: 0, max: 1, step: 0.05, default: 0.5},
  {
    id: 'slopeBlendMode',
    label: 'layers[1].blendMode',
    min: 0,
    max: 3,
    step: 1,
    default: 2,
    options: ['normal', 'multiply', 'luminosity', 'overlay']
  },
  {id: 'opennessOpacity', label: 'layers[2].opacity', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'skyViewOpacity', label: 'layers[3].opacity', min: 0, max: 1, step: 0.05, default: 0.5}
];
