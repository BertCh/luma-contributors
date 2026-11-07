// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Both sliders rewrite input buffers per frame: `size` scales the zones about their centers and
// `shift` slides the points east.
export const parameters: GpuSceneParameter[] = [
  {id: 'size', label: 'Data: size', min: 0.6, max: 1.4, step: 0.02, default: 1},
  {id: 'shift', label: 'Data: shift', min: -0.3, max: 0.3, step: 0.01, default: 0}
];
