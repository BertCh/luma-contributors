// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPULineLoopRemoval has no per-frame prop; the slider changes how loopy the input is.
export const parameters: GpuSceneParameter[] = [
  {id: 'tangle', label: 'Data: tangle', min: 0.8, max: 3, step: 0.1, default: 2.2}
];
