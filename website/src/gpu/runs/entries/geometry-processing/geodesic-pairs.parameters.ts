// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUGeodesicPairs has no per-frame prop; the slider changes how far apart the pairs are.
export const parameters: GpuSceneParameter[] = [
  {id: 'spread', label: 'Data: spread', min: 0.2, max: 1, step: 0.05, default: 0.8}
];
