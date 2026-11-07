// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `maximumLevel` and `mode` are GPUS2PolygonCovering props (compile time); `size` rewrites the
// polygon vertices per frame.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumLevel', label: 'maximumLevel', min: 7, max: 11, step: 1, default: 10},
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['exterior', 'interior']
  },
  {id: 'size', label: 'Data: size', min: 0.5, max: 1.2, step: 0.05, default: 1}
];
