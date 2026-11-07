// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to the per-frame settings of GPUColorScale (`scale`, `paletteCount`) and the class count of
// the GPUClassBreaks that feeds its domain.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'scale',
    label: 'scale',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['quantize', 'threshold', 'quantile']
  },
  {id: 'paletteCount', label: 'paletteCount', min: 2, max: 8, step: 1, default: 6},
  {id: 'classCount', label: 'Data: breaks classCount', min: 2, max: 8, step: 1, default: 6}
];
