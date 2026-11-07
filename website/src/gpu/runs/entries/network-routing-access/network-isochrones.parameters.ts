// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// breakWidth scales the per-frame costBreaks; bufferRadius is a word of the `parameters` view.
export const parameters: GpuSceneParameter[] = [
  {id: 'breakWidth', label: 'costBreaks (width)', min: 0.1, max: 0.6, step: 0.02, default: 0.25},
  {
    id: 'bufferRadius',
    label: 'parameters (bufferRadius)',
    min: 0.01,
    max: 0.08,
    step: 0.005,
    default: 0.035
  },
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 3, step: 0.1, default: 0.6}
];
