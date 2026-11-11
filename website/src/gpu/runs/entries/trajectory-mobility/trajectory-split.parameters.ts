// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// mode is compile-time; maximumGap is the per-frame gap rule of mode 'gap', in playback seconds
// (a vessel hour is 25 s).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'mode',
    label: 'mode',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['gap', 'stop', 'value-change']
  },
  {id: 'maximumGap', label: 'maximumGap', min: 4, max: 40, step: 1, default: 10, unit: 'playback s'}
];
