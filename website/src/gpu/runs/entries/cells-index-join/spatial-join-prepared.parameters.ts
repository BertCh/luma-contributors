// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The zones are prepared once; `time` moves the left points every frame (a buffer rewrite) and the
// prepared index is reused. `spread` scales the point cloud.
export const parameters: GpuSceneParameter[] = [
  {id: 'time', label: 'Data: time', min: 0, max: 1, step: 0.01, default: 0.3},
  {id: 'spread', label: 'Data: spread', min: 0.15, max: 0.8, step: 0.01, default: 0.4}
];
