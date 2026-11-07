// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// levelCount is the per-frame count of active rows of the `levels` view; the interval is the spacing of the
// level values written to that view.
export const parameters: GpuSceneParameter[] = [
  {id: 'levelCount', label: 'levelCount', min: 1, max: 8, step: 1, default: 5},
  {id: 'interval', label: 'levels (interval)', min: 5, max: 20, step: 1, default: 12}
];
