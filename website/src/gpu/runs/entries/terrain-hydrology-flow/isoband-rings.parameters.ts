// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// breakCount is the per-frame count of active break rows; the interval is the spacing of the break values.
export const parameters: GpuSceneParameter[] = [
  {id: 'breakCount', label: 'breakCount', min: 1, max: 8, step: 1, default: 3},
  {id: 'interval', label: 'breaks (interval)', min: 8, max: 25, step: 1, default: 18}
];
