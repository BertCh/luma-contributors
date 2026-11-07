// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// breakCount is the per-frame count of active rows of the `breaks` view; the interval is the spacing of the
// break values written to that view.
export const parameters: GpuSceneParameter[] = [
  {id: 'breakCount', label: 'breakCount', min: 1, max: 8, step: 1, default: 5},
  {id: 'interval', label: 'breaks (interval)', min: 5, max: 20, step: 1, default: 14}
];
