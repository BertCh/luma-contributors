// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The column picks which output of GPUNetworkAnalyticsColumns is drawn.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'column',
    label: 'column',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['components', 'degree', 'normalized degree']
  },
  {id: 'closures', label: 'Data: closures', min: 0, max: 0.6, step: 0.05, default: 0.35}
];
