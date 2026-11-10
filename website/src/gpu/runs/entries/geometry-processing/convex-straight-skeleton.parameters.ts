// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// offsetDistance is the inward parallel offset requested from GPUConvexStraightSkeleton.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'offsetDistance',
    label: 'offsetDistance',
    min: 0.005,
    max: 0.09,
    step: 0.005,
    default: 0.035
  },
  {id: 'stretch', label: 'Data: stretch', min: 0, max: 0.8, step: 0.05, default: 0.2}
];
