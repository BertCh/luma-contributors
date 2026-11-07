// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// offsetDistance is the inward offset requested from GPUVectorStraightSkeleton.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'offsetDistance',
    label: 'offsetDistance',
    min: 0.004,
    max: 0.06,
    step: 0.004,
    default: 0.02
  },
  {id: 'irregularity', label: 'Data: irregularity', min: 0.1, max: 0.7, step: 0.05, default: 0.4}
];
