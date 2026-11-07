// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Both are the operation's compile-time props (maximumCutDepth is in elevation units).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumCutDepth',
    label: 'maximumCutDepth',
    min: 0.002,
    max: 0.06,
    step: 0.002,
    default: 0.05
  },
  {id: 'maximumSteps', label: 'maximumSteps', min: 1, max: 12, step: 1, default: 8}
];
