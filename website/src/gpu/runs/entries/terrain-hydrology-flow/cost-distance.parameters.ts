// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// costLimit is a per-frame setting; frictionScale is the scale of the `frictionParameters` view; sourceCount
// is the per-frame `sourceCount` view that activates the first rows of `sources`.
export const parameters: GpuSceneParameter[] = [
  {id: 'costLimit', label: 'costLimit', min: 0.2, max: 8, step: 0.2, default: 4},
  {
    id: 'frictionScale',
    label: 'frictionParameters (scale)',
    min: 0.5,
    max: 3,
    step: 0.25,
    default: 1
  },
  {id: 'sourceCount', label: 'sourceCount', min: 1, max: 4, step: 1, default: 2}
];
