// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Both sliders reshape the landscape (and so rebuild); `classCount` is also the recipe's own prop.
export const parameters: GpuSceneParameter[] = [
  {id: 'classCount', label: 'classCount', min: 2, max: 6, step: 1, default: 4},
  {id: 'fragmentation', label: 'Data: fragmentation', min: 2, max: 8, step: 1, default: 3}
];
