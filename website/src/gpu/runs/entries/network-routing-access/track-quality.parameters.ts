// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumSpeed and maximumGap are the two rows of the recipe's `thresholds` view.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumSpeed',
    label: 'thresholds (maximum speed)',
    min: 0.2,
    max: 3,
    step: 0.1,
    default: 0.8
  },
  {id: 'maximumGap', label: 'thresholds (maximum gap)', min: 1, max: 10, step: 0.5, default: 4},
  {id: 'glitchRate', label: 'Data: glitch rate', min: 0, max: 0.2, step: 0.01, default: 0.05}
];
