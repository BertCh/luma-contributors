// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'lag', label: 'lag', min: -4, max: 8, step: 2, default: 4, unit: 'slices'},
  {id: 'minimumPairCount', label: 'minimumPairCount', min: 3, max: 60, step: 1, default: 3},
  {id: 'gapFraction', label: 'Data: gap fraction', min: 0, max: 0.4, step: 0.02, default: 0.1}
];
