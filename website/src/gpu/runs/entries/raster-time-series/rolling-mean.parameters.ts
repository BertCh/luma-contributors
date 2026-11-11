// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'windowLength', label: 'windowLength', min: 2, max: 16, step: 2, default: 6},
  {id: 'gapFraction', label: 'Data: gap fraction', min: 0, max: 0.12, step: 0.01, default: 0.03}
];
