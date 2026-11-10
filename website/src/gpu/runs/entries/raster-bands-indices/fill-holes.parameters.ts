// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'maximumIterations', label: 'maximumIterations', min: 4, max: 160, step: 4, default: 120},
  {id: 'threshold', label: 'Data: threshold', min: 0.3, max: 0.6, step: 0.01, default: 0.4},
  {id: 'pits', label: 'Data: pits', min: 0, max: 0.2, step: 0.01, default: 0.06}
];
