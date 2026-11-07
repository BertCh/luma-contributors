// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'maximumIterations', label: 'maximumIterations', min: 0, max: 200, step: 5, default: 100},
  {id: 'areaScale', label: 'areaScale', min: 0.3, max: 1.6, step: 0.05, default: 1},
  {id: 'attraction', label: 'attraction', min: 0, max: 0.2, step: 0.01, default: 0.03}
];
