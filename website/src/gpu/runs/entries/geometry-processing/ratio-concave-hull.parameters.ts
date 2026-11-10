// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// ratio is a constructor prop of GPURatioConcaveHull; eventCount selects more or fewer epicenters.
export const parameters: GpuSceneParameter[] = [
  {id: 'ratio', label: 'ratio', min: 0.02, max: 1, step: 0.02, default: 0.2},
  {id: 'eventCount', label: 'Data: event count', min: 120, max: 360, step: 20, default: 240}
];
