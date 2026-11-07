// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'model', label: 'model', min: 0, max: 1, step: 1, default: 0, options: ['tobler', 'grass']},
  {id: 'frictionWeight', label: 'frictionWeight', min: 0, max: 4, step: 0.25, default: 1},
  {id: 'sourceX', label: 'Data: source x', min: 0.1, max: 0.9, step: 0.02, default: 0.3},
  {id: 'maximumIterations', label: 'maximumIterations', min: 20, max: 300, step: 10, default: 220}
];
