// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'ratio', label: 'ratio', min: 0.05, max: 0.5, step: 0.01, default: 0.25},
  {id: 'iterationCount', label: 'iterationCount', min: 1, max: 5, step: 1, default: 2}
];
