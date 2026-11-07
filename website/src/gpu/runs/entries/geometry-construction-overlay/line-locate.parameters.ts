// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'measureOffset', label: 'measureOffset', min: 0, max: 0.6, step: 0.01, default: 0.1},
  {id: 'measureScale', label: 'measureScale', min: 0.4, max: 2, step: 0.05, default: 1}
];
