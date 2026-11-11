// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'gapWidth', label: 'gapWidth', min: 0, max: 0.1, step: 0.002, default: 0.03},
  {id: 'gap', label: 'Data: gap', min: -0.03, max: 0.05, step: 0.002, default: 0.012}
];
