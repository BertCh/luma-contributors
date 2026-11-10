// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'classCount', label: 'classCount', min: 2, max: 6, step: 1, default: 4},
  {id: 'grain', label: 'Data: sampling grain', min: 1, max: 8, step: 1, default: 1}
];
