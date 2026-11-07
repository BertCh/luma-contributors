// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'rowCount', label: 'Data: rowCount', min: 3, max: 12, step: 1, default: 7, unit: 'owners'},
  {id: 'spread', label: 'Data: spread', min: 0, max: 1, step: 0.05, default: 0.6}
];
