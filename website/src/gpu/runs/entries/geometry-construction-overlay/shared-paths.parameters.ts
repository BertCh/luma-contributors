// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'overlap', label: 'overlap', min: 1, max: 9, step: 1, default: 5, unit: 'vertices'},
  {id: 'detour', label: 'detour', min: 0, max: 0.1, step: 0.005, default: 0.05}
];
