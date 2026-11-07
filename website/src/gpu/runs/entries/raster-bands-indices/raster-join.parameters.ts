// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'gridWidth', label: 'grid.width', min: 12, max: 120, step: 4, default: 40, unit: 'cells'}
];
