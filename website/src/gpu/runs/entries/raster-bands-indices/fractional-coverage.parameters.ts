// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'gridWidth', label: 'grid.width', min: 6, max: 80, step: 2, default: 14, unit: 'cells'}
];
