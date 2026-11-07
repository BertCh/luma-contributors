// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'radius', label: 'radius', min: 0, max: 6, step: 1, default: 2, unit: 'cells'},
  {id: 'quantile', label: 'quantile', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'noise', label: 'Data: speckle', min: 0, max: 0.3, step: 0.01, default: 0.12}
];
