// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'soilAdjustment', label: 'soilAdjustment', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'epsilon', label: 'epsilon', min: 0, max: 0.8, step: 0.04, default: 0}
];
