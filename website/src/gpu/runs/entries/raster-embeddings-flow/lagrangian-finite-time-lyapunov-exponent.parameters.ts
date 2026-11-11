// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'integrationTime', label: 'integrationTime', min: -2, max: 2, step: 0.05, default: 1.2}
];
