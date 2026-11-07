// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'haze', label: 'Data: haze', min: 0, max: 0.3, step: 0.01, default: 0.05}
];
