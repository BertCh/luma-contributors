// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'threshold', label: 'Data: threshold', min: 0.3, max: 0.6, step: 0.01, default: 0.4}
];
