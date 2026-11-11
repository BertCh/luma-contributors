// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'sliceCount', label: 'sliceCount', min: 3, max: 8, step: 1, default: 8},
  {id: 'haze', label: 'Data: undetected haze', min: 0, max: 1, step: 0.05, default: 1}
];
