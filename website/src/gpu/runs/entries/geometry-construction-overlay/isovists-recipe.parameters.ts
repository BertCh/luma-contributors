// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.08, max: 0.6, step: 0.02, default: 0.3},
  {id: 'rayCount', label: 'rayCount', min: 12, max: 180, step: 6, default: 120},
  {id: 'fill', label: 'Data: fill', min: 0.45, max: 0.9, step: 0.05, default: 0.7}
];
