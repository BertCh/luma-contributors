// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'severity', label: 'Data: clearing severity', min: 0, max: 1, step: 0.05, default: 0.8},
  {id: 'noise', label: 'Data: year-to-year noise', min: 0, max: 0.8, step: 0.05, default: 0.4}
];
