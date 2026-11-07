// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to getGPURasterReclassifyParameterValues({breakCount}) and the break table.
export const parameters: GpuSceneParameter[] = [
  {id: 'breakCount', label: 'breakCount', min: 1, max: 7, step: 1, default: 4},
  {id: 'firstBreak', label: 'breaks[0]', min: 0.05, max: 0.5, step: 0.01, default: 0.2}
];
