// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'standardDeviations', label: 'standardDeviations', min: 0.5, max: 3, step: 0.1, default: 1},
  {id: 'groupCount', label: 'groupCount', min: 1, max: 4, step: 1, default: 4}
];
