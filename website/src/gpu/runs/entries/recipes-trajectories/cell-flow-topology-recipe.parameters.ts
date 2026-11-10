// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'pairCount', label: 'Data: pairCount', min: 40, max: 400, step: 20, default: 200},
  {id: 'pull', label: 'Data: eastWestPull', min: 0, max: 3, step: 0.1, default: 0}
];
