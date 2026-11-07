// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'sweep', label: 'Data: sweep', min: 0, max: 1, step: 0.02, default: 0.3},
  {id: 'tilt', label: 'Data: tilt', min: 0, max: 1, step: 0.05, default: 0.4}
];
