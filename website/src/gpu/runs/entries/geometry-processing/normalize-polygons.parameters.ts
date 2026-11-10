// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUNormalizePolygons has no prop; the slider moves where each input ring starts.
export const parameters: GpuSceneParameter[] = [
  {id: 'start', label: 'input ring start', min: 0, max: 7, step: 1, default: 3}
];
