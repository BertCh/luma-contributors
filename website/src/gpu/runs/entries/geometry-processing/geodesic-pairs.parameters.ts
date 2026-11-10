// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUGeodesicPairs has no per-frame prop; the slider changes how many cities measure to the target.
export const parameters: GpuSceneParameter[] = [
  {id: 'cityCount', label: 'Data: cities', min: 3, max: 18, step: 1, default: 12}
];
