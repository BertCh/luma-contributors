// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `maximumPathLength` is a GPUCellGridPath prop (compile time): longer paths are cut off (`counts` still reports their true length). `spread`
// is the radius, in degrees, of the destination fan around the pointer; it is written per frame.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumPathLength', label: 'maximumPathLength', min: 3, max: 24, step: 1, default: 20},
  {id: 'spread', label: 'Data: spread', min: 0.1, max: 0.6, step: 0.01, default: 0.35}
];
