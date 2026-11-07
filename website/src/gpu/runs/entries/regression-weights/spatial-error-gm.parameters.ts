// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// estimator is a GPUSpatialErrorGM prop; strength is the true lambda.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'estimator',
    label: 'estimator',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['homoskedastic', 'kelejian-prucha']
  },
  {id: 'strength', label: 'Data: spatial error lambda', min: 0, max: 0.8, step: 0.05, default: 0.6},
  {id: 'noise', label: 'Data: noise', min: 0.1, max: 1.2, step: 0.05, default: 0.4}
];
