// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumPower and traceMode are GPUSpatialErrorMaximumLikelihood props; strength is the true lambda.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumPower', label: 'maximumPower', min: 4, max: 64, step: 1, default: 24},
  {id: 'strength', label: 'Data: spatial error lambda', min: 0, max: 0.8, step: 0.05, default: 0.6}
];
