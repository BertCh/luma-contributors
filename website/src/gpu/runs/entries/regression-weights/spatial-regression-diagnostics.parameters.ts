// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// restandardizeRows is a GPUSpatialRegressionDiagnostics prop; the others set the dependence in the data.
export const parameters: GpuSceneParameter[] = [
  {id: 'lagRho', label: 'Data: spatial lag rho', min: 0, max: 0.8, step: 0.05, default: 0.5},
  {
    id: 'errorLambda',
    label: 'Data: spatial error lambda',
    min: 0,
    max: 0.8,
    step: 0.05,
    default: 0
  },
  {
    id: 'restandardizeRows',
    label: 'restandardizeRows',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  }
];
