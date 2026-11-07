// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// instrumentOrder is GPUSpatialTwoStageLeastSquares' prop; strength is the true spatial lag rho.
export const parameters: GpuSceneParameter[] = [
  {id: 'instrumentOrder', label: 'instrumentOrder', min: 1, max: 2, step: 1, default: 2},
  {id: 'strength', label: 'Data: spatial lag rho', min: 0, max: 0.8, step: 0.05, default: 0.5},
  {id: 'noise', label: 'Data: noise', min: 0.1, max: 1.2, step: 0.05, default: 0.4}
];

export const legend: RunLegendItem[] = [
  {label: 'left: structural residual per tract, blue negative to red positive', color: '#c4242b'},
  {label: 'bars, left to right: intercept, x1, x2, spatial lag rho (estimates)', color: '#8a6bd1'},
  {label: 'dots: the true coefficients', color: '#1c2230'}
];
