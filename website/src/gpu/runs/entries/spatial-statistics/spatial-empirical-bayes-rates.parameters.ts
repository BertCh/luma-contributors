// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUSpatialEmpiricalBayesRates has no scalar props; the sliders shape the data it smooths.
export const parameters: GpuSceneParameter[] = [
  {id: 'riskContrast', label: 'Data: riskContrast', min: 0, max: 1, step: 0.05, default: 1},
  {id: 'populationScale', label: 'Data: populationScale', min: 0.2, max: 5, step: 0.2, default: 3}
];
