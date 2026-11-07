// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// addExcessRiskRecipe takes events and populations only; both sliders shape the input counts.
export const parameters: GpuSceneParameter[] = [
  {id: 'contrast', label: 'Data: riskContrast', min: 0, max: 2, step: 0.1, default: 1},
  {id: 'smallShare', label: 'Data: smallTractShare', min: 0, max: 0.8, step: 0.05, default: 0.4}
];
