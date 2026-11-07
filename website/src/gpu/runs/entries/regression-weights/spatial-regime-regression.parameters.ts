// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// regimeCount is a GPUSpatialRegimeRegression prop (vertical bands of tracts); contrast scales how
// much the true coefficients differ between regimes.
export const parameters: GpuSceneParameter[] = [
  {id: 'regimeCount', label: 'regimeCount', min: 1, max: 4, step: 1, default: 3},
  {id: 'contrast', label: 'Data: regime contrast', min: 0, max: 1.5, step: 0.1, default: 1},
  {id: 'noise', label: 'Data: noise', min: 0.1, max: 1.2, step: 0.05, default: 0.4}
];
