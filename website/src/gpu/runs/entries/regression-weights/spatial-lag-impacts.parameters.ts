// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumPower is GPUSpatialLagImpacts' series length; strength is the true spatial lag rho.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumPower', label: 'maximumPower', min: 2, max: 40, step: 1, default: 24},
  {id: 'strength', label: 'Data: spatial lag rho', min: 0, max: 0.8, step: 0.05, default: 0.5}
];
