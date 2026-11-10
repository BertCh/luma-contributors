// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// facilityCount and maximumDistance are per-frame props of addStraightLineCatchmentsRecipe.
export const parameters: GpuSceneParameter[] = [
  {id: 'facilityCount', label: 'facilityCount', min: 2, max: 14, step: 1, default: 8},
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.1, max: 1.5, step: 0.05, default: 0.4}
];
