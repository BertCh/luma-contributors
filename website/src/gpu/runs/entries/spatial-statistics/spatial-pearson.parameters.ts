// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The statistic has no tunable props; the sliders shape the two variables it correlates.
export const parameters: GpuSceneParameter[] = [
  {id: 'noise', label: 'Data: noise', min: 0, max: 1, step: 0.05, default: 0.25},
  {id: 'radius', label: 'radius', min: 0.04, max: 0.14, step: 0.005, default: 0.08}
];
