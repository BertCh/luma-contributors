// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The rates come from these data sliders.
export const parameters: GpuSceneParameter[] = [
  {id: 'riskContrast', label: 'Data: age-share contrast', min: 0, max: 1, step: 0.05, default: 1},
  {
    id: 'populationScale',
    label: 'Data: populationScale',
    min: 0.01,
    max: 5,
    step: 0.01,
    default: 0.02
  }
];
