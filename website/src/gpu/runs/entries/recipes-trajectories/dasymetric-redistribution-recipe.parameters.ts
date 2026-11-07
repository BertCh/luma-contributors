// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// denominator is a compile-time prop of addDasymetricRedistributionRecipe; the urban core
// strength reshapes the ancillary raster that weights the redistribution.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'denominator',
    label: 'denominator',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['zone', 'overlap']
  },
  {id: 'core', label: 'Data: ancillaryContrast', min: 0, max: 1, step: 0.05, default: 0.8},
  {id: 'totals', label: 'Data: sourceContrast', min: 0, max: 2, step: 0.1, default: 1}
];
