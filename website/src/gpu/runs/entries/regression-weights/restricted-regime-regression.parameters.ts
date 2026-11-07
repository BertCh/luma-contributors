// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';
import type {RunLegendItem} from '../../legend';

// The three booleans are GPURestrictedRegimeRegression's `common` prop: a common coefficient is
// shared by both regimes, otherwise each regime gets its own. Contrast sets the true difference.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'commonIntercept',
    label: 'common[intercept]',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  },
  {
    id: 'commonFirst',
    label: 'common[x1]',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['false', 'true']
  },
  {
    id: 'commonSecond',
    label: 'common[x2]',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['false', 'true']
  },
  {id: 'contrast', label: 'Data: regime contrast', min: 0, max: 1.5, step: 0.1, default: 1}
];

export const legend: RunLegendItem[] = [
  {
    label:
      'upper bars: coefficients in order intercept, x1, x2 (shared = one bar, regime-specific = west, east pair)',
    color: '#8a6bd1'
  },
  {label: 'lower bar: Chow F statistic', color: '#f2994d'}
];
