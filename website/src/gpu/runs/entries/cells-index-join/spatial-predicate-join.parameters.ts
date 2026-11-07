// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `predicate` is GPUSpatialPredicateJoin's compile-time prop; `distance` is its per-frame view and
// only matters for 'dwithin'.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'predicate',
    label: 'predicate',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['dwithin', 'within']
  },
  {id: 'distance', label: 'distance', min: 0.01, max: 0.2, step: 0.005, default: 0.06}
];
