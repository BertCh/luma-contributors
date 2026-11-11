// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `epsilon` is the per-frame entropic regularization of GPUSinkhornDivergence, in squared cells.
// `maximumIterations` is compile-time. The pointer also shifts the target mass (see the run).
export const parameters: GpuSceneParameter[] = [
  {id: 'epsilon', label: 'epsilon', min: 4, max: 48, step: 2, default: 12, unit: 'cell²'},
  {id: 'maximumIterations', label: 'maximumIterations', min: 2, max: 60, step: 2, default: 40}
];
