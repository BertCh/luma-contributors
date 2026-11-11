// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// bandwidth and sensitivity are per-frame; bandwidthFactorLimit is compile-time (it sizes the
// gather of every cell).

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 0.01, max: 0.06, step: 0.0025, default: 0.025},
  {id: 'sensitivity', label: 'sensitivity', min: 0, max: 1, step: 0.05, default: 0.5},
  {id: 'bandwidthFactorLimit', label: 'bandwidthFactorLimit', min: 1, max: 8, step: 1, default: 4}
];
