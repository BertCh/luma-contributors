// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The window end picks which half-hour buckets are the cases (an input); maximumWindowLength is a
// per-frame value of getGPUExpectationBasedScanParameterValues.

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'windowEnd', label: 'cases end', min: 10, max: 18, step: 0.5, default: 12, unit: 'h'},
  {id: 'maximumWindowLength', label: 'maximumWindowLength', min: 1, max: 8, step: 1, default: 4}
];
