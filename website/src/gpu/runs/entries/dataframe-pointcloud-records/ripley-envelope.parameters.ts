// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPURipleyEnvelope's compile-time `simulationCount` and `rank`, and the input clustering.
export const parameters: GpuSceneParameter[] = [
  {id: 'simulationCount', label: 'simulationCount', min: 19, max: 99, step: 20, default: 39},
  {id: 'rank', label: 'rank', min: 1, max: 4, step: 1, default: 1},
  {id: 'clustering', label: 'Data: clustering', min: 0, max: 1, step: 0.05, default: 0.35}
];
