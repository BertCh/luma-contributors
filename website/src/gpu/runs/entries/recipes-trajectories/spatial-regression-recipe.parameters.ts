// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// bandwidth scales the fixed bandwidth ladder of the geographically weighted fits (a per-frame
// parameter buffer); the other two sliders shape the response.
export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 0.15, max: 0.6, step: 0.05, default: 0.3},
  {id: 'dependence', label: 'Data: errorDependence', min: 0, max: 0.9, step: 0.05, default: 0.6},
  {id: 'variation', label: 'Data: slopeVariation', min: 0, max: 2, step: 0.1, default: 1}
];
