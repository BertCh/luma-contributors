// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The settings of getGPUAffineTransformParameterValues, applied about each polygon's centroid.
export const parameters: GpuSceneParameter[] = [
  {id: 'rotate', label: 'rotate', min: -90, max: 90, step: 5, default: 30, unit: '°'},
  {id: 'scale', label: 'scale', min: 0.4, max: 1.4, step: 0.05, default: 0.8},
  {id: 'skew', label: 'skew', min: -35, max: 35, step: 5, default: 0, unit: '°'}
];
