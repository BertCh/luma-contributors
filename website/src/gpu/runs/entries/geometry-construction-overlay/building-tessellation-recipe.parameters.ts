// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'sampleSpacing', label: 'sampleSpacing', min: 0.06, max: 0.2, step: 0.01, default: 0.09},
  {id: 'fill', label: 'Data: footprint scale', min: 0.45, max: 1, step: 0.05, default: 1}
];
