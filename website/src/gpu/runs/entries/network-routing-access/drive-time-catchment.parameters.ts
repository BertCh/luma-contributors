// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// breakWidth scales the recipe's per-frame costBreaks (one band per width); facilities sizes its input.
export const parameters: GpuSceneParameter[] = [
  {id: 'breakWidth', label: 'costBreaks (width)', min: 0.1, max: 0.6, step: 0.02, default: 0.25},
  {id: 'facilityCount', label: 'facilities', min: 1, max: 6, step: 1, default: 3}
];
