// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// trackEventCapacity is a compile-time prop of GPUZoneEvents; the zone size only rewrites the
// zone vertices (per-frame input).
export const parameters: GpuSceneParameter[] = [
  {id: 'trackEventCapacity', label: 'trackEventCapacity', min: 1, max: 6, step: 1, default: 6},
  {id: 'zoneSize', label: 'Data: zoneSize', min: 0.5, max: 1.6, step: 0.05, default: 1}
];
