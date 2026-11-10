// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// trackEventCapacity is a compile-time prop of addFleetDwellZoneEventsRecipe; the zone size only
// moves the zone polygons.
export const parameters: GpuSceneParameter[] = [
  {id: 'trackEventCapacity', label: 'trackEventCapacity', min: 1, max: 6, step: 1, default: 6},
  {id: 'zoneSize', label: 'zoneSize', min: 0.5, max: 1.6, step: 0.05, default: 1}
];
