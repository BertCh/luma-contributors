// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import commutes from './chicago-commutes.json';
import {fitLocalMap} from './fit-local-map';

export const COMMUTE_FLOWS = commutes.flows;
export const TRACT_LOCATIONS = commutes.tracts;

export function getCommutePositions(world: {width: number; height: number}): Float32Array {
  const project = fitLocalMap(TRACT_LOCATIONS, world);
  return Float32Array.from(TRACT_LOCATIONS.flatMap(([x, y]) => project(x, y)));
}
