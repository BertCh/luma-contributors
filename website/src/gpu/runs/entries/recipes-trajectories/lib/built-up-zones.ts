// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getBuiltFraction} from './land-cover';
import {createQuadZones} from './quads';

/** Synthetic reporting zones valued at their center from the measured built-up fraction. */
export function createBuiltUpZones(
  random: () => number,
  world: {width: number; height: number},
  columns = 16,
  rows = 9
) {
  const zones = createQuadZones(random, world, columns, rows);
  const values = Float32Array.from({length: zones.count}, (_, index) =>
    getBuiltFraction(
      zones.centers[index * 2] / world.width,
      zones.centers[index * 2 + 1] / world.height
    )
  );
  return {...zones, values};
}
