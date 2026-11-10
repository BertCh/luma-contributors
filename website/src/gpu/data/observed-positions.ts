// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fitLocalMap} from './fit-local-map';

/** Local equirectangular projection followed by a uniform fit into the map panel. */
export function projectObservedPositions(
  coordinates: number[][],
  world: {width: number; height: number}
): Float32Array {
  const meanLatitude =
    coordinates.reduce((sum, [, latitude]) => sum + latitude, 0) / coordinates.length;
  const longitudeScale = Math.cos((meanLatitude * Math.PI) / 180);
  const local = coordinates.map(([longitude, latitude]) => [longitude * longitudeScale, latitude]);
  const project = fitLocalMap(local, world);
  return Float32Array.from(local.flatMap(([east, north]) => project(east, north)));
}
