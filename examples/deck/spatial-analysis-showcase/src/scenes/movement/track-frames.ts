// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGreatCircleDistanceMeters} from '../../cartography/reference-geometry';

/**
 * Great-circle path length in meters of every track (on the sphere of
 * `GPU_EARTH_MEAN_RADIUS_METERS`), summed in float64 on the CPU for labels and rankings.
 */
export function getGreatCircleTrackLengths(
  lngLat: Float32Array,
  offsets: Uint32Array
): Float32Array {
  const lengths = new Float32Array(offsets.length - 1);
  for (let track = 0; track < lengths.length; track++) {
    let length = 0;
    for (let vertex = offsets[track]; vertex < offsets[track + 1] - 1; vertex++) {
      length += getGreatCircleDistanceMeters(
        [lngLat[vertex * 2], lngLat[vertex * 2 + 1]],
        [lngLat[vertex * 2 + 2], lngLat[vertex * 2 + 3]]
      );
    }
    lengths[track] = length;
  }
  return lengths;
}
