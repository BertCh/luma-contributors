// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import events from '../../../../data/recipes/turkiye-quakes.json';

export const earthquakeEvents = events;

/** Map the fixed 34–44 E, 35–41 N window into the run's planar drawing extent. */
export function getEventPosition(event: number[], width: number, height: number): [number, number] {
  return [((event[0] - 34) / 10) * width, ((event[1] - 35) / 6) * height];
}

/** The observed events as packed (x, y) positions in a `width` by `height` drawing extent, for a context layer. */
export function getEventDots(width: number, height: number): Float32Array {
  return Float32Array.from(
    earthquakeEvents.flatMap(event => getEventPosition(event, width, height))
  );
}
