// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type FlowFieldBounds = readonly [number, number, number, number];

/**
 * Spawn rectangle for a viewport, normalized to the field bounds ([west, south, east, north] in
 * 0..1), padded by a tenth of the viewport. Undefined when the viewport covers the field, misses
 * it, or has no usable bounds: particles then spread over the whole field.
 */
export function getFlowFieldSpawnRegion(
  viewportBounds: readonly [number, number, number, number] | undefined,
  fieldBounds: readonly [number, number, number, number]
): [number, number, number, number] | undefined {
  if (!viewportBounds || !viewportBounds.every(Number.isFinite)) return undefined;
  let [west, south, east, north] = viewportBounds;
  if (!(east > west && north > south) || east - west >= 360) return undefined;
  const padLongitude = (east - west) * 0.1;
  const padLatitude = (north - south) * 0.1;
  west -= padLongitude;
  east += padLongitude;
  south -= padLatitude;
  north += padLatitude;
  const [fieldWest, fieldSouth, fieldEast, fieldNorth] = fieldBounds;
  for (const shift of [0, -360, 360]) {
    const shiftedWest = west + shift;
    const shiftedEast = east + shift;
    if (shiftedEast < fieldWest || shiftedWest > fieldEast) continue;
    const region: [number, number, number, number] = [
      Math.max(0, (shiftedWest - fieldWest) / (fieldEast - fieldWest)),
      Math.max(0, (south - fieldSouth) / (fieldNorth - fieldSouth)),
      Math.min(1, (shiftedEast - fieldWest) / (fieldEast - fieldWest)),
      Math.min(1, (north - fieldSouth) / (fieldNorth - fieldSouth))
    ];
    if (!(region[2] > region[0] && region[3] > region[1])) return undefined;
    return region[0] <= 0 && region[1] <= 0 && region[2] >= 1 && region[3] >= 1
      ? undefined
      : region;
  }
  return undefined;
}

/** Convert the padded spawn rectangle into a field domain accepted by the public simulation. */
export function getFlowFieldViewportBounds(
  viewportBounds: FlowFieldBounds | undefined,
  fieldBounds: FlowFieldBounds
): FlowFieldBounds {
  const region = getFlowFieldSpawnRegion(viewportBounds, fieldBounds);
  if (!region) return fieldBounds;
  return [
    fieldBounds[0] + region[0] * (fieldBounds[2] - fieldBounds[0]),
    fieldBounds[1] + region[1] * (fieldBounds[3] - fieldBounds[1]),
    fieldBounds[0] + region[2] * (fieldBounds[2] - fieldBounds[0]),
    fieldBounds[1] + region[3] * (fieldBounds[3] - fieldBounds[1])
  ];
}

/**
 * Settle for 150 ms before moving the simulation domain. Ignore edge movement below 2% of the
 * active extent so fractional camera jitter never repeatedly reseeds particles. Source/filter
 * bounds changes bypass this viewport-only debounce in the caller.
 */
export class FlowFieldViewportTracker {
  private pending?: FlowFieldBounds;
  private changedAt = 0;

  update(desired: FlowFieldBounds, active: FlowFieldBounds, timestamp: number): boolean {
    const isClose = (first: FlowFieldBounds, second: FlowFieldBounds) =>
      first.every(
        (value, index) =>
          Math.abs(value - second[index]) <= 0.02 * (active[(index % 2) + 2] - active[index % 2])
      );
    if (isClose(desired, active)) {
      this.pending = undefined;
      return false;
    }
    if (!this.pending || !isClose(desired, this.pending)) {
      this.pending = [...desired];
      this.changedAt = timestamp;
      return false;
    }
    if (timestamp - this.changedAt < 150) return false;
    this.pending = undefined;
    return true;
  }
}
