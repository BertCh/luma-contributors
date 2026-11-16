// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

type Extent = [number, number, number, number];
type Position = readonly number[];
type LongitudeInterval = readonly [number, number];

/** Longitude on [0, 360); source coordinates themselves are never rewritten. */
export function normalizeAnalysisLongitude(longitude: number): number {
  const remainder = longitude % 360;
  const wrapped = remainder < 0 ? remainder + 360 : remainder + 0;
  // Adding 360 to a tiny negative remainder can round to 360 itself.
  return wrapped === 360 ? 0 : wrapped;
}

/**
 * Exact covering extent of points and literal geometry edges, [west, south, east, north].
 * Edges follow their written longitude intervals, including broad uncut GeoJSON edges. Cut
 * parts remain separate. Pole longitudes remain source bounds, as in the spacing statistics;
 * canonicalising physical locations for distance calculations is a separate operation.
 *
 * The covering arc excludes the largest uncovered gap on the circle. Equal gaps prefer the
 * first gap in ascending longitude order, considering the gap across 0/360 last. Plain source
 * bounds win unless the arc is shorter by more than 1e-9 degrees. Thus 150..250 keeps its source
 * range, 350..10 becomes -10..10, and 178..-178 becomes west > east across the antimeridian.
 *
 * Stores O(vertices + edges) intervals and sorts them in O(n log n) when read. Unlike bucket
 * approximations this resolves gaps of any width; there is no retained angular approximation.
 */
export class AnalysisGeographicExtentAccumulator {
  private minimumX = Infinity;
  private minimumY = Infinity;
  private maximumX = -Infinity;
  private maximumY = -Infinity;
  private intervals: LongitudeInterval[] = [];

  add(longitude: number, latitude: number): void {
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return;
    this.minimumX = Math.min(this.minimumX, longitude);
    this.minimumY = Math.min(this.minimumY, latitude);
    this.maximumX = Math.max(this.maximumX, longitude);
    this.maximumY = Math.max(this.maximumY, latitude);
    const wrapped = normalizeAnalysisLongitude(longitude);
    this.intervals.push([wrapped, wrapped]);
  }

  /** Connect adjacent finite positions, optionally including the closing edge of a ring. */
  addPath(positions: readonly Position[], close: boolean): void {
    let previous: Position | undefined;
    for (const position of positions) {
      this.add(position[0], position[1]);
      if (previous) this.addEdge(previous, position);
      previous = position;
    }
    if (close && previous && positions.length > 0) this.addEdge(previous, positions[0]);
  }

  private addEdge(first: Position, second: Position): void {
    if (![first[0], first[1], second[0], second[1]].every(Number.isFinite)) return;
    const width = Math.abs(second[0] - first[0]);
    if (width >= 360) {
      this.intervals.push([0, 360]);
      return;
    }
    const start = normalizeAnalysisLongitude(Math.min(first[0], second[0]));
    const end = start + width;
    this.intervals.push([start, Math.min(end, 360)]);
    if (end > 360) this.intervals.push([0, end - 360]);
  }

  /** Plain minimum and maximum in source coordinates; absent without a finite position. */
  getPlain(): Extent | undefined {
    return this.minimumX === Infinity
      ? undefined
      : [this.minimumX, this.minimumY, this.maximumX, this.maximumY];
  }

  /** Geographic covering arc, preserving plain bounds when equally short. */
  getCovering(): Extent | undefined {
    const plain = this.getPlain();
    if (!plain) return undefined;
    this.intervals.sort((first, second) => first[0] - second[0] || first[1] - second[1]);
    const firstMinimum = this.intervals[0][0];
    let previousMaximum = this.intervals[0][1];
    let largestGap = -1;
    let westValue = firstMinimum;
    let eastValue = previousMaximum;
    for (let index = 1; index < this.intervals.length; index++) {
      const [minimum, maximum] = this.intervals[index];
      if (minimum - previousMaximum > largestGap) {
        largestGap = minimum - previousMaximum;
        westValue = minimum;
        eastValue = previousMaximum;
      }
      previousMaximum = Math.max(previousMaximum, maximum);
    }
    // Ties retain the first gap in ascending longitude order.
    if (firstMinimum + 360 - previousMaximum > largestGap) {
      largestGap = firstMinimum + 360 - previousMaximum;
      westValue = firstMinimum;
      eastValue = previousMaximum;
    }
    if (largestGap <= 0 || !(360 - largestGap < plain[2] - plain[0] - 1e-9)) return plain;
    const toSigned = (degrees: number) => (degrees > 180 ? degrees - 360 : degrees);
    return [toSigned(westValue), plain[1], toSigned(eastValue), plain[3]];
  }
}
