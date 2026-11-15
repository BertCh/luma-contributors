// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Longitude, latitude and altitude. */
type Coordinates = [number, number, number];
type Path = Coordinates[];
type Polygon = Path[];

/** Deck's GlobeView picks 5 degree cut cells; Kepler passes the same resolution. */
export const GLOBE_GEOMETRY_RESOLUTION_DEGREES = 5;

/** Longitudes beyond this (in degrees) are not geographic and would explode grid cutting. */
const MAXIMUM_GLOBE_LONGITUDE_DEGREES = 540;
/** Upper bound on cut cells per polygon and subdivision steps per edge. */
const MAXIMUM_GLOBE_WORK_ITEMS = 65536;

export type KeplerGlobeGeometryOptions = {
  /** Maximum cut cell and edge length in degrees. Defaults to 5, matching Deck's GlobeView. */
  resolutionDegrees?: number;
  /** Decoded MVT coordinates already belong to a tile interval, including its buffered edges. */
  unwrapLongitude?: boolean;
};

/**
 * Shifts longitudes by whole turns so consecutive vertices differ by at most 180 degrees. A path
 * across the antimeridian then continues past 180 instead of jumping across the globe. Globe
 * projection is periodic in longitude, so values beyond +/-180 project correctly.
 */
export function unwrapGlobePath(path: Path): Path {
  if (path.length < 2) return path;
  const result: Path = [path[0]];
  let offset = 0;
  let previousLongitude = path[0][0];
  for (let index = 1; index < path.length; index++) {
    let longitude = path[index][0] + offset;
    while (longitude - previousLongitude > 180) {
      offset -= 360;
      longitude -= 360;
    }
    while (longitude - previousLongitude < -180) {
      offset += 360;
      longitude += 360;
    }
    previousLongitude = longitude;
    result.push([longitude, path[index][1], path[index][2]]);
  }
  return result;
}

/**
 * Unwraps a closed ring. A ring that no longer closes (it encircles a pole) keeps its source
 * longitudes, because an unwrapped copy would leave a gap across the seam.
 */
function unwrapGlobeRing(ring: Path): Path {
  const unwrapped = unwrapGlobePath(ring);
  const first = unwrapped[0];
  const last = unwrapped[unwrapped.length - 1];
  return first && last && Math.abs(first[0] - last[0]) > 1e-9 ? ring : unwrapped;
}

/** Inserts evenly spaced vertices so no edge spans more than `resolutionDegrees` in either axis. */
export function subdivideGlobePath(
  path: Path,
  resolutionDegrees = GLOBE_GEOMETRY_RESOLUTION_DEGREES
): Path {
  if (path.length < 2) return path;
  const result: Path = [path[0]];
  for (let index = 1; index < path.length; index++) {
    const start = path[index - 1];
    const end = path[index];
    const count = Math.ceil(
      Math.max(Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1])) / resolutionDegrees
    );
    if (count > MAXIMUM_GLOBE_WORK_ITEMS)
      throw new Error('Globe edge subdivision exceeds the supported limit');
    for (let step = 1; step < count; step++) {
      const ratio = step / count;
      result.push([
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
        start[2] + (end[2] - start[2]) * ratio
      ]);
    }
    result.push(end);
  }
  return result;
}

/**
 * One Sutherland-Hodgman pass. Known limitation, shared with Deck's grid cutting: a concave ring
 * that meets a slab in several disjoint regions yields one ring joined by zero-width edges along
 * the clip line. Tessellation tolerates these; holes use the same method.
 */
/** Clips against `coordinate >= limit` (keep upper) or `<= limit`. */
function clipRing(ring: Path, axis: 0 | 1, limit: number, keepUpper: boolean): Path {
  const inside = (point: Coordinates) => (keepUpper ? point[axis] >= limit : point[axis] <= limit);
  const result: Path = [];
  for (let index = 0; index < ring.length; index++) {
    const current = ring[index];
    const previous = ring[(index + ring.length - 1) % ring.length];
    const currentInside = inside(current);
    if (currentInside !== inside(previous)) {
      const ratio = (limit - previous[axis]) / (current[axis] - previous[axis]);
      const crossing: Coordinates = [
        previous[0] + (current[0] - previous[0]) * ratio,
        previous[1] + (current[1] - previous[1]) * ratio,
        previous[2] + (current[2] - previous[2]) * ratio
      ];
      crossing[axis] = limit;
      result.push(crossing);
    }
    if (currentInside) result.push(current);
  }
  return result;
}

/** Clips an open (unduplicated) ring to one axis-aligned interval. */
function clipRingToInterval(ring: Path, axis: 0 | 1, minimum: number, maximum: number): Path {
  const lower = clipRing(ring, axis, minimum, true);
  return lower.length ? clipRing(lower, axis, maximum, false) : lower;
}

function openRing(ring: Path): Path {
  const first = ring[0];
  const last = ring[ring.length - 1];
  return first && last && first[0] === last[0] && first[1] === last[1] && first[2] === last[2]
    ? ring.slice(0, -1)
    : ring;
}

function closeRing(ring: Path): Path {
  return [...ring, [...ring[0]]];
}

/** Unwraps every ring and shifts holes by whole turns to sit beside their outer ring. */
export function alignGlobePolygon(polygon: Polygon): Polygon {
  if (!polygon.length) return polygon;
  const outer = unwrapGlobeRing(polygon[0]);
  const outerCenter = outer.reduce((sum, point) => sum + point[0], 0) / outer.length;
  const holes = polygon.slice(1).map(hole => {
    const unwrapped = unwrapGlobeRing(hole);
    const center = unwrapped.reduce((sum, point) => sum + point[0], 0) / unwrapped.length;
    const shift = Math.round((outerCenter - center) / 360) * 360;
    return shift
      ? unwrapped.map((point): Coordinates => [point[0] + shift, point[1], point[2]])
      : unwrapped;
  });
  return [outer, ...holes];
}

/**
 * Cuts an aligned polygon on a longitude/latitude grid so each tessellated piece stays within one
 * cell and follows the sphere, like Deck's `cutPolygonByGrid` on a globe. Holes are clipped per
 * cell. A polygon that already fits one cell is returned unchanged.
 */
export function cutGlobePolygon(
  polygon: Polygon,
  resolutionDegrees = GLOBE_GEOMETRY_RESOLUTION_DEGREES
): Polygon[] {
  if (!polygon.length) return [];
  const [outer, ...holes] = polygon;
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (const point of outer) {
    bounds[0] = Math.min(bounds[0], point[0]);
    bounds[1] = Math.min(bounds[1], point[1]);
    bounds[2] = Math.max(bounds[2], point[0]);
    bounds[3] = Math.max(bounds[3], point[1]);
  }
  if (bounds[2] - bounds[0] <= resolutionDegrees && bounds[3] - bounds[1] <= resolutionDegrees)
    return [[outer, ...holes]];
  const pieces: Polygon[] = [];
  const outerOpen = openRing(outer);
  const holesOpen = holes.map(openRing);
  const firstColumn = Math.floor(bounds[0] / resolutionDegrees);
  const lastColumn = Math.ceil(bounds[2] / resolutionDegrees);
  const firstRow = Math.floor(bounds[1] / resolutionDegrees);
  const lastRow = Math.ceil(bounds[3] / resolutionDegrees);
  if ((lastColumn - firstColumn) * (lastRow - firstRow) > MAXIMUM_GLOBE_WORK_ITEMS)
    throw new Error('Globe polygon cut grid exceeds the supported limit');
  for (let column = firstColumn; column < lastColumn; column++) {
    const west = column * resolutionDegrees;
    const east = west + resolutionDegrees;
    const outerSlab = clipRingToInterval(outerOpen, 0, west, east);
    if (outerSlab.length < 3) continue;
    const holeSlabs = holesOpen.map(hole => clipRingToInterval(hole, 0, west, east));
    for (let row = firstRow; row < lastRow; row++) {
      const south = row * resolutionDegrees;
      const north = south + resolutionDegrees;
      const cell = clipRingToInterval(outerSlab, 1, south, north);
      if (cell.length < 3) continue;
      const cellHoles = holeSlabs
        .map(hole => clipRingToInterval(hole, 1, south, north))
        .filter(hole => hole.length >= 3)
        .map(closeRing);
      pieces.push([closeRing(cell), ...cellHoles]);
    }
  }
  return pieces;
}

/** Applies the GlobeView preparation to one feature's primitives; points are unchanged. */
export function prepareGlobeGeometryParts(
  parts: {paths: Path[]; polygons: Polygon[]},
  options: KeplerGlobeGeometryOptions = {}
): {paths: Path[]; polygons: Polygon[]; outlines: Path[][]} {
  const resolution = options.resolutionDegrees ?? GLOBE_GEOMETRY_RESOLUTION_DEGREES;
  assertGlobeCoordinateRange(parts);
  const aligned =
    options.unwrapLongitude === false ? parts.polygons : parts.polygons.map(alignGlobePolygon);
  return {
    paths: parts.paths.map(path =>
      subdivideGlobePath(
        options.unwrapLongitude === false ? path : unwrapGlobePath(path),
        resolution
      )
    ),
    polygons: aligned.flatMap(polygon => cutGlobePolygon(polygon, resolution)),
    // Outlines keep the source rings (subdivided), never the grid cut edges.
    outlines: aligned.map(polygon => polygon.map(ring => subdivideGlobePath(ring, resolution)))
  };
}

/** Rejects projected or otherwise non-geographic input before unwrapping and grid cutting. */
function assertGlobeCoordinateRange(parts: {paths: Path[]; polygons: Polygon[]}): void {
  const isGeographic = (vertex: Coordinates): boolean =>
    Math.abs(vertex[0]) <= MAXIMUM_GLOBE_LONGITUDE_DEGREES && Math.abs(vertex[1]) <= 90;
  const isValid =
    parts.paths.every(path => path.every(isGeographic)) &&
    parts.polygons.every(polygon => polygon.every(ring => ring.every(isGeographic)));
  // Globe geometry requires longitude/latitude degrees; projected coordinates are not supported.
  if (!isValid) throw new Error('Globe geometry requires longitude and latitude in degrees');
}
