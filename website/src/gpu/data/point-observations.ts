// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import rainfall from './helene-rain.json';
import seismicity from './japan-earthquakes.json';

type World = {width: number; height: number};
type Random = () => number;
type Observation = {longitude: number; latitude: number; value: number; secondary: number};

function decodeObservations(snapshot: typeof seismicity): Observation[] {
  const bytes = Uint8Array.from(atob(snapshot.data), character => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const [west, south, east, north] = snapshot.bounds;
  return Array.from({length: bytes.length / 8}, (_, row) => {
    const offset = row * 8;
    return {
      longitude: west + ((east - west) * view.getUint16(offset, true)) / 10000,
      latitude: south + ((north - south) * view.getUint16(offset + 2, true)) / 10000,
      value: view.getUint16(offset + 4, true),
      secondary: view.getUint16(offset + 6, true)
    };
  });
}

// Uniform sampling without replacement. Sampling more rows than available starts a new pass.
function sampleRows(random: Random, rows: Observation[], count: number): Observation[] {
  const shuffled = [...rows];
  return Array.from({length: count}, (_, index) => {
    const start = index % shuffled.length;
    const selected = start + Math.floor(random() * (shuffled.length - start));
    [shuffled[start], shuffled[selected]] = [shuffled[selected], shuffled[start]];
    return shuffled[start];
  });
}

// North-up local equirectangular coordinates, letterboxed with a small perimeter.
export function projectRows(
  rows: readonly Pick<Observation, 'longitude' | 'latitude'>[],
  bounds: number[],
  world: World
): Float32Array {
  const [west, south, east, north] = bounds;
  const longitudeScale = Math.cos(((north + south) * Math.PI) / 360);
  const width = (east - west) * longitudeScale;
  const height = north - south;
  const scale = 0.96 * Math.min(world.width / width, world.height / height);
  const positions = new Float32Array(rows.length * 2);
  rows.forEach(({longitude, latitude}, row) => {
    positions[row * 2] =
      (world.width - width * scale) / 2 + (longitude - west) * scale * longitudeScale;
    positions[row * 2 + 1] = (world.height - height * scale) / 2 + (latitude - south) * scale;
  });
  return positions;
}

const CAROLINAS = [-85.3, 32, -78, 36.7];
let stations: Observation[] | undefined;
let carolinaStations: Observation[] | undefined;

/** Daily measured precipitation in millimeters. The Carolina window includes both observation days. */
export function getRainStations(
  random: Random,
  world: World,
  count: number,
  focusCarolinas = false
) {
  stations ??= decodeObservations(rainfall);
  if (focusCarolinas && !carolinaStations) {
    const [west, south, east, north] = CAROLINAS;
    carolinaStations = stations.filter(
      row =>
        row.longitude >= west &&
        row.longitude <= east &&
        row.latitude >= south &&
        row.latitude <= north &&
        row.secondary !== 65535
    );
  }
  const rows = sampleRows(random, focusCarolinas ? carolinaStations! : stations, count);
  return {
    count,
    positions: projectRows(rows, focusCarolinas ? CAROLINAS : rainfall.bounds, world),
    rain: Float32Array.from(rows, row => row.value / 10),
    previousRain: Float32Array.from(rows, row =>
      row.secondary === 65535 ? NaN : row.secondary / 10
    )
  };
}

let earthquakes: Observation[] | undefined;
let sparseEarthquakes: Observation[] | undefined;

function getSparseEvents(rows: Observation[]): Observation[] {
  const cell = (row: Observation) =>
    Math.floor(row.longitude * 2) * 360 + Math.floor(row.latitude * 2);
  const counts = new Map<number, number>();
  for (const row of rows) counts.set(cell(row), (counts.get(cell(row)) ?? 0) + 1);
  return rows.filter(row => (counts.get(cell(row)) ?? 0) <= 8);
}

/** Magnitude and epicenter from the ComCat extract in and around Japan. */
export function getEarthquakes(random: Random, world: World, count: number, scatterShare = 0) {
  earthquakes ??= decodeObservations(seismicity);
  const scattered = new Uint8Array(count);
  let rows: Observation[];
  if (scatterShare > 0) {
    sparseEarthquakes ??= getSparseEvents(earthquakes);
    let backgroundCount = 0;
    for (let row = 0; row < count; row++) {
      if (random() < scatterShare && backgroundCount < sparseEarthquakes.length) {
        scattered[row] = 1;
        backgroundCount++;
      }
    }
    const background = sampleRows(random, sparseEarthquakes, backgroundCount);
    const selected = new Set(background);
    const remaining = sampleRows(
      random,
      earthquakes.filter(row => !selected.has(row)),
      count - backgroundCount
    );
    let backgroundIndex = 0;
    let remainingIndex = 0;
    rows = Array.from(scattered, isScattered =>
      isScattered ? background[backgroundIndex++] : remaining[remainingIndex++]
    );
  } else {
    rows = sampleRows(random, earthquakes, count);
  }
  return {
    count,
    positions: projectRows(rows, seismicity.bounds, world),
    magnitude: Float32Array.from(rows, row => row.value / 100),
    scattered
  };
}

// The offshore Honshu study window keeps cell boundaries visible at H3 levels 3–6.
// All analyzed positions and decoded cell boundaries are in geographic degrees.
export const HONSHU_WINDOW = [140, 36.1, 144.4, 38.6] as const;
let honshuEvents: Observation[] | undefined;

function getHonshuCatalog(): Observation[] {
  earthquakes ??= decodeObservations(seismicity);
  const [west, south, east, north] = HONSHU_WINDOW;
  honshuEvents ??= earthquakes.filter(
    row =>
      row.longitude >= west &&
      row.longitude <= east &&
      row.latitude >= south &&
      row.latitude <= north
  );
  return honshuEvents;
}

/** Select actual epicenters close to the requested geographic sampling sites. */
export function getHonshuEventSites(
  targets: readonly (readonly [number, number])[]
): [number, number][] {
  const catalog = getHonshuCatalog();
  const midpointLatitude = (HONSHU_WINDOW[1] + HONSHU_WINDOW[3]) / 2;
  const longitudeScale = Math.cos((midpointLatitude * Math.PI) / 180);
  return targets.map(([longitude, latitude]) => {
    let nearest = catalog[0];
    let minimum = Infinity;
    for (const row of catalog) {
      const distance =
        ((row.longitude - longitude) * longitudeScale) ** 2 + (row.latitude - latitude) ** 2;
      if (distance < minimum) {
        nearest = row;
        minimum = distance;
      }
    }
    return [nearest.longitude, nearest.latitude];
  });
}

export function getLargestHonshuEvent(): [number, number] {
  const largest = getHonshuCatalog().reduce((best, row) => (row.value > best.value ? row : best));
  return [largest.longitude, largest.latitude];
}

export function getHonshuEvents(random: Random, maximumCount: number) {
  const catalog = getHonshuCatalog();
  const rows = sampleRows(random, catalog, Math.min(maximumCount, catalog.length));
  const positions = new Float32Array(rows.length * 2);
  for (let row = 0; row < rows.length; row++) {
    positions[row * 2] = rows[row].longitude;
    positions[row * 2 + 1] = rows[row].latitude;
  }
  return {count: rows.length, positions, values: Float32Array.from(rows, row => row.value / 100)};
}

/** Capacity for all directed neighbor pairs at the largest live radius. Reserve one self
 * slot per station and allow for float32 distance rounding at the radius cutoff. */
export function getRainNeighborCapacity(positions: Float32Array, radius: number): number {
  const squaredLimit = (radius + 1e-6) ** 2;
  let count = positions.length / 2;
  for (let first = 0; first < positions.length; first += 2) {
    for (let second = first + 2; second < positions.length; second += 2) {
      const east = positions[first] - positions[second];
      const north = positions[first + 1] - positions[second + 1];
      if (east * east + north * north <= squaredLimit) count += 2;
    }
  }
  return count;
}
