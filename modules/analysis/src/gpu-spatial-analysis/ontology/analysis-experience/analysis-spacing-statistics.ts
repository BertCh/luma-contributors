// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../../analysis-core/contracts/earth';
import {AnalysisGeographicExtentAccumulator} from './analysis-geographic-extent';
import type {
  GPUAnalysisCoordinateModel,
  GPUAnalysisSpacingStatistics
} from './analysis-experience-types';

/**
 * Neighbour rank `k` of the k-distance curve when the caller gives none. Schubert et al. (2017,
 * "DBSCAN Revisited, Revisited", section 4.2) use `k = 2 * dimension - 1` for the sorted
 * k-distance plot, which is 3 in two dimensions.
 */
export const GPU_ANALYSIS_DEFAULT_NEIGHBOR_RANK = 3;

/**
 * Largest neighbour rank accepted. The search keeps the `k` best candidates of each point in a
 * sorted list that is updated by insertion, so the cost per candidate grows with `k`; 64 is far
 * above any rank used to pick a distance band or a DBSCAN radius.
 */
export const GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK = 64;

/**
 * Number of points the neighbour distances are computed on before a deterministic sample is taken.
 * The grid search is expected O(n). Measured in Node on an Apple laptop, for the whole call:
 * 20,000 uniform points 37 ms, 200,000 uniform points 390 ms, 100,000 points in 50 tight clusters
 * with one far outlier 290 ms (200,000 of them 1.2 s), 100,000 geographic points over the whole
 * globe 320 ms. 100,000 keeps the worst measured case at about a third of a second, well under a
 * second, while a sample of that size makes the neighbour distances only 1.4 times too large for a
 * homogeneous pattern of 200,000 points.
 */
export const GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE = 100_000;

/** The k-distance curve is reduced to this many evenly spaced quantiles, enough to draw and to find a knee. */
export const GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT = 512;

/**
 * Cell counts per axis stay below this value so that a cell index fits a 32-bit integer and the
 * hash of two indices stays well mixed. One axis can therefore resolve a distance down to about
 * a millionth of the extent; closer points share a cell, which costs speed, not correctness.
 */
const MAXIMUM_CELLS_PER_AXIS = 1 << 20;

/**
 * A point whose search has looked up more than `max(1,024, searchedPointCount / 16)` grid cells
 * is compared with every searched point instead. A cell lookup costs about 15 ns and a comparison
 * about 1 ns, so 16 lookups per point of the set is roughly the cost of the full scan; the floor
 * keeps small sets on the grid.
 */
const MINIMUM_RING_LOOKUP_BUDGET = 1024;
const RING_LOOKUPS_PER_POINT_SCAN = 16;

/** Sweeps of Jacobi rotations for the 3 x 3 projection-axis problem; 3 to 5 reach rounding in practice. */
const JACOBI_MAXIMUM_SWEEPS = 30;

/** Interquartile ranges above the upper quartile where the upper fence of the nearest-neighbour distances lies (Tukey's outer fence). */
const UPPER_FENCE_INTERQUARTILE_RANGES = 3;

/** Target average number of points in an occupied cell. Two keeps the first ring of the search short. */
const TARGET_POINTS_PER_CELL = 2;

/** When occupied cells hold more points than this on average, the cells are shrunk and the grid is rebuilt. */
const REFINE_POINTS_PER_CELL = 6;

/** Rebuilds allowed for clustered or collinear data, where the extent-based cell size is too large. */
const MAXIMUM_GRID_REFINEMENTS = 4;

/** More than 64 candidates in one cell warrants a robust rebuild rather than repeated dense scans. */
const MAXIMUM_CELL_POPULATION_BEFORE_ROBUST_GRID = 64;

/** Trim one percent per axis when outliers prevent refinement; this covers up to 100 outliers in 20,000 rows. */
const ROBUST_GRID_TAIL_SHARE = 0.01;

/**
 * A mean unit vector shorter than this has no usable direction (the points cancel out, as two
 * antipodal points do), so the centre falls back to the mean longitude and latitude. The value is
 * far above rounding noise (1e-16) and far below any real spread.
 */
const MINIMUM_UNIT_VECTOR_LENGTH = 1e-9;

/** A small power-of-two table avoids tiny allocations while open addressing stays below half full. */
const MINIMUM_LOCATION_TABLE_SIZE = 16;

export type GPUAnalysisSpacingOptions = {
  /** `geographic`: x is longitude and y is latitude in degrees, distances are great-circle metres. `planar`: Euclidean in coordinate units. */
  coordinateModel: GPUAnalysisCoordinateModel;
  /** Rank `k` of the k-distance curve, an integer from 1 to {@link GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK}. Defaults to {@link GPU_ANALYSIS_DEFAULT_NEIGHBOR_RANK}. */
  neighborRank?: number;
  /** Largest number of points to compute neighbour distances on, at least 2. Defaults to {@link GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE}. */
  maximumSampleSize?: number;
};

/**
 * Computes how a point set is spread: extent, mean centre, standard distance, nearest-neighbour
 * distances, the k-distance curve and the number of repeated locations. The result feeds the
 * default rules in `analysis-default-rules.ts`.
 *
 * Input: `coordinates` holds `x0, y0, x1, y1, ...`. A row with a non-finite coordinate is skipped
 * and not counted in `pointCount`. In `geographic` mode a row whose latitude lies outside
 * [-90, 90] degrees is skipped too, because no great-circle distance exists for it. An odd length
 * throws a `RangeError`.
 *
 * Distances
 * - `planar`: Euclidean distance in coordinate units.
 * - `geographic`: haversine great-circle distance on a sphere of radius
 *   `GPU_EARTH_MEAN_RADIUS_METERS` (6371008.8 m, the WGS84 mean radius used elsewhere in this
 *   package). The error against the WGS84 ellipsoid is up to about 0.3 percent, which is far
 *   below the precision of a default that a user is going to edit.
 *
 * Method: exact nearest and k-th neighbours with a uniform grid hash, expected O(n). The grid cell
 * size is chosen for about two points per occupied cell and is shrunk, up to four times, when
 * clustering or collinearity leaves occupied cells holding more than six points on average. Each
 * point scans its own cell and then square rings of cells outward until the k-th best distance is
 * no larger than the distance to the nearest unvisited ring, so the result is exact, not
 * approximate. If a refined cell still holds more than 64 points, a second grid uses the central
 * 98 percent of each projected axis and retains outliers in boundary cells. Clamping is
 * nonexpansive, so the ring lower bounds and exactness still hold. The cell lookups of one point are bounded (1,024 or searchedPointCount / 16,
 * whichever is larger); past the bound the point is compared with every searched point, which is
 * exact and cheaper than more rings, so no layout makes the rings quadratic. Worst case (all
 * points in a few cells) it is still O(n squared), which the sample limit bounds; points at one
 * location stop the scan early once `k` zero distances are found.
 *
 * Geographic grid: longitude and latitude cannot be gridded directly (cells shrink towards the
 * poles and the longitude wraps). The points are placed on the sphere as 3D vectors and projected
 * orthographically onto a plane, and that plane is gridded. The projection axis is the direction
 * of least variance of the searched points: the eigenvector of the smallest eigenvalue of the
 * 3 x 3 covariance matrix of their positions (cyclic Jacobi rotations). A city sits in a nearly flat
 * patch and is viewed from above; points along a great circle (a meridian, a flight corridor)
 * lie in a plane and are viewed along its normal, so nothing folds onto a segment. (An axis along
 * the mean direction, used before, folds points on a meridian circle onto one line: 4,000
 * random points on longitudes 0 and 180 took 34 s.) An orthographic projection never increases a
 * distance, so the distance between two points in the plane is a lower bound of the true
 * distance for any axis, and a ring that lies further away than the current k-th best distance
 * cannot hold a nearer point. Candidates are compared by their true 3D chord (monotone in the
 * great-circle distance), and the reported distance is the haversine of the chosen pair. Poles
 * and the antimeridian are canonicalized before the search. Measured whole-call times: 16,000 points on a great
 * circle (random or evenly spaced) 33 ms and 17 ms, 64,000 on one meridian 81 ms, 64,000 on two
 * meridians 120 degrees apart 77 ms, 100,000 over the whole globe 283 ms.
 *
 * Antimeridian (geographic): two points at 179.5 and -179.5 degrees longitude on the same
 * latitude are close, and are treated so for neighbour distances, standard distance and median
 * distance from the centre. The mean centre is the mean of the 3D unit vectors converted back to
 * longitude and latitude, so it also lies near 180 degrees; if the vectors cancel out (a
 * norm below 1e-9, for example two antipodal points) the arithmetic mean of longitude and latitude
 * is used. The longitude range of `extent` is taken on the circle: the smallest arc that covers
 * every longitude (360 degrees minus the largest gap between sorted longitudes). When that arc is
 * shorter than the plain range of the source longitudes, `extent[0]` (west) and `extent[2]`
 * (east) are the ends of the arc in [-180, 180] and WEST IS GREATER THAN EAST if the arc crosses
 * the antimeridian (a box centred on 180 reports `[178, south, -178, north]`); otherwise they are
 * the minimum and maximum of the source longitudes. `extentWidth` is the arc along the parallel
 * at the middle latitude of the extent (the arc length in degrees from west eastwards, times the
 * cosine of that latitude), `extentHeight` the meridian arc, and `extentDiagonal` their
 * hypotenuse, all in metres, so two identical boxes at longitude 0 and at 180 have the same
 * sizes; a region that spans more than about 60 degrees of latitude is only roughly measured.
 * {@link getGPUAnalysisExtentSizes} applies the same measure to any extent.
 *
 * Sampling: neighbour distances are computed on all points when `pointCount` is at most
 * `maximumSampleSize`. Above it they are computed on a pseudo-random subset of exactly
 * `maximumSampleSize` rows: the rows whose index has the smallest values of a seeded 32-bit
 * integer hash (the murmur3 finalizer of `index xor 0x9e3779b9`; the index counts valid rows). The
 * subset depends on the row index only, so it is the same on every run and in the Python oracle
 * of the spec, and a file sorted by time, by cluster or by space is sampled as evenly as a
 * shuffled one (a fixed stride, used before, sampled one vehicle of 20 interleaved ones and made
 * the median 20.5 times too large on a time-ordered track). `sampled` is true and `sampleSize` is
 * the number of rows searched. A sampled point has fewer candidate neighbours than in the full
 * set, so its distance cannot shrink; aggregate statistics describe a different population and
 * may move in either direction. For a pattern that looks the same everywhere the increase is about
 * `sqrt(pointCount / sampleSize)` (see {@link getGPUAnalysisSamplingFactor}), for a line-like
 * pattern (a track) it is larger, for a lattice the sample can be sparser than the lattice and
 * the ratio has no simple form, and nothing here corrects the values. Measured on 100,000 points
 * sampled to 5,000 (a factor of 20, so 4.47 for a homogeneous pattern), the median
 * nearest-neighbour distance:
 *
 * | pattern (order of the rows) | full set | sample | sample / full | corrected / full |
 * | --- | --- | --- | --- | --- |
 * | 20 vehicles, interleaved | 0.162 | 0.737 | 4.54 | 1.02 |
 * | one time-ordered track | 0.964 | 6.95 | 7.22 | 1.61 |
 * | 10 tracks, one after another | 0.156 | 0.725 | 4.63 | 1.04 |
 * | 10 clusters sorted by cluster, densities 1 to 100 | 0.0287 | 0.136 | 4.76 | 1.06 |
 * | 316 x 316 lattice in row order (sample 4,999) | 1.00 | 2.24 | 2.24 | 0.50 |
 * | uniform random (control) | 1.49 | 6.69 | 4.50 | 1.01 |
 * | 50 clusters, random order | 0.0800 | 0.359 | 4.49 | 1.00 |
 *
 * Extent, centre, standard distance, median distance from the centre and `duplicateLocationCount`
 * always use every valid point, never the sample.
 *
 * `nearestNeighbor` is the distance from each point to its nearest other point. A repeated
 * location is another point, so a duplicate has a distance of 0, and `minimum` is 0 whenever
 * `duplicateLocationCount` is positive, even if the sample holds no pair (it is the full set's
 * minimum). `lowerQuartile` and `upperQuartile` use linear interpolation between order
 * statistics (NumPy's default); `upperFence` is the upper quartile plus three interquartile
 * ranges, capped at `maximum`; `aboveUpperFenceCount` counts the searched points with a distance
 * strictly above it (in a sample, sampled points). `nonZeroMedian` is the median, over distinct
 * locations, of the distance from each to its nearest other distinct location. It is computed
 * on the sample's distinct locations when sampled, using the same canonical location identity as
 * duplicate detection, and is absent with fewer than two distinct searched locations.
 * `kDistance.distances` holds the
 * ascending distances to the k-th nearest other point, reduced to at most
 * {@link GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT} evenly spaced quantiles that include the smallest
 * and the largest.
 *
 * Degenerate inputs are defined, never thrown:
 * - 0 valid points: the function returns `undefined`, because no extent, centre or spread exists
 *   (0 would be a wrong answer). No field of a returned result is ever NaN.
 * - 1 point: extent is that point, the extent sizes, `standardDistance` and
 *   `medianDistanceFromCenter` are 0, `nearestNeighbor` and `kDistance` are absent.
 * - All points identical: extent sizes and standard distance are 0, every neighbour distance is
 *   0, `duplicateLocationCount` is `pointCount - 1`.
 * - Collinear points: ordinary; the grid adapts (the extent of one side is 0).
 * - Fewer than `k + 1` points: `nearestNeighbor` is present (from 2 points), `kDistance` is absent.
 *
 * @param coordinates Alternating x and y values.
 * @param options Coordinate model, k-distance rank and sample limit.
 * @returns The spacing statistics, all distances in `unit`; `undefined` when no row is valid.
 * @throws RangeError for an odd coordinate length, a `neighborRank` that is not an integer in
 * [1, 64] or a `maximumSampleSize` that is not an integer of at least 2.
 */
export function computeGPUAnalysisSpacingStatistics(
  coordinates: Float64Array | Float32Array | readonly number[],
  options: GPUAnalysisSpacingOptions
): GPUAnalysisSpacingStatistics | undefined {
  const functionName = 'computeGPUAnalysisSpacingStatistics';
  const neighborRank = options.neighborRank ?? GPU_ANALYSIS_DEFAULT_NEIGHBOR_RANK;
  const maximumSampleSize = options.maximumSampleSize ?? GPU_ANALYSIS_DEFAULT_MAXIMUM_SAMPLE_SIZE;
  if (coordinates.length % 2 !== 0) {
    throw new RangeError(
      `${functionName} coordinates must hold x, y pairs, so an even length; got ${coordinates.length}`
    );
  }
  if (
    !Number.isInteger(neighborRank) ||
    neighborRank < 1 ||
    neighborRank > GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK
  ) {
    throw new RangeError(
      `${functionName} neighborRank must be an integer from 1 to ${GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK}; got ${neighborRank}`
    );
  }
  if (!Number.isInteger(maximumSampleSize) || maximumSampleSize < 2) {
    throw new RangeError(
      `${functionName} maximumSampleSize must be an integer of at least 2; got ${maximumSampleSize}`
    );
  }

  const isGeographic = options.coordinateModel === 'geographic';
  const valid = gatherValidPoints(coordinates, isGeographic);
  const pointCount = valid.count;
  const unit = isGeographic ? 'metre' : 'coordinate-unit';
  if (pointCount === 0) {
    return undefined;
  }

  const extent = isGeographic
    ? computeGeographicExtent(valid.x, valid.y)
    : computeExtent(valid.x, valid.y);
  const extentSizes = isGeographic
    ? getGeographicExtentSizes(extent)
    : {width: extent[2] - extent[0], height: extent[3] - extent[1]};

  // Preserve source longitude bounds above, but use one representation of each physical
  // location for duplicate detection, centres and neighbour searches.
  if (isGeographic) {
    canonicalizeGeographicPoints(valid);
  }
  const sphere = isGeographic ? createSpherePoints(valid.x, valid.y) : undefined;
  const centerInfo = sphere
    ? computeGeographicCenter(sphere, valid.x, valid.y)
    : computePlanarCenter(valid.x, valid.y);
  const distancesFromCenter = new Float64Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    distancesFromCenter[index] = sphere
      ? computeHaversineDistance(
          sphere.longitude[index],
          sphere.latitude[index],
          centerInfo.longitudeRadians,
          centerInfo.latitudeRadians
        )
      : Math.hypot(valid.x[index] - centerInfo.center[0], valid.y[index] - centerInfo.center[1]);
  }
  // An individual offset can exceed the number range even when the RMS remains representable.
  // Recompute in scaled coordinates in that case, then rescale only the aggregate statistics.
  let centerDistanceScale = 1;
  if (!sphere && distancesFromCenter.some(distance => !Number.isFinite(distance))) {
    for (let index = 0; index < pointCount; index++) {
      centerDistanceScale = Math.max(
        centerDistanceScale,
        Math.abs(valid.x[index]),
        Math.abs(valid.y[index])
      );
    }
    for (let index = 0; index < pointCount; index++) {
      distancesFromCenter[index] = Math.hypot(
        valid.x[index] / centerDistanceScale - centerInfo.center[0] / centerDistanceScale,
        valid.y[index] / centerDistanceScale - centerInfo.center[1] / centerDistanceScale
      );
    }
  }
  const standardDistance = computeRootMeanSquare(distancesFromCenter) * centerDistanceScale;
  const medianDistanceFromCenter =
    computeMedianOfUnsorted(distancesFromCenter) * centerDistanceScale;

  const distinctLocationIndices = selectDistinctLocationIndices(valid.x, valid.y);
  const duplicateLocationCount = pointCount - distinctLocationIndices.length;

  const sampleSize = Math.min(pointCount, maximumSampleSize);
  const sampled = sampleSize < pointCount;
  const sampleIndices = selectSampleIndices(pointCount, sampleSize);

  let nearestNeighbor: GPUAnalysisSpacingStatistics['nearestNeighbor'];
  let kDistance: GPUAnalysisSpacingStatistics['kDistance'];
  if (sampleSize >= 2) {
    const searched = findNeighborDistances(valid, sphere, sampleIndices, neighborRank);
    const sortedNearest = Float64Array.from(searched.nearest).sort();
    const count = sortedNearest.length;
    let sum = 0;
    for (let index = 0; index < count; index++) {
      sum += sortedNearest[index];
    }
    const maximum = sortedNearest[count - 1];
    const lowerQuartile = getQuantileOfSorted(sortedNearest, 0.25);
    const upperQuartile = getQuantileOfSorted(sortedNearest, 0.75);
    const upperFence = Math.min(
      maximum,
      upperQuartile === lowerQuartile
        ? upperQuartile
        : upperQuartile + UPPER_FENCE_INTERQUARTILE_RANGES * (upperQuartile - lowerQuartile)
    );
    let aboveUpperFenceCount = 0;
    for (let index = count - 1; index >= 0 && sortedNearest[index] > upperFence; index--) {
      aboveUpperFenceCount++;
    }
    const distinctSampleIndices = sampled
      ? selectDistinctLocationIndices(valid.x, valid.y, sampleIndices)
      : distinctLocationIndices;
    const nonZeroMedian =
      distinctSampleIndices.length < 2
        ? undefined
        : distinctSampleIndices.length === sampleSize
          ? getMedianOfSorted(sortedNearest)
          : getMedianOfSorted(
              findNeighborDistances(valid, sphere, distinctSampleIndices, 1).nearest.sort()
            );
    nearestNeighbor = {
      // The sample may miss a repeated location; the full set is known to hold one when
      // `duplicateLocationCount` is positive, and then the smallest distance is 0.
      minimum: duplicateLocationCount > 0 ? 0 : sortedNearest[0],
      lowerQuartile,
      median: getMedianOfSorted(sortedNearest),
      upperQuartile,
      mean: Number.isFinite(sum) ? sum / count : computeScaledMean(sortedNearest),
      maximum,
      upperFence,
      aboveUpperFenceCount,
      ...(nonZeroMedian === undefined ? {} : {nonZeroMedian})
    };
    if (searched.kth) {
      kDistance = {
        k: neighborRank,
        distances: reduceToQuantiles(Float64Array.from(searched.kth).sort())
      };
    }
  }

  const statistics: GPUAnalysisSpacingStatistics = {
    pointCount,
    sampleSize,
    sampled,
    coordinateModel: options.coordinateModel,
    unit,
    extent,
    extentWidth: extentSizes.width,
    extentHeight: extentSizes.height,
    extentDiagonal: Math.hypot(extentSizes.width, extentSizes.height),
    center: centerInfo.center,
    standardDistance,
    medianDistanceFromCenter,
    duplicateLocationCount
  };
  if (nearestNeighbor) {
    statistics.nearestNeighbor = nearestNeighbor;
  }
  if (kDistance) {
    statistics.kDistance = kDistance;
  }
  return statistics;
}

/**
 * Factor by which nearest-neighbour and k-th neighbour distances computed on a sample of
 * `sampleSize` points are too large compared with the full `pointCount`, for a homogeneous
 * (complete spatial randomness) pattern: multiply a sampled distance by `sqrt(sampleSize /
 * pointCount)` to estimate the full-set value. The expected neighbour distance of any rank scales
 * with `density^(-1/2)` in two dimensions (Clark and Evans 1954), and a sample holds
 * `sampleSize / pointCount` of the density. The factor is 1 when nothing was sampled.
 *
 * It is an estimate for a pattern that looks the same everywhere and for nothing else: on a
 * time-ordered single track the factor leaves the median 1.6 times too large, and on a lattice it
 * halves the value (see the table in {@link computeGPUAnalysisSpacingStatistics}). Nothing in this
 * module applies it.
 *
 * @throws RangeError if `sampleSize` is not an integer from 1 to `pointCount`.
 */
export function getGPUAnalysisSamplingFactor(pointCount: number, sampleSize: number): number {
  if (!Number.isInteger(pointCount) || !Number.isInteger(sampleSize)) {
    throw new RangeError(
      `getGPUAnalysisSamplingFactor pointCount and sampleSize must be integers; got ${pointCount} and ${sampleSize}`
    );
  }
  if (sampleSize < 1 || sampleSize > pointCount) {
    throw new RangeError(
      `getGPUAnalysisSamplingFactor sampleSize must be an integer from 1 to pointCount (${pointCount}); got ${sampleSize}`
    );
  }
  return Math.sqrt(sampleSize / pointCount);
}

// Points

type ValidPoints = {x: Float64Array; y: Float64Array; count: number};

function gatherValidPoints(
  coordinates: Float64Array | Float32Array | readonly number[],
  isGeographic: boolean
): ValidPoints {
  const rowCount = coordinates.length / 2;
  const x = new Float64Array(rowCount);
  const y = new Float64Array(rowCount);
  let count = 0;
  for (let row = 0; row < rowCount; row++) {
    const valueX = coordinates[2 * row];
    const valueY = coordinates[2 * row + 1];
    if (!Number.isFinite(valueX) || !Number.isFinite(valueY)) {
      continue;
    }
    if (isGeographic && (valueY < -90 || valueY > 90)) {
      continue;
    }
    x[count] = valueX;
    y[count] = valueY;
    count++;
  }
  return {x: x.subarray(0, count), y: y.subarray(0, count), count};
}

/** Longitude aliases and all longitudes at an exact pole identify one physical location. */
function canonicalizeGeographicPoints(points: ValidPoints): void {
  for (let index = 0; index < points.count; index++) {
    let longitude = points.x[index] % 360;
    if (longitude > 180) {
      longitude -= 360;
    } else if (longitude <= -180) {
      longitude += 360;
    }
    points.x[index] = Math.abs(points.y[index]) === 90 ? 0 : longitude;
  }
}

/** Scale before squaring so finite distances do not overflow or underflow their squares. */
function computeRootMeanSquare(values: Float64Array): number {
  let maximum = 0;
  for (const value of values) {
    maximum = Math.max(maximum, value);
  }
  if (maximum === 0 || !Number.isFinite(maximum)) {
    return maximum;
  }
  let scaledSquares = 0;
  for (const value of values) {
    scaledSquares += (value / maximum) ** 2;
  }
  return maximum * Math.sqrt(scaledSquares / values.length);
}

function computeExtent(
  x: Float64Array,
  y: Float64Array
): readonly [number, number, number, number] {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < x.length; index++) {
    minimumX = Math.min(minimumX, x[index]);
    maximumX = Math.max(maximumX, x[index]);
    minimumY = Math.min(minimumY, y[index]);
    maximumY = Math.max(maximumY, y[index]);
  }
  return [minimumX, minimumY, maximumX, maximumY];
}

/** Exact point covering arc; geometry profiling shares the same interval conventions. */
function computeGeographicExtent(
  longitude: Float64Array,
  latitude: Float64Array
): readonly [number, number, number, number] {
  const accumulator = new AnalysisGeographicExtentAccumulator();
  for (let index = 0; index < longitude.length; index++) {
    accumulator.add(longitude[index], latitude[index]);
  }
  return accumulator.getCovering() as [number, number, number, number];
}

/**
 * Width, height and diagonal of an extent `[west, south, east, north]`, the way the spacing
 * statistics measure their own: planar extents in coordinate units; geographic extents in metres
 * on the sphere of radius `GPU_EARTH_MEAN_RADIUS_METERS`, the width being the arc along the
 * parallel at the middle latitude (times the cosine of that latitude) and the height the meridian
 * arc. A geographic extent whose west is greater than its east crosses the antimeridian, and its
 * width is the arc from west eastwards to east (east minus west plus 360 degrees).
 */
export function getGPUAnalysisExtentSizes(
  extent: readonly [number, number, number, number],
  coordinateModel: GPUAnalysisCoordinateModel
): {width: number; height: number; diagonal: number} {
  const sizes =
    coordinateModel === 'geographic'
      ? getGeographicExtentSizes(extent)
      : {width: extent[2] - extent[0], height: extent[3] - extent[1]};
  return {...sizes, diagonal: Math.hypot(sizes.width, sizes.height)};
}

function getGeographicExtentSizes(extent: readonly [number, number, number, number]): {
  width: number;
  height: number;
} {
  const radiansPerDegree = Math.PI / 180;
  const middleLatitude = ((extent[1] + extent[3]) / 2) * radiansPerDegree;
  const arcDegrees = extent[2] >= extent[0] ? extent[2] - extent[0] : extent[2] - extent[0] + 360;
  return {
    width: GPU_EARTH_MEAN_RADIUS_METERS * Math.cos(middleLatitude) * arcDegrees * radiansPerDegree,
    height: GPU_EARTH_MEAN_RADIUS_METERS * (extent[3] - extent[1]) * radiansPerDegree
  };
}

/** First row at each canonical location, over all rows or only the supplied sample. */
function selectDistinctLocationIndices(
  coordinatesX: Float64Array,
  coordinatesY: Float64Array,
  sampleIndices?: Int32Array
): Int32Array {
  // Open-addressing table of the first point seen at each location, compared by value, so -0 and
  // 0 are one location.
  const pointCount = sampleIndices?.length ?? coordinatesX.length;
  let tableSize = MINIMUM_LOCATION_TABLE_SIZE;
  while (tableSize < 2 * pointCount) {
    tableSize *= 2;
  }
  const mask = tableSize - 1;
  const table = new Int32Array(tableSize).fill(-1);
  const bitsX = new Uint32Array(
    coordinatesX.buffer,
    coordinatesX.byteOffset,
    2 * coordinatesX.length
  );
  const bitsY = new Uint32Array(
    coordinatesY.buffer,
    coordinatesY.byteOffset,
    2 * coordinatesY.length
  );
  const distinctIndices = new Int32Array(pointCount);
  let distinctCount = 0;
  for (let position = 0; position < pointCount; position++) {
    const index = sampleIndices ? sampleIndices[position] : position;
    // Normalise -0 to 0 before hashing the bit patterns.
    const hashX =
      coordinatesX[index] === 0 ? 0 : Math.imul(bitsX[2 * index] ^ bitsX[2 * index + 1], 73856093);
    const hashY =
      coordinatesY[index] === 0 ? 0 : Math.imul(bitsY[2 * index] ^ bitsY[2 * index + 1], 19349663);
    let slot = (hashX ^ hashY ^ (hashX >>> 15) ^ (hashY >>> 13)) & mask;
    let isDuplicate = false;
    while (table[slot] !== -1) {
      const other = table[slot];
      if (
        coordinatesX[other] === coordinatesX[index] &&
        coordinatesY[other] === coordinatesY[index]
      ) {
        isDuplicate = true;
        break;
      }
      slot = (slot + 1) & mask;
    }
    if (!isDuplicate) {
      table[slot] = index;
      distinctIndices[distinctCount++] = index;
    }
  }
  return distinctIndices.subarray(0, distinctCount);
}

// Centres

type CenterInfo = {
  center: readonly [number, number];
  longitudeRadians: number;
  latitudeRadians: number;
};

function computePlanarCenter(x: Float64Array, y: Float64Array): CenterInfo {
  // Offsets from the first point keep the sum accurate for large coordinates with a small spread.
  let sumX = 0;
  let sumY = 0;
  for (let index = 0; index < x.length; index++) {
    sumX += x[index] - x[0];
    sumY += y[index] - y[0];
  }
  const center: [number, number] = [
    Number.isFinite(sumX) ? x[0] + sumX / x.length : computeScaledMean(x),
    Number.isFinite(sumY) ? y[0] + sumY / x.length : computeScaledMean(y)
  ];
  return {center, longitudeRadians: NaN, latitudeRadians: NaN};
}

/** Fallback when summing offsets overflows although their mean is representable. */
function computeScaledMean(values: Float64Array): number {
  let maximum = 0;
  for (const value of values) {
    maximum = Math.max(maximum, Math.abs(value));
  }
  if (!Number.isFinite(maximum)) {
    return maximum;
  }
  let sum = 0;
  for (const value of values) {
    sum += value / maximum / values.length;
  }
  return maximum * Math.max(-1, Math.min(1, sum));
}

type SpherePoints = {
  /** Longitude and latitude in radians. */
  longitude: Float64Array;
  latitude: Float64Array;
  /** Position on the sphere in metres. */
  sphereX: Float64Array;
  sphereY: Float64Array;
  sphereZ: Float64Array;
};

function createSpherePoints(
  longitudeDegrees: Float64Array,
  latitudeDegrees: Float64Array
): SpherePoints {
  const count = longitudeDegrees.length;
  const radiansPerDegree = Math.PI / 180;
  const longitude = new Float64Array(count);
  const latitude = new Float64Array(count);
  const sphereX = new Float64Array(count);
  const sphereY = new Float64Array(count);
  const sphereZ = new Float64Array(count);
  for (let index = 0; index < count; index++) {
    const longitudeRadians = longitudeDegrees[index] * radiansPerDegree;
    const latitudeRadians = latitudeDegrees[index] * radiansPerDegree;
    longitude[index] = longitudeRadians;
    latitude[index] = latitudeRadians;
    const cosineLatitude = Math.abs(latitudeDegrees[index]) === 90 ? 0 : Math.cos(latitudeRadians);
    sphereX[index] = GPU_EARTH_MEAN_RADIUS_METERS * cosineLatitude * Math.cos(longitudeRadians);
    sphereY[index] = GPU_EARTH_MEAN_RADIUS_METERS * cosineLatitude * Math.sin(longitudeRadians);
    sphereZ[index] = GPU_EARTH_MEAN_RADIUS_METERS * Math.sin(latitudeRadians);
  }
  return {longitude, latitude, sphereX, sphereY, sphereZ};
}

function computeGeographicCenter(
  sphere: SpherePoints,
  longitudeDegrees: Float64Array,
  latitudeDegrees: Float64Array
): CenterInfo {
  const count = sphere.longitude.length;
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  for (let index = 0; index < count; index++) {
    sumX += sphere.sphereX[index];
    sumY += sphere.sphereY[index];
    sumZ += sphere.sphereZ[index];
  }
  const norm = Math.hypot(sumX, sumY, sumZ) / count / GPU_EARTH_MEAN_RADIUS_METERS;
  const degreesPerRadian = 180 / Math.PI;
  if (norm < MINIMUM_UNIT_VECTOR_LENGTH) {
    let meanLongitude = 0;
    let meanLatitude = 0;
    for (let index = 0; index < count; index++) {
      meanLongitude += longitudeDegrees[index];
      meanLatitude += latitudeDegrees[index];
    }
    meanLongitude /= count;
    meanLatitude /= count;
    return {
      center: [meanLongitude, meanLatitude],
      longitudeRadians: meanLongitude / degreesPerRadian,
      latitudeRadians: meanLatitude / degreesPerRadian
    };
  }
  const longitudeRadians = Math.atan2(sumY, sumX);
  const latitudeRadians = Math.atan2(sumZ, Math.hypot(sumX, sumY));
  return {
    center: [longitudeRadians * degreesPerRadian, latitudeRadians * degreesPerRadian],
    longitudeRadians,
    latitudeRadians
  };
}

/** Haversine great-circle distance in metres between two points given in radians. */
function computeHaversineDistance(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number
): number {
  if (
    latitudeA === latitudeB &&
    (longitudeA === longitudeB || Math.abs(latitudeA) === Math.PI / 2)
  ) {
    return 0;
  }
  const halfLatitude = Math.sin((latitudeB - latitudeA) / 2);
  const halfLongitude = Math.sin((longitudeB - longitudeA) / 2);
  const haversine =
    halfLatitude * halfLatitude +
    Math.cos(latitudeA) * Math.cos(latitudeB) * halfLongitude * halfLongitude;
  const clamped = Math.min(1, Math.max(0, haversine));
  return 2 * GPU_EARTH_MEAN_RADIUS_METERS * Math.atan2(Math.sqrt(clamped), Math.sqrt(1 - clamped));
}

// Order statistics

/** Quantile with linear interpolation between order statistics (NumPy's default method). */
function getQuantileOfSorted(sorted: Float64Array, fraction: number): number {
  const position = fraction * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(sorted.length - 1, lower + 1);
  if (sorted[lower] === sorted[upper] || position === lower) {
    return sorted[lower];
  }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function getMedianOfSorted(sorted: Float64Array): number {
  return getQuantileOfSorted(sorted, 0.5);
}

function computeMedianOfUnsorted(values: Float64Array): number {
  return getMedianOfSorted(Float64Array.from(values).sort());
}

function reduceToQuantiles(sortedAscending: Float64Array): readonly number[] {
  const length = sortedAscending.length;
  if (length <= GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT) {
    return Array.from(sortedAscending);
  }
  const quantiles: number[] = [];
  for (let quantile = 0; quantile < GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT; quantile++) {
    const position = Math.round(
      (quantile * (length - 1)) / (GPU_ANALYSIS_K_DISTANCE_QUANTILE_COUNT - 1)
    );
    quantiles.push(sortedAscending[position]);
  }
  return quantiles;
}

// Sampling

/**
 * Seed of the sampling hash. The value is the 32-bit golden-ratio constant; any constant gives
 * a different but equally uniform subset.
 */
export const GPU_ANALYSIS_SAMPLING_SEED = 0x9e3779b9;

/**
 * 32-bit integer hash of a row index: the murmur3 finalizer of `index xor seed`. Both steps are
 * bijections on 32-bit integers, so different rows never share a hash.
 */
function hashRowIndex(index: number): number {
  let hash = (index ^ GPU_ANALYSIS_SAMPLING_SEED) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return hash >>> 0;
}

/**
 * Ascending indices (among the valid rows) of a pseudo-random subset of exactly `sampleSize`
 * rows: the rows with the `sampleSize` smallest values of {@link hashRowIndex}. The subset depends
 * only on the row index, not on the coordinates or on their order, so a file sorted by time,
 * cluster or space is sampled as evenly as a shuffled one, and the same input always gives the
 * same subset. Python: `fmix32(i ^ 0x9e3779b9)`, then `argsort(hashes)[:sampleSize]`, sorted.
 */
function selectSampleIndices(pointCount: number, sampleSize: number): Int32Array {
  const indices = new Int32Array(sampleSize);
  if (sampleSize === pointCount) {
    for (let index = 0; index < pointCount; index++) {
      indices[index] = index;
    }
    return indices;
  }
  const hashes = new Uint32Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    hashes[index] = hashRowIndex(index);
  }
  const threshold = Uint32Array.from(hashes).sort()[sampleSize - 1];
  let count = 0;
  for (let index = 0; index < pointCount; index++) {
    if (hashes[index] <= threshold) {
      indices[count++] = index;
    }
  }
  return indices;
}

// Projection axis

/**
 * Unit vector of least variance of the searched points on the sphere: the eigenvector of the
 * smallest eigenvalue of the 3 x 3 covariance matrix of their positions, found by cyclic Jacobi
 * rotations (a few sweeps converge a symmetric 3 x 3 matrix to rounding). All points at one
 * location give a zero matrix, and any axis is then fine.
 */
function findLeastVarianceAxis(
  sphere: SpherePoints,
  sampleIndices: Int32Array
): readonly [number, number, number] {
  const count = sampleIndices.length;
  let meanX = 0;
  let meanY = 0;
  let meanZ = 0;
  for (let sample = 0; sample < count; sample++) {
    const index = sampleIndices[sample];
    meanX += sphere.sphereX[index] / count;
    meanY += sphere.sphereY[index] / count;
    meanZ += sphere.sphereZ[index] / count;
  }
  // Row-major symmetric matrix; positions are divided by the radius so the entries are of order 1.
  const matrix = new Float64Array(9);
  for (let sample = 0; sample < count; sample++) {
    const index = sampleIndices[sample];
    const deltaX = (sphere.sphereX[index] - meanX) / GPU_EARTH_MEAN_RADIUS_METERS;
    const deltaY = (sphere.sphereY[index] - meanY) / GPU_EARTH_MEAN_RADIUS_METERS;
    const deltaZ = (sphere.sphereZ[index] - meanZ) / GPU_EARTH_MEAN_RADIUS_METERS;
    matrix[0] += deltaX * deltaX;
    matrix[1] += deltaX * deltaY;
    matrix[2] += deltaX * deltaZ;
    matrix[4] += deltaY * deltaY;
    matrix[5] += deltaY * deltaZ;
    matrix[8] += deltaZ * deltaZ;
  }
  matrix[3] = matrix[1];
  matrix[6] = matrix[2];
  matrix[7] = matrix[5];
  const vectors = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let sweep = 0; sweep < JACOBI_MAXIMUM_SWEEPS; sweep++) {
    const offDiagonal = matrix[1] ** 2 + matrix[2] ** 2 + matrix[5] ** 2;
    const diagonal = matrix[0] ** 2 + matrix[4] ** 2 + matrix[8] ** 2;
    if (offDiagonal <= 1e-30 * diagonal || offDiagonal === 0) {
      break;
    }
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2]
    ]) {
      const offValue = matrix[3 * p + q];
      if (offValue === 0) {
        continue;
      }
      const theta = (matrix[3 * q + q] - matrix[3 * p + p]) / (2 * offValue);
      const tangent = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const cosine = 1 / Math.sqrt(tangent * tangent + 1);
      const sine = tangent * cosine;
      for (let k = 0; k < 3; k++) {
        const columnP = matrix[3 * k + p];
        const columnQ = matrix[3 * k + q];
        matrix[3 * k + p] = cosine * columnP - sine * columnQ;
        matrix[3 * k + q] = sine * columnP + cosine * columnQ;
      }
      for (let k = 0; k < 3; k++) {
        const rowP = matrix[3 * p + k];
        const rowQ = matrix[3 * q + k];
        matrix[3 * p + k] = cosine * rowP - sine * rowQ;
        matrix[3 * q + k] = sine * rowP + cosine * rowQ;
      }
      for (let k = 0; k < 3; k++) {
        const vectorP = vectors[3 * k + p];
        const vectorQ = vectors[3 * k + q];
        vectors[3 * k + p] = cosine * vectorP - sine * vectorQ;
        vectors[3 * k + q] = sine * vectorP + cosine * vectorQ;
      }
    }
  }
  let smallest = 0;
  for (let column = 1; column < 3; column++) {
    if (matrix[3 * column + column] < matrix[3 * smallest + smallest]) {
      smallest = column;
    }
  }
  return [vectors[smallest], vectors[3 + smallest], vectors[6 + smallest]];
}

// Grid hash search

type GridHash = {
  cellSize: number;
  minimumU: number;
  minimumV: number;
  cellsU: number;
  cellsV: number;
  /** Open-addressing table of occupied cells: cell indices and the id of the cell they hold. */
  tableMask: number;
  tableU: Int32Array;
  tableV: Int32Array;
  tableCell: Int32Array;
  /** CSR layout: members of cell `id` are `members[cellStart[id] .. cellStart[id + 1])`. */
  cellStart: Int32Array;
  members: Int32Array;
  cellOfPoint: Int32Array;
  cellU: Int32Array;
  cellV: Int32Array;
  occupiedCellCount: number;
};

function hashCell(cellU: number, cellV: number, mask: number): number {
  return (Math.imul(cellU, 73856093) ^ Math.imul(cellV, 19349663)) & mask;
}

function buildGridHash(
  gridU: Float64Array,
  gridV: Float64Array,
  pointCount: number,
  bounds: readonly [number, number, number, number],
  cellSize: number,
  clampToBounds = false
): GridHash {
  const [minimumU, minimumV, maximumU, maximumV] = bounds;
  const cellsU = Math.floor((maximumU - minimumU) / cellSize) + 1;
  const cellsV = Math.floor((maximumV - minimumV) / cellSize) + 1;
  let tableSize = 16;
  while (tableSize < 2 * pointCount) {
    tableSize *= 2;
  }
  const tableMask = tableSize - 1;
  const tableU = new Int32Array(tableSize);
  const tableV = new Int32Array(tableSize);
  const tableCell = new Int32Array(tableSize).fill(-1);
  const cellOfPoint = new Int32Array(pointCount);
  const cellU = new Int32Array(pointCount);
  const cellV = new Int32Array(pointCount);
  let occupiedCellCount = 0;
  for (let index = 0; index < pointCount; index++) {
    const coordinateU = clampToBounds
      ? Math.max(minimumU, Math.min(maximumU, gridU[index]))
      : gridU[index];
    const coordinateV = clampToBounds
      ? Math.max(minimumV, Math.min(maximumV, gridV[index]))
      : gridV[index];
    const indexU = Math.floor((coordinateU - minimumU) / cellSize);
    const indexV = Math.floor((coordinateV - minimumV) / cellSize);
    cellU[index] = indexU;
    cellV[index] = indexV;
    let slot = hashCell(indexU, indexV, tableMask);
    while (tableCell[slot] !== -1 && (tableU[slot] !== indexU || tableV[slot] !== indexV)) {
      slot = (slot + 1) & tableMask;
    }
    if (tableCell[slot] === -1) {
      tableU[slot] = indexU;
      tableV[slot] = indexV;
      tableCell[slot] = occupiedCellCount;
      occupiedCellCount++;
    }
    cellOfPoint[index] = tableCell[slot];
  }
  const cellStart = new Int32Array(occupiedCellCount + 1);
  for (let index = 0; index < pointCount; index++) {
    cellStart[cellOfPoint[index] + 1]++;
  }
  for (let cell = 0; cell < occupiedCellCount; cell++) {
    cellStart[cell + 1] += cellStart[cell];
  }
  const fill = cellStart.slice(0, occupiedCellCount);
  const members = new Int32Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    members[fill[cellOfPoint[index]]++] = index;
  }
  return {
    cellSize,
    minimumU,
    minimumV,
    cellsU,
    cellsV,
    tableMask,
    tableU,
    tableV,
    tableCell,
    cellStart,
    members,
    cellOfPoint,
    cellU,
    cellV,
    occupiedCellCount
  };
}

function chooseGridHash(
  gridU: Float64Array,
  gridV: Float64Array,
  pointCount: number
): GridHash | undefined {
  let minimumU = Infinity;
  let minimumV = Infinity;
  let maximumU = -Infinity;
  let maximumV = -Infinity;
  for (let index = 0; index < pointCount; index++) {
    minimumU = Math.min(minimumU, gridU[index]);
    maximumU = Math.max(maximumU, gridU[index]);
    minimumV = Math.min(minimumV, gridV[index]);
    maximumV = Math.max(maximumV, gridV[index]);
  }
  const bounds = [minimumU, minimumV, maximumU, maximumV] as const;
  const grid = buildRefinedGridHash(gridU, gridV, pointCount, bounds);
  if (!grid) return undefined;
  let maximumPopulation = 0;
  for (let cell = 0; cell < grid.occupiedCellCount; cell++) {
    maximumPopulation = Math.max(
      maximumPopulation,
      grid.cellStart[cell + 1] - grid.cellStart[cell]
    );
  }
  if (maximumPopulation <= MAXIMUM_CELL_POPULATION_BEFORE_ROBUST_GRID) return grid;

  const sortedU = gridU.slice().sort();
  const sortedV = gridV.slice().sort();
  const tailCount = Math.floor((pointCount - 1) * ROBUST_GRID_TAIL_SHARE);
  const robustBounds = [
    sortedU[tailCount],
    sortedV[tailCount],
    sortedU[pointCount - 1 - tailCount],
    sortedV[pointCount - 1 - tailCount]
  ] as const;
  // Clamping each axis cannot increase projected distance. Ring distances in this clamped
  // plane therefore remain lower bounds on true distances, even for points outside the bounds.
  // Boundary cells retain every outlier; no candidate is discarded or sampled away.
  return buildRefinedGridHash(gridU, gridV, pointCount, robustBounds, true) ?? grid;
}

function buildRefinedGridHash(
  gridU: Float64Array,
  gridV: Float64Array,
  pointCount: number,
  bounds: readonly [number, number, number, number],
  clampToBounds = false
): GridHash | undefined {
  const [minimumU, minimumV, maximumU, maximumV] = bounds;
  const sizeU = maximumU - minimumU;
  const sizeV = maximumV - minimumV;
  const longestSide = Math.max(sizeU, sizeV);
  if (!(longestSide > 0)) {
    return undefined;
  }
  const minimumCellSize = longestSide / (MAXIMUM_CELLS_PER_AXIS - 1);
  let cellSize = Math.sqrt((TARGET_POINTS_PER_CELL * sizeU * sizeV) / pointCount);
  if (!(cellSize > 0)) {
    cellSize = (TARGET_POINTS_PER_CELL * longestSide) / pointCount;
  }
  cellSize = Math.max(cellSize, minimumCellSize);
  let grid = buildGridHash(gridU, gridV, pointCount, bounds, cellSize, clampToBounds);
  for (let attempt = 0; attempt < MAXIMUM_GRID_REFINEMENTS; attempt++) {
    const pointsPerCell = pointCount / grid.occupiedCellCount;
    if (pointsPerCell <= REFINE_POINTS_PER_CELL || cellSize <= minimumCellSize) {
      break;
    }
    cellSize = Math.max(
      minimumCellSize,
      cellSize * Math.sqrt(TARGET_POINTS_PER_CELL / pointsPerCell)
    );
    grid = buildGridHash(gridU, gridV, pointCount, bounds, cellSize, clampToBounds);
  }
  return grid;
}

function findCellId(grid: GridHash, cellU: number, cellV: number): number {
  if (cellU < 0 || cellV < 0 || cellU >= grid.cellsU || cellV >= grid.cellsV) {
    return -1;
  }
  let slot = hashCell(cellU, cellV, grid.tableMask);
  while (grid.tableCell[slot] !== -1) {
    if (grid.tableU[slot] === cellU && grid.tableV[slot] === cellV) {
      return grid.tableCell[slot];
    }
    slot = (slot + 1) & grid.tableMask;
  }
  return -1;
}

/**
 * Distances to the nearest and k-th nearest other point of every searched point. `kth` is present
 * only when every searched point has at least `k` other points.
 */
function findNeighborDistances(
  valid: ValidPoints,
  sphere: SpherePoints | undefined,
  sampleIndices: Int32Array,
  neighborRank: number
): {nearest: Float64Array; kth?: Float64Array} {
  const pointCount = sampleIndices.length;
  const nearest = new Float64Array(pointCount);
  const kth = pointCount > neighborRank ? new Float64Array(pointCount) : undefined;

  // Planar: grid the coordinates. Geographic: grid the orthographic projection (see the doc comment).
  const gridU = new Float64Array(pointCount);
  const gridV = new Float64Array(pointCount);
  const positionX = new Float64Array(pointCount);
  const positionY = new Float64Array(pointCount);
  const positionZ = sphere ? new Float64Array(pointCount) : undefined;
  const longitude = sphere ? new Float64Array(pointCount) : undefined;
  const latitude = sphere ? new Float64Array(pointCount) : undefined;
  if (sphere && positionZ && longitude && latitude) {
    // Projection axis: the direction of least variance of the searched points on the sphere (the
    // eigenvector of the smallest eigenvalue of their 3 x 3 covariance matrix). The two other
    // principal directions span the plane that is gridded, so a patch is seen from above, a band
    // along a great circle is seen from the side of its plane, and nothing folds onto a segment.
    const axis = findLeastVarianceAxis(sphere, sampleIndices);
    const [axisX, axisY, axisZ] = axis;
    // Any unit vector not parallel to the axis gives a first basis vector of the projection plane.
    const absoluteX = Math.abs(axisX);
    const absoluteY = Math.abs(axisY);
    const absoluteZ = Math.abs(axisZ);
    const helperX = absoluteX <= absoluteY && absoluteX <= absoluteZ ? 1 : 0;
    const helperY = absoluteY < absoluteX && absoluteY <= absoluteZ ? 1 : 0;
    const helperZ = helperX === 0 && helperY === 0 ? 1 : 0;
    let firstX = axisY * helperZ - axisZ * helperY;
    let firstY = axisZ * helperX - axisX * helperZ;
    let firstZ = axisX * helperY - axisY * helperX;
    const firstNorm = Math.hypot(firstX, firstY, firstZ);
    firstX /= firstNorm;
    firstY /= firstNorm;
    firstZ /= firstNorm;
    const secondX = axisY * firstZ - axisZ * firstY;
    const secondY = axisZ * firstX - axisX * firstZ;
    const secondZ = axisX * firstY - axisY * firstX;
    for (let sample = 0; sample < pointCount; sample++) {
      const index = sampleIndices[sample];
      const pointX = sphere.sphereX[index];
      const pointY = sphere.sphereY[index];
      const pointZ = sphere.sphereZ[index];
      positionX[sample] = pointX;
      positionY[sample] = pointY;
      positionZ[sample] = pointZ;
      longitude[sample] = sphere.longitude[index];
      latitude[sample] = sphere.latitude[index];
      gridU[sample] = pointX * firstX + pointY * firstY + pointZ * firstZ;
      gridV[sample] = pointX * secondX + pointY * secondY + pointZ * secondZ;
    }
  } else {
    for (let sample = 0; sample < pointCount; sample++) {
      const index = sampleIndices[sample];
      positionX[sample] = valid.x[index];
      positionY[sample] = valid.y[index];
      gridU[sample] = valid.x[index];
      gridV[sample] = valid.y[index];
    }
  }

  // Keep grid arithmetic in range even when products of finite extents overflow or
  // underflow. Candidate distances retain the original source coordinates.
  let gridScale = 1;
  if (!sphere) {
    let maximumCoordinate = 0;
    for (let sample = 0; sample < pointCount; sample++) {
      maximumCoordinate = Math.max(
        maximumCoordinate,
        Math.abs(gridU[sample]),
        Math.abs(gridV[sample])
      );
    }
    let maximumDifference = 0;
    for (let sample = 0; sample < pointCount; sample++) {
      maximumDifference = Math.max(
        maximumDifference,
        Math.abs(positionX[sample] - positionX[0]),
        Math.abs(positionY[sample] - positionY[0])
      );
    }
    const useOffsets = Number.isFinite(maximumDifference);
    gridScale = (useOffsets ? maximumDifference : maximumCoordinate) || 1;
    for (let sample = 0; sample < pointCount; sample++) {
      gridU[sample] = useOffsets
        ? (positionX[sample] - positionX[0]) / gridScale
        : positionX[sample] / gridScale;
      gridV[sample] = useOffsets
        ? (positionY[sample] - positionY[0]) / gridScale
        : positionY[sample] / gridScale;
    }
  }
  const grid = chooseGridHash(gridU, gridV, pointCount);
  if (!grid) {
    // Every point shares one location (in the projection; on a sphere that is one location too,
    // up to rounding of the projection, which is below any distance worth reporting).
    nearest.fill(0);
    kth?.fill(0);
    return {nearest, kth};
  }

  const bestMeasures = new Float64Array(neighborRank);
  const bestIndex = new Int32Array(neighborRank);
  const maximumRing = Math.max(grid.cellsU, grid.cellsV);
  const ringLookupBudget = Math.max(
    MINIMUM_RING_LOOKUP_BUDGET,
    pointCount / RING_LOOKUPS_PER_POINT_SCAN
  );
  const cellSize = grid.cellSize;
  const {cellStart, members} = grid;
  const isSphere = positionZ !== undefined;

  let found = 0;
  let done = false;
  let originX = 0;
  let originY = 0;
  let originZ = 0;
  let point = 0;
  // Offers one candidate to the ascending list of the best `neighborRank` points.
  const considerCandidate = (candidate: number): void => {
    if (candidate === point) {
      return;
    }
    const differenceX = positionX[candidate] - originX;
    const differenceY = positionY[candidate] - originY;
    // Sphere coordinates are bounded, so squared chords are safe and cheap to compare.
    // Planar distances need Math.hypot to scale differences before squaring.
    const distanceMeasure = positionZ
      ? differenceX * differenceX +
        differenceY * differenceY +
        (positionZ[candidate] - originZ) ** 2
      : Math.hypot(differenceX, differenceY);
    if (found === neighborRank && distanceMeasure >= bestMeasures[neighborRank - 1]) {
      return;
    }
    let position = found < neighborRank ? found : neighborRank - 1;
    while (position > 0 && bestMeasures[position - 1] > distanceMeasure) {
      bestMeasures[position] = bestMeasures[position - 1];
      bestIndex[position] = bestIndex[position - 1];
      position--;
    }
    bestMeasures[position] = distanceMeasure;
    bestIndex[position] = candidate;
    if (found < neighborRank) {
      found++;
    }
    // `neighborRank` points at distance 0 cannot be improved: stop, so a location repeated
    // many times does not cost the square of its multiplicity.
    if (found === neighborRank && bestMeasures[neighborRank - 1] === 0) {
      done = true;
    }
  };
  const scanCell = (cellU: number, cellV: number): void => {
    const cellId = findCellId(grid, cellU, cellV);
    if (cellId < 0) {
      return;
    }
    for (let slot = cellStart[cellId]; slot < cellStart[cellId + 1] && !done; slot++) {
      considerCandidate(members[slot]);
    }
  };

  for (point = 0; point < pointCount; point++) {
    found = 0;
    done = false;
    originX = positionX[point];
    originY = positionY[point];
    originZ = positionZ ? positionZ[point] : 0;
    const centerU = grid.cellU[point];
    const centerV = grid.cellV[point];

    let lookupCount = 0;
    for (let ring = 0; ring <= maximumRing && !done; ring++) {
      lookupCount += 8 * ring;
      if (lookupCount > ringLookupBudget) {
        // The cells looked up for this point so far, and the next ring, cost more than comparing it
        // with every point (an isolated point far from the rest, or a layout the grid fits badly):
        // scanning every point is cheaper, and exact. The budget bounds the work per point.
        found = 0;
        for (let candidate = 0; candidate < pointCount && !done; candidate++) {
          considerCandidate(candidate);
        }
        break;
      }
      if (ring === 0) {
        scanCell(centerU, centerV);
      } else {
        for (let offset = -ring; offset <= ring && !done; offset++) {
          scanCell(centerU + offset, centerV - ring);
          scanCell(centerU + offset, centerV + ring);
        }
        for (let offset = -ring + 1; offset <= ring - 1 && !done; offset++) {
          scanCell(centerU - ring, centerV + offset);
          scanCell(centerU + ring, centerV + offset);
        }
      }
      // A point in an unvisited ring is at least `ring * cellSize` away in the grid plane, which
      // never exceeds its true distance.
      const reach = ring * cellSize * gridScale;
      const reachMeasure = isSphere ? reach * reach : reach;
      if (found === neighborRank && bestMeasures[neighborRank - 1] <= reachMeasure) {
        break;
      }
    }

    if (found === 0) {
      nearest[point] = NaN;
      continue;
    }
    nearest[point] = getPairDistance(
      point,
      bestIndex[0],
      bestMeasures[0],
      isSphere,
      longitude,
      latitude
    );
    if (kth && found === neighborRank) {
      kth[point] = getPairDistance(
        point,
        bestIndex[neighborRank - 1],
        bestMeasures[neighborRank - 1],
        isSphere,
        longitude,
        latitude
      );
    }
  }
  return {nearest, kth};
}

/** Euclidean distance for planar points, haversine of the chosen pair for geographic points. */
function getPairDistance(
  point: number,
  neighbor: number,
  distance: number,
  isSphere: boolean,
  longitude: Float64Array | undefined,
  latitude: Float64Array | undefined
): number {
  if (isSphere && longitude && latitude) {
    return computeHaversineDistance(
      longitude[point],
      latitude[point],
      longitude[neighbor],
      latitude[neighbor]
    );
  }
  return distance;
}
