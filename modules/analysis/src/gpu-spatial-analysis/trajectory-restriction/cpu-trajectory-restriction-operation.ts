// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getCPUTrajectoryRows} from '../../analysis-core/operations/cpu-trajectory-rows';
import {computeTriangulationOrientation} from '../../analysis-core/polygon-triangulation/cpu-triangulation-predicates';
import {joinTimeWords} from '../../dataframe-analysis/time-window-filter/time-words';
import {GPU_TRAJECTORY_RESTRICTION_STATUS as FLAGS} from './gpu-trajectory-restriction';
import {
  GPUTrajectoryRestrictionOperation,
  validateRestrictionInputNames,
  type TrajectoryRestrictionOperationProps
} from './gpu-trajectory-restriction-operation';

/** Packed input rows. Int64 times and origin are little-endian signed low/high words. */
export type CPUTrajectoryRestrictionInput = {
  positions: Float32Array;
  timestamps: Float32Array | Uint32Array;
  trackOffsets: Uint32Array;
  timeOrigin?: Uint32Array;
  /** The contributor's seven scalar parameters followed by polygon coordinates. No packer exists. */
  parameters: Float32Array;
  pairs?: {first: Uint32Array; second: Uint32Array; count?: Uint32Array};
};

/** Packed positions, timestamps and track offsets for the CPU restriction form. */
export type CPUTrajectoryRestrictionGraphInput = CPUTrajectoryRestrictionInput;

/** Output capacities and compile-time predicate choices. GPU scratch and shader caps are absent. */
export type CPUTrajectoryRestrictionOptions = Pick<
  TrajectoryRestrictionOperationProps,
  | 'pairwiseTracks'
  | 'intervalsPerPair'
  | 'polygonRingVertexCounts'
  | 'timeBoundary'
  | 'geometryBoundary'
  | 'intervalsPerSegment'
  | 'mergeIntervals'
  | 'spatialContext'
  | 'compactCapacity'
>;

/** Every output, including cleared unused slots; compact fields exist only with compactCapacity. */
export type CPUTrajectoryRestrictionResult = {
  intervals: Float32Array;
  positions: Float32Array;
  status: Uint32Array;
  relationships: Uint32Array;
  compactIntervals?: Float32Array;
  compactPositions?: Float32Array;
  compactSources?: Uint32Array;
  compactStatus?: Uint32Array;
  compactSummary?: Uint32Array;
};

type Point = readonly [number, number];
type Segment = {
  time: number;
  finish: number;
  rawFinish: number;
  duration: number;
  position: Point;
  movement: Point;
  gap: boolean;
  invalid: boolean;
};
type IntervalRow = {
  begin: number;
  end: number;
  positions: readonly number[];
  flags: number;
  source: number;
  firstSource: number;
  track: number;
};

function subtractPoints(first: Point, second: Point): Point {
  return [first[0] - second[0], first[1] - second[1]];
}
function computeDot(first: Point, second: Point): number {
  return first[0] * second[0] + first[1] * second[1];
}
function computeCross(first: Point, second: Point): number {
  return first[0] * second[1] - first[1] * second[0];
}
function interpolatePoint(first: Point, movement: Point, fraction: number): Point {
  return [first[0] + movement[0] * fraction, first[1] + movement[1] * fraction];
}

function validateOptions(
  id: string,
  options: CPUTrajectoryRestrictionOptions,
  pairwise: boolean
): void {
  resolveGPUSpatialContext(
    id,
    options.spatialContext,
    pairwise
      ? {...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT, measure: 'distance'}
      : GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
  );
  validateCapacity(id, 'intervalsPerSegment', options.intervalsPerSegment ?? 1);
  for (const count of options.polygonRingVertexCounts ?? [])
    validateCapacity(id, 'polygonRingVertexCounts entry', count, 3);
  validateRequirement(
    id,
    options.timeBoundary === undefined ||
      ['closed', 'open', 'left-closed', 'right-closed'].includes(options.timeBoundary),
    'timeBoundary must be closed, open, left-closed or right-closed'
  );
  validateRequirement(
    id,
    options.geometryBoundary === undefined || ['closed', 'open'].includes(options.geometryBoundary),
    'geometryBoundary must be closed or open'
  );
  if (options.compactCapacity !== undefined)
    validateCapacity(id, 'compactCapacity', options.compactCapacity, 0);
  validateRequirement(
    id,
    !options.mergeIntervals || options.compactCapacity !== undefined,
    'mergeIntervals requires compactCapacity'
  );
  if (pairwise) {
    validateCapacity(id, 'intervalsPerPair', options.intervalsPerPair!);
    validateRequirement(
      id,
      options.polygonRingVertexCounts === undefined &&
        options.compactCapacity === undefined &&
        options.intervalsPerSegment === undefined,
      'pairwise mode excludes polygonRingVertexCounts, compactCapacity and intervalsPerSegment'
    );
  } else
    validateRequirement(
      id,
      options.intervalsPerPair === undefined,
      'intervalsPerPair requires pairwise mode'
    );
}

/**
 * Restricts planar interpolated tracks to a box, even-odd polygon or continuous pairwise distance.
 * Preserves slot order, endpoint policies, last-fix singletons, gap support, relationship counts,
 * stable interval union and the contributor's three-word compact summary.
 *
 * What differs from the GPU contributor:
 * - Arithmetic and compaction use float64, rounded once at publication. Threshold decisions,
 *   root endpoint inclusion and very narrow polygon pieces can differ near rounding boundaries.
 * - Polygon vertices, static pairs and interval capacities have no GPU shader/array maxima.
 *   Caller-selected slot capacities still drop excess intervals and set capacityExceeded.
 * - Exact orientation uses the existing arbitrary-precision CPU predicate, without a GPU
 *   exponent-span limit. Subnormal coordinates retain their values.
 * - Int64 times are subtracted exactly before conversion to float64, including relative origin
 *   times; the GPU's intermediate float32 word-time table is not reproduced.
 * - Unowned observation rows are initialized to NaN intervals/positions and zero status; the
 *   GPU leaves those rows unwritten. Empty inputs and zero compact capacity are supported.
 *   Offsets are clamped independently; reversed track ranges are empty, as on the GPU.
 *   Overlapping nonempty geometry ranges throw an Error naming trackOffsets, because their GPU
 *   invocations race on shared output slots. Pairwise ranges may overlap.
 * - Nonfinite fixes/parameters follow the GPU invalid/gap branches. Finite values whose
 *   intermediate arithmetic overflows float32 can remain valid on the CPU.
 *
 * @throws Error naming the argument for inconsistent packed arrays or invalid options.
 */
export function computeTrajectoryRestrictionOnCPU(
  input: CPUTrajectoryRestrictionInput,
  options: CPUTrajectoryRestrictionOptions = {}
): CPUTrajectoryRestrictionResult {
  const id = 'trajectory-restriction';
  validateRestrictionInputNames(id, input);
  const {positions, parameters, pairs, timestamps, trackOffsets} = input;
  const pairwise = pairs !== undefined || options.pairwiseTracks !== undefined;
  validateOptions(id, options, pairwise);
  if (pairs && options.pairwiseTracks)
    throw new Error('pairs and pairwiseTracks must not both be given');
  if (trackOffsets.length < 1) throw new Error('trackOffsets must contain a terminal offset');
  // Reuse the packed-array checks and exact timestamp subtraction, without imposing the shared
  // helper's monotonic partition policy: restriction clamps each track endpoint independently.
  const trajectory = getCPUTrajectoryRows({
    positions,
    timestamps,
    trackOffsets: Uint32Array.of(0, positions.length / 2)
  });
  const rowCount = trajectory.rowCount;
  const offsets = trackOffsets.map(offset => Math.min(offset, rowCount));
  const trackCount = offsets.length - 1;
  if (!pairwise) {
    const ranges = Array.from({length: trackCount}, (_, track) => [
      offsets[track],
      offsets[track + 1]
    ])
      .filter(([start, end]) => start < end)
      .sort(([firstStart], [secondStart]) => firstStart - secondStart);
    for (let range = 1; range < ranges.length; range++) {
      if (ranges[range][0] < ranges[range - 1][1])
        throw new Error(
          'trackOffsets must not assign an observation to overlapping geometry tracks'
        );
    }
  }
  const wordTimes = timestamps instanceof Uint32Array;
  if (wordTimes !== (input.timeOrigin !== undefined))
    throw new Error('timeOrigin is required only for word times');
  if (input.timeOrigin && input.timeOrigin.length < 2)
    throw new Error('timeOrigin must contain at least two words');
  const origin = input.timeOrigin ? joinTimeWords(input.timeOrigin[0], input.timeOrigin[1]) : 0n;
  const getTime = (row: number): number =>
    wordTimes
      ? Number(joinTimeWords(timestamps[row * 2], timestamps[row * 2 + 1]) - origin)
      : timestamps[row];
  const getPosition = (row: number): Point => [positions[row * 2], positions[row * 2 + 1]];
  const rings = options.polygonRingVertexCounts ?? [];
  const vertexCount = rings.reduce((total, count) => total + count, 0);
  if (parameters.length !== 7 + 2 * vertexCount)
    throw new Error('parameters must contain seven scalars and exactly the polygon vertices');
  if (
    pairs &&
    (pairs.first.length !== pairs.second.length || (pairs.count && pairs.count.length < 1))
  )
    throw new Error('pairs first/second lengths must agree and count must contain a row');
  for (const pair of options.pairwiseTracks ?? []) {
    if (
      pair.length !== 2 ||
      !pair.every(track => Number.isInteger(track) && track >= 0 && track < trackCount)
    )
      throw new Error('pairwiseTracks must contain pairs of track indices below the track count');
  }
  const groupCount = pairwise
    ? (pairs?.first.length ?? options.pairwiseTracks!.length)
    : trackCount;
  const slots = pairwise ? options.intervalsPerPair! : (options.intervalsPerSegment ?? 1);
  const intervalCount = pairwise ? groupCount * slots : rowCount * slots;
  validateCapacity(id, 'output interval rows', intervalCount, 0);
  const intervalValues = new Float64Array(intervalCount * 2).fill(Number.NaN);
  const positionValues = new Float64Array(intervalCount * 4).fill(Number.NaN);
  const status = new Uint32Array(intervalCount);
  const relationships = new Uint32Array(groupCount * 4);
  const [minimumTime, maximumTime, minimumX, minimumY, maximumX, maximumY, rawMaximumGap] =
    parameters;
  const maximumGap = rawMaximumGap === 0 ? Number.POSITIVE_INFINITY : rawMaximumGap;
  const closedStart = options.timeBoundary !== 'open' && options.timeBoundary !== 'right-closed';
  const closedEnd = options.timeBoundary !== 'open' && options.timeBoundary !== 'left-closed';
  const closedGeometry = options.geometryBoundary !== 'open';
  const timeIncluded = (time: number) =>
    (closedStart ? time >= minimumTime : time > minimumTime) &&
    (closedEnd ? time <= maximumTime : time < maximumTime);
  const validTimes =
    Number.isFinite(minimumTime) &&
    Number.isFinite(maximumTime) &&
    minimumTime <= maximumTime &&
    maximumGap > 0;
  const validParameters = pairwise
    ? validTimes && Number.isFinite(minimumX) && minimumX >= 0
    : validTimes &&
      Array.from(parameters).every((value, index) => index === 6 || Number.isFinite(value)) &&
      (vertexCount > 0 || (minimumX <= maximumX && minimumY <= maximumY));
  const writeRelationship = (
    group: number,
    anyInside: boolean,
    allInside: boolean,
    support: number,
    unknown: number
  ) => {
    const uncertain = support === 0 || unknown > 0;
    relationships.set(
      [
        anyInside ? 1 : uncertain ? 0xffffffff : 0,
        allInside ? (uncertain ? 0xffffffff : 1) : 0,
        support,
        unknown
      ],
      group * 4
    );
  };
  const writeInterval = (
    source: number,
    slot: number,
    begin: number,
    end: number,
    first: Point,
    last: Point,
    flags: number
  ) => {
    if (slot >= slots) {
      status[source * slots] |= FLAGS.capacityExceeded;
      return;
    }
    const outputRow = source * slots + slot;
    intervalValues.set([begin, end], outputRow * 2);
    const clipped = [first[0], first[1], last[0], last[1]];
    positionValues.set(clipped, outputRow * 4);
    status[outputRow] |= flags;
  };
  const loadSegment = (row: number, end: number, collapseGap: boolean): Segment => {
    const next = Math.min(row + 1, end - 1);
    const time = getTime(row);
    const rawFinish = getTime(next);
    const rawDuration = trajectory.getTimeDifference(next, row);
    // In no-limit mode an infinite next timestamp still follows the GPU's gap-collapse branch;
    // a finite float64 duration is never an artificial gap merely for exceeding float32 range.
    const gap = rawDuration > maximumGap || (rawMaximumGap === 0 && rawDuration === Infinity);
    const duration = collapseGap && gap ? 0 : rawDuration;
    const finish = collapseGap && gap ? time : rawFinish;
    const position = getPosition(row);
    const movement: Point =
      collapseGap && gap ? [0, 0] : subtractPoints(getPosition(next), position);
    const invalid =
      ![time, finish, ...position, ...movement].every(Number.isFinite) ||
      duration < 0 ||
      (duration === 0 && (movement[0] !== 0 || movement[1] !== 0));
    return {time, finish, rawFinish, duration, position, movement, gap, invalid};
  };
  if (pairwise) {
    for (let pair = 0; pair < groupCount; pair++) {
      const base = pair * slots;
      if (pairs?.count && pair >= pairs.count[0]) {
        relationships.set([0xffffffff, 0xffffffff, 0, 0], pair * 4);
        continue;
      }
      const firstTrack = pairs ? pairs.first[pair] : options.pairwiseTracks![pair][0];
      const secondTrack = pairs ? pairs.second[pair] : options.pairwiseTracks![pair][1];
      if (firstTrack >= trackCount || secondTrack >= trackCount || !validParameters) {
        status[base] = FLAGS.invalid;
        relationships.set([0xffffffff, 0xffffffff, 0, 1], pair * 4);
        continue;
      }
      const firstSegments: Segment[] = [];
      const secondSegments: Segment[] = [];
      for (let row = offsets[firstTrack]; row < offsets[firstTrack + 1]; row++)
        firstSegments.push(loadSegment(row, offsets[firstTrack + 1], true));
      for (let row = offsets[secondTrack]; row < offsets[secondTrack + 1]; row++)
        secondSegments.push(loadSegment(row, offsets[secondTrack + 1], true));
      let unknown =
        firstSegments.filter(segment => segment.invalid).length +
        secondSegments.filter(segment => segment.invalid).length;
      if (unknown > 0) status[base] |= FLAGS.invalid;
      let support = 0;
      let written = 0;
      let anyInside = false;
      let allInside = true;
      let firstSecond = 0;
      for (const first of firstSegments) {
        if (first.invalid) continue;
        if (first.time > maximumTime) break;
        if (first.rawFinish < minimumTime) continue;
        while (
          firstSecond < secondSegments.length &&
          (secondSegments[firstSecond].invalid ||
            secondSegments[firstSecond].rawFinish < first.time)
        )
          firstSecond++;
        for (let secondIndex = firstSecond; secondIndex < secondSegments.length; secondIndex++) {
          const second = secondSegments[secondIndex];
          if (second.invalid) continue;
          if (second.time > first.rawFinish || second.time > maximumTime) break;
          const lower = Math.max(minimumTime, first.time, second.time);
          const upper = Math.min(maximumTime, first.finish, second.finish);
          if (
            (first.gap || second.gap) &&
            lower < Math.min(maximumTime, first.rawFinish, second.rawFinish)
          ) {
            unknown++;
            status[base] |= FLAGS.gap;
          }
          if (lower > upper || (lower === upper && !timeIncluded(lower))) continue;
          support++;
          const firstVelocity: Point =
            first.duration > 0
              ? [first.movement[0] / first.duration, first.movement[1] / first.duration]
              : [0, 0];
          const secondVelocity: Point =
            second.duration > 0
              ? [second.movement[0] / second.duration, second.movement[1] / second.duration]
              : [0, 0];
          const firstPoint = interpolatePoint(first.position, firstVelocity, lower - first.time);
          const relative = subtractPoints(
            firstPoint,
            interpolatePoint(second.position, secondVelocity, lower - second.time)
          );
          const velocity = subtractPoints(firstVelocity, secondVelocity);
          const quadratic = computeDot(velocity, velocity);
          const limitSquared = minimumX * minimumX;
          let begin = 0;
          let finish = upper - lower;
          let matches = true;
          if (quadratic === 0)
            matches = closedGeometry
              ? computeDot(relative, relative) <= limitSquared
              : computeDot(relative, relative) < limitSquared;
          else {
            const center = -computeDot(relative, velocity) / quadratic;
            const closest = interpolatePoint(relative, velocity, center);
            const remainingSquared = limitSquared - computeDot(closest, closest);
            if (remainingSquared < 0) matches = false;
            else {
              const radius = Math.sqrt(remainingSquared / quadratic);
              begin = Math.max(begin, center - radius);
              finish = Math.min(finish, center + radius);
              matches = begin <= finish;
            }
          }
          const beginTime = lower + begin;
          const finishTime = lower + finish;
          const beginRelative = interpolatePoint(relative, velocity, begin);
          const finishRelative = interpolatePoint(relative, velocity, finish);
          const beginInside = closedGeometry
            ? computeDot(beginRelative, beginRelative) <= limitSquared
            : computeDot(beginRelative, beginRelative) < limitSquared;
          const finishInside = closedGeometry
            ? computeDot(finishRelative, finishRelative) <= limitSquared
            : computeDot(finishRelative, finishRelative) < limitSquared;
          const includeStart = timeIncluded(beginTime) && beginInside;
          const includeEnd = timeIncluded(finishTime) && finishInside;
          if (begin === finish && !(includeStart && includeEnd)) matches = false;
          const fullyInside =
            matches &&
            begin === 0 &&
            finish === upper - lower &&
            (!timeIncluded(lower) || includeStart) &&
            (!timeIncluded(upper) || includeEnd);
          allInside = allInside && fullyInside;
          if (!matches) continue;
          anyInside = true;
          writeInterval(
            pair,
            written++,
            beginTime,
            finishTime,
            interpolatePoint(firstPoint, firstVelocity, begin),
            interpolatePoint(firstPoint, firstVelocity, finish),
            FLAGS.temporalSupport |
              (begin === finish ? FLAGS.singleton : FLAGS.interval) |
              (fullyInside ? FLAGS.fullTemporalSupportInside : 0) |
              (includeStart ? 0 : FLAGS.startOpen) |
              (includeEnd ? 0 : FLAGS.endOpen)
          );
        }
      }
      writeRelationship(pair, anyInside, allInside, support, unknown);
    }
  } else {
    const vertices: Point[] = Array.from({length: vertexCount}, (_, vertex) => [
      parameters[7 + vertex * 2],
      parameters[8 + vertex * 2]
    ]);
    let ringStart = 0;
    const nextVertices = rings.flatMap(count => {
      const start = ringStart;
      ringStart += count;
      return Array.from({length: count}, (_, vertex) => start + ((vertex + 1) % count));
    });
    const inside = (point: Point, onBoundary = false): boolean => {
      if (!vertexCount)
        return closedGeometry
          ? point[0] >= minimumX &&
              point[0] <= maximumX &&
              point[1] >= minimumY &&
              point[1] <= maximumY
          : point[0] > minimumX &&
              point[0] < maximumX &&
              point[1] > minimumY &&
              point[1] < maximumY;
      if (onBoundary) return closedGeometry;
      let winding = 0;
      let boundaryHit = false;
      for (let edge = 0; edge < vertexCount; edge++) {
        const first = vertices[edge];
        const second = vertices[nextVertices[edge]];
        const straddles = first[1] >= point[1] !== second[1] >= point[1];
        const inBox =
          point[0] >= Math.min(first[0], second[0]) &&
          point[0] <= Math.max(first[0], second[0]) &&
          point[1] >= Math.min(first[1], second[1]) &&
          point[1] <= Math.max(first[1], second[1]);
        if (!straddles && !inBox) continue;
        const orientation = computeTriangulationOrientation(first, second, point);
        if (inBox && orientation === 0) boundaryHit = true;
        const side =
          orientation ||
          (second[0] !== first[0]
            ? -Math.sign(second[0] - first[0])
            : Math.sign(second[1] - first[1]));
        if (straddles && second[1] > first[1] && side > 0) winding++;
        if (straddles && second[1] <= first[1] && side < 0) winding--;
      }
      return boundaryHit ? closedGeometry : (winding & 1) !== 0;
    };
    const extended = Boolean(
      options.polygonRingVertexCounts ||
        options.timeBoundary ||
        options.geometryBoundary ||
        options.intervalsPerSegment ||
        options.compactCapacity !== undefined
    );
    for (let track = 0; track < trackCount; track++) {
      let anyInside = false;
      let allInside = true;
      let support = 0;
      let unknown = 0;
      for (let row = offsets[track]; row < offsets[track + 1]; row++) {
        const base = row * slots;
        const segment = loadSegment(row, offsets[track + 1], false);
        const {time, finish, duration, position, movement, gap} = segment;
        if (!validParameters || segment.invalid) {
          status[base] = FLAGS.invalid;
          unknown++;
          continue;
        }
        if (finish < minimumTime || time > maximumTime) continue;
        if (gap) {
          status[base] = FLAGS.gap;
          if (Math.max(minimumTime, time) < Math.min(maximumTime, finish)) unknown++;
          if (!timeIncluded(time)) continue;
          support++;
          status[base] |= FLAGS.temporalSupport;
          const contained = inside(position);
          anyInside ||= contained;
          allInside &&= contained;
          if (contained)
            writeInterval(
              row,
              0,
              time,
              time,
              position,
              position,
              FLAGS.singleton | FLAGS.fullTemporalSupportInside
            );
          continue;
        }
        let lower = duration > 0 ? Math.max(0, (minimumTime - time) / duration) : 0;
        let upper = duration > 0 ? Math.min(1, (maximumTime - time) / duration) : extended ? 0 : 1;
        if (
          extended &&
          (lower > upper ||
            ((lower === upper || duration === 0) && !timeIncluded(time + duration * lower)))
        )
          continue;
        support++;
        status[base] = FLAGS.temporalSupport;
        if (!extended) {
          const timeLower = lower;
          const timeUpper = upper;
          let intersects = true;
          for (let axis = 0; axis < 2; axis++) {
            const minimum = axis === 0 ? minimumX : minimumY;
            const maximum = axis === 0 ? maximumX : maximumY;
            if (movement[axis] === 0) {
              if (position[axis] < minimum || position[axis] > maximum) intersects = false;
            } else {
              const first = (minimum - position[axis]) / movement[axis];
              const second = (maximum - position[axis]) / movement[axis];
              lower = Math.max(lower, Math.min(first, second));
              upper = Math.min(upper, Math.max(first, second));
            }
          }
          intersects &&= lower <= upper;
          const fullyInside = intersects && lower === timeLower && upper === timeUpper;
          allInside &&= fullyInside;
          if (!intersects) continue;
          anyInside = true;
          writeInterval(
            row,
            0,
            time + duration * lower,
            time + duration * upper,
            interpolatePoint(position, movement, lower),
            interpolatePoint(position, movement, upper),
            (duration === 0 || lower === upper ? FLAGS.singleton : FLAGS.interval) |
              (fullyInside ? FLAGS.fullTemporalSupportInside : 0)
          );
          continue;
        }
        const boundaries = [lower, upper];
        const collinearRanges: Point[] = [];
        const lengthSquared = computeDot(movement, movement);
        if (vertexCount && lengthSquared > 0) {
          const secondPosition = getPosition(Math.min(row + 1, offsets[track + 1] - 1));
          for (let edge = 0; edge < vertexCount; edge++) {
            const firstVertex = vertices[edge];
            const secondVertex = vertices[nextVertices[edge]];
            if (firstVertex[0] === secondVertex[0] && firstVertex[1] === secondVertex[1]) continue;
            const firstSide = computeTriangulationOrientation(firstVertex, secondVertex, position);
            const secondSide = computeTriangulationOrientation(
              firstVertex,
              secondVertex,
              secondPosition
            );
            const firstEdgeSide = computeTriangulationOrientation(
              position,
              secondPosition,
              firstVertex
            );
            const secondEdgeSide = computeTriangulationOrientation(
              position,
              secondPosition,
              secondVertex
            );
            if (firstSide === 0 && secondSide === 0) {
              const begin =
                computeDot(subtractPoints(firstVertex, position), movement) / lengthSquared;
              const end =
                computeDot(subtractPoints(secondVertex, position), movement) / lengthSquared;
              if (begin > lower && begin < upper) boundaries.push(begin);
              if (end > lower && end < upper) boundaries.push(end);
              collinearRanges.push([Math.min(begin, end), Math.max(begin, end)]);
            } else if (firstSide * secondSide <= 0 && firstEdgeSide * secondEdgeSide <= 0) {
              let fraction =
                firstSide === 0
                  ? 0
                  : secondSide === 0
                    ? 1
                    : firstEdgeSide === 0
                      ? computeDot(subtractPoints(firstVertex, position), movement) / lengthSquared
                      : secondEdgeSide === 0
                        ? computeDot(subtractPoints(secondVertex, position), movement) /
                          lengthSquared
                        : computeCross(
                            subtractPoints(firstVertex, position),
                            subtractPoints(secondVertex, firstVertex)
                          ) / computeCross(movement, subtractPoints(secondVertex, firstVertex));
              fraction = Math.max(0, Math.min(1, fraction));
              if (fraction > lower && fraction < upper) boundaries.push(fraction);
            }
          }
        } else if (!vertexCount) {
          for (let axis = 0; axis < 2; axis++)
            if (movement[axis] !== 0)
              for (const side of [
                axis === 0 ? minimumX : minimumY,
                axis === 0 ? maximumX : maximumY
              ]) {
                const fraction = (side - position[axis]) / movement[axis];
                if (fraction > lower && fraction < upper) boundaries.push(fraction);
              }
        }
        const unique = [...new Set(boundaries)].sort((first, second) => first - second);
        const classify = (fraction: number, midpointRange?: Point): boolean => {
          const point =
            fraction <= 0
              ? position
              : fraction >= 1
                ? getPosition(Math.min(row + 1, offsets[track + 1] - 1))
                : interpolatePoint(position, movement, fraction);
          let onBoundary =
            vertexCount > 0 && !midpointRange && fraction > lower && fraction < upper;
          if (vertexCount > 0 && midpointRange) {
            const beginPoint = interpolatePoint(position, movement, midpointRange[0]);
            const endPoint = interpolatePoint(position, movement, midpointRange[1]);
            const betweenCrossings = midpointRange[0] > lower && midpointRange[1] < upper;
            onBoundary =
              (betweenCrossings &&
                point.every(
                  (coordinate, axis) =>
                    coordinate === beginPoint[axis] || coordinate === endPoint[axis]
                )) ||
              collinearRanges.some(range => fraction >= range[0] && fraction <= range[1]);
          }
          return inside(point, onBoundary);
        };
        let written = 0;
        let fullyInside = true;
        let previousInterior = false;
        for (let cursor = 0; cursor < unique.length; cursor++) {
          const begin = unique[cursor];
          const beginTime = time + duration * begin;
          const pointSupported = timeIncluded(beginTime);
          const pointInside = classify(begin);
          const hasNext = cursor + 1 < unique.length && duration > 0;
          const end = hasNext ? unique[cursor + 1] : begin;
          const nextInterior = hasNext && classify((begin + end) / 2, [begin, end]);
          if (hasNext) fullyInside &&= nextInterior;
          if (pointSupported) fullyInside &&= pointInside;
          if (nextInterior) {
            const includeEnd = timeIncluded(time + duration * end) && classify(end);
            writeInterval(
              row,
              written++,
              beginTime,
              time + duration * end,
              interpolatePoint(position, movement, begin),
              interpolatePoint(position, movement, end),
              FLAGS.interval |
                (pointSupported && pointInside ? 0 : FLAGS.startOpen) |
                (includeEnd ? 0 : FLAGS.endOpen)
            );
            anyInside = true;
          } else if (pointSupported && pointInside && !previousInterior) {
            const point = interpolatePoint(position, movement, begin);
            writeInterval(row, written++, beginTime, beginTime, point, point, FLAGS.singleton);
            anyInside = true;
          }
          previousInterior = nextInterior;
        }
        allInside &&= fullyInside;
        if (fullyInside) status[base] |= FLAGS.fullTemporalSupportInside;
      }
      writeRelationship(track, anyInside, allInside, support, unknown);
    }
  }
  const result: CPUTrajectoryRestrictionResult = {
    intervals: Float32Array.from(intervalValues),
    positions: Float32Array.from(positionValues),
    status,
    relationships
  };
  if (options.compactCapacity !== undefined) {
    const capacity = options.compactCapacity;
    const compact: IntervalRow[] = [];
    let track = 0;
    // The GPU compactor scans output slots in row order and advances its track cursor against
    // the original offsets. Preserve that order even when reversed empty ranges separate tracks.
    for (let source = 0; source < intervalCount; source++) {
      if ((status[source] & (FLAGS.interval | FLAGS.singleton)) === 0) continue;
      const sourceRow = Math.floor(source / slots);
      while (track + 1 < trackCount && sourceRow >= offsets[track + 1]) track++;
      const interval: IntervalRow = {
        begin: intervalValues[source * 2],
        end: intervalValues[source * 2 + 1],
        positions: Array.from(positionValues.subarray(source * 4, source * 4 + 4)),
        flags: status[source],
        source,
        firstSource: source,
        track
      };
      const previous = compact[compact.length - 1];
      const merge =
        options.mergeIntervals &&
        previous &&
        previous.track === interval.track &&
        (interval.begin < previous.end ||
          (interval.begin === previous.end &&
            ((previous.flags & FLAGS.endOpen) === 0 || (interval.flags & FLAGS.startOpen) === 0)));
      if (merge) {
        if (interval.end >= previous.end) {
          previous.end = interval.end;
          previous.flags = (previous.flags & ~FLAGS.endOpen) | (interval.flags & FLAGS.endOpen);
          previous.positions = [
            previous.positions[0],
            previous.positions[1],
            interval.positions[2],
            interval.positions[3]
          ];
          previous.source = interval.source;
        }
        previous.flags =
          (previous.flags & ~3) |
          (previous.begin === previous.end ? FLAGS.singleton : FLAGS.interval);
      } else compact.push({...interval});
    }
    result.compactIntervals = new Float32Array(capacity * 2).fill(Number.NaN);
    result.compactPositions = new Float32Array(capacity * 4).fill(Number.NaN);
    result.compactSources = new Uint32Array(capacity * 2).fill(0xffffffff);
    result.compactStatus = new Uint32Array(capacity);
    for (let row = 0; row < compact.length; row++) {
      const interval = compact[row];
      if (row >= capacity) continue;
      result.compactIntervals.set([interval.begin, interval.end], row * 2);
      result.compactPositions.set(interval.positions, row * 4);
      result.compactSources.set(
        [Math.floor(interval.firstSource / slots), Math.floor(interval.source / slots)],
        row * 2
      );
      result.compactStatus[row] = interval.flags;
    }
    const upstreamOverflow = status.reduce(
      (total, flags) => total + Number((flags & FLAGS.capacityExceeded) !== 0),
      0
    );
    result.compactSummary = Uint32Array.of(
      compact.length,
      Math.min(compact.length, capacity),
      Math.max(0, compact.length - capacity) + upstreamOverflow
    );
  }
  return result;
}

/** CPU graph twin; removes GPU scratch caps and retains every output layout and selected capacity. */
export class CPUTrajectoryRestrictionOperation extends GPUCPUOperation {
  constructor(props: TrajectoryRestrictionOperationProps) {
    const id = props.id ?? 'trajectory-restriction';
    const pairwise = props.pairs !== undefined || props.pairwiseTracks !== undefined;
    validateOptions(id, props, pairwise);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'trackCount', props.trackCount, 0);
    validateRequirement(
      id,
      !(props.pairs && props.pairwiseTracks),
      'takes pairs or pairwiseTracks'
    );
    if (props.pairs) validateCapacity(id, 'pairCapacity', props.pairCapacity!, 0);
    else validateRequirement(id, props.pairCapacity === undefined, 'pairCapacity requires pairs');
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        ['float32', 'uint32x2'].includes(props.timestampFormat),
      'timestampFormat must be float32 or uint32x2'
    );
    validateRequirement(
      id,
      (props.timestampFormat === 'uint32x2') === (props.timeOrigin !== undefined),
      'timeOrigin is required only with uint32x2 timestampFormat'
    );
    for (const pair of props.pairwiseTracks ?? [])
      validateRequirement(
        id,
        pair.length === 2 &&
          pair.every(track => Number.isInteger(track) && track >= 0 && track < props.trackCount),
        'pairwiseTracks must contain pairs of track indices below trackCount'
      );
    const compactPorts = [
      props.output.compactIntervals,
      props.output.compactPositions,
      props.output.compactSources,
      props.output.compactStatus,
      props.output.compactSummary
    ];
    validateRequirement(
      id,
      props.compactCapacity !== undefined
        ? compactPorts.every(value => value !== undefined)
        : compactPorts.every(value => value === undefined),
      'compact outputs must all be present exactly when compactCapacity is given'
    );
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: [props.timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
    const operation = new GPUTrajectoryRestrictionOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const positions = getCPUOperationFloat32Rows(id, inputs, 'sourcePositions', 'float32x2');
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions must match rowCount`);
        if (offsets.length !== props.trackCount + 1)
          throw new Error(`${id} trackOffsets must match trackCount`);
        const pairs = props.pairs
          ? {
              first: getCPUOperationUint32Rows(id, inputs, 'pairFirst'),
              second: getCPUOperationUint32Rows(id, inputs, 'pairSecond'),
              count:
                props.pairs.count !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'pairCount')
                  : undefined
            }
          : undefined;
        if (pairs && pairs.first.length !== props.pairCapacity)
          throw new Error(`${id} pairs.first must match pairCapacity`);
        const result = computeTrajectoryRestrictionOnCPU(
          {
            positions,
            trackOffsets: offsets,
            pairs,
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'times', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'times'),
            timeOrigin:
              props.timeOrigin !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'timeOrigin')
                : undefined,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectoryRestrictionResult]!
          ])
        );
      }
    });
  }
}
