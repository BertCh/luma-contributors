// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveGyrationCenter, type GyrationCenter} from './gyration-center';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getCPUTrajectoryDistance} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_MOVEMENT_SENTINEL} from '../trajectory-staypoints/movement-common';
import {
  GPUMobilityIndicatorsOperation,
  snapshotMobilityIndicatorsOperationProps,
  type MobilityIndicatorsOperationProps
} from './gpu-mobility-indicators-operation';

/** Packed tables. Counts clamp to table capacities, as on GPU. */
export type CPUMobilityIndicatorsInput = {
  timestamps: Float32Array | Uint32Array;
  staypoints: {
    tracks: Uint32Array;
    centroids: Float32Array;
    startRows: Uint32Array;
    durations: Float32Array;
    locationIds?: Uint32Array;
    count: Uint32Array;
    overflow?: Uint32Array;
    candidateOverflow?: Uint32Array;
    /** Optional loss of the location-label iteration; other indicators remain usable. */
    locationIterationLimitReached?: Uint32Array;
  };
  triplegs?: {
    tracks: Uint32Array;
    startRows: Uint32Array;
    endRows: Uint32Array;
    count: Uint32Array;
    staypointFinishRows: Uint32Array;
    overflow?: Uint32Array;
    candidateOverflow?: Uint32Array;
  };
};
/** Real entropy can be skipped when unused; the default computes every indicator. */
export type CPUMobilityIndicatorsOptions = {
  trackCount: number;
  spatialContext?: GPUSpatialContext;
  gyrationCenter?: GyrationCenter;
  includeRealEntropy?: boolean;
  /**
   * Optional cap on usable location keys per track for `realEntropy`: an over-cap track
   * gets NaN and sets `candidateOverflow`. Omitted, the full sequence is used.
   */
  maximumEntropySequenceRows?: number;
};
/** Same output layouts, including all unused capacity rows and scalar candidateOverflow. */
export type CPUMobilityIndicatorsResult = {
  radiusOfGyration: Float32Array;
  jumpLengths: Float32Array;
  waitingTimes: Float32Array;
  randomEntropy: Float32Array;
  uncorrelatedEntropy: Float32Array;
  realEntropy: Float32Array;
  homeLocations: Uint32Array;
  workLocations: Uint32Array;
  trackingQuality: Float32Array;
  candidateOverflow: Uint32Array;
  excludedLocationCount: Uint32Array;
};

/** Nonoverlapping longest-previous matches in O(n^2) time and O(n) scratch, without a sequence cap. */
function computeSequenceEntropy(sequence: readonly number[]): number {
  const length = sequence.length;
  if (length < 2) return 0;
  let nextMatches = new Uint32Array(length + 1);
  let matches = new Uint32Array(length + 1);
  let sumLambda = 3;
  for (let position = length - 1; position >= 1; position--) {
    let longestMatch = 0;
    for (let candidate = 0; candidate < position; candidate++) {
      matches[candidate] =
        sequence[candidate] === sequence[position] ? 1 + nextMatches[candidate + 1] : 0;
      longestMatch = Math.max(longestMatch, Math.min(matches[candidate], position - candidate));
    }
    if (position + 1 < length) {
      const symbolLength =
        longestMatch + 1 >= length - position ? length - position + 1 : longestMatch + 1;
      sumLambda += symbolLength;
    }
    [matches, nextMatches] = [nextMatches, matches];
  }
  return (length * Math.log2(length)) / sumLambda;
}

/**
 * Computes staypoint mobility measures with float64 distances, maps for location shares/durations
 * and an uncapped Lempel-Ziv estimator. Radius of gyration defaults to the spherical mean in
 * great-circle contexts and coordinate mean in planar contexts; explicit coordinate-mean retains
 * arithmetic longitude/latitude centering. An indeterminate spherical center yields NaN.
 * Missing and unsettled location keys are excluded from location indicators. Tracking quality
 * uses positive endpoint time differences, not the staypoint duration column; overlaps are summed.
 *
 * What differs from the GPU contributor:
 * - Arithmetic uses float64 and rounds each output once. Int64 timestamps are subtracted exactly
 *   before conversion. Distances, entropies and duration-ranking ties near rounding can differ.
 * - Real entropy uses the full sequence unless the maximumEntropySequenceRows option is given;
 *   then, as on the GPU, a longer track gets NaN and sets candidateOverflow. The operation twin
 *   passes the cap only when its props set one, so the GPU default cap of 1024 is not applied.
 *   candidateOverflow also reports supplied staypoint loss. includeRealEntropy=false skips that
 *   unused calculation and ignores its cap. An explicit cap is validated only when entropy is
 *   requested; the CPU operation validates that prop at construction.
 * - Inconsistent columns, invalid row indices and unordered active track IDs throw an Error.
 * - Nonfinite coordinates/durations/times use JavaScript arithmetic and comparisons; GPU results
 *   for such inputs are unspecified.
 *
 * @throws Error naming an inconsistent timestamps, staypoints or triplegs argument.
 */
export function computeMobilityIndicatorsOnCPU(
  input: CPUMobilityIndicatorsInput,
  options: CPUMobilityIndicatorsOptions
): CPUMobilityIndicatorsResult {
  const id = 'mobility-indicators';
  const gyrationCenter = resolveGyrationCenter(id, options.gyrationCenter, options.spatialContext);
  const {trackCount} = options;
  validateCapacity(id, 'trackCount', trackCount);
  const {maximumEntropySequenceRows} = options;
  if (options.includeRealEntropy !== false && maximumEntropySequenceRows !== undefined)
    validateCapacity(id, 'maximumEntropySequenceRows', maximumEntropySequenceRows);
  const {timestamps, staypoints, triplegs} = input;
  const isWordTime = timestamps instanceof Uint32Array;
  const rowCount = timestamps.length / (isWordTime ? 2 : 1);
  if (!Number.isInteger(rowCount))
    throw new Error('timestamps must contain complete uint32x2 rows');
  const getTimestampWords = (row: number) =>
    BigInt.asIntN(64, (BigInt(timestamps[row * 2 + 1]) << 32n) | BigInt(timestamps[row * 2]));
  const getTimeDifference = (later: number, earlier: number) =>
    isWordTime
      ? Number(getTimestampWords(later) - getTimestampWords(earlier))
      : timestamps[later] - timestamps[earlier];
  const capacity = staypoints.tracks.length;
  if (staypoints.centroids.length !== capacity * 2)
    throw new Error('staypoints.centroids length must match staypoints.tracks');
  for (const column of ['startRows', 'durations'] as const)
    if (staypoints[column].length !== capacity)
      throw new Error(`staypoints.${column} length must match staypoints.tracks`);
  if (staypoints.locationIds && staypoints.locationIds.length !== capacity)
    throw new Error('staypoints.locationIds length must match staypoints.tracks');
  if (staypoints.count.length < 1)
    throw new Error('staypoints.count must contain at least one row');
  if (triplegs) {
    if (
      triplegs.startRows.length !== triplegs.tracks.length ||
      triplegs.endRows.length !== triplegs.tracks.length ||
      triplegs.staypointFinishRows.length !== capacity
    )
      throw new Error('mobility-indicators triplegs fixed shapes must match table capacities');
    if (triplegs.count.length < 1) throw new Error('triplegs.count must contain at least one row');
  }
  for (const [tableName, table] of [
    ['staypoints', staypoints],
    ['triplegs', triplegs]
  ] as const) {
    for (const name of ['overflow', 'candidateOverflow'] as const) {
      const flag = table?.[name];
      if (flag && flag.length < 1)
        throw new Error(`mobility-indicators ${tableName}.${name} must contain at least one row`);
    }
  }
  const locationLoss = staypoints.locationIterationLimitReached;
  validateRequirement(
    id,
    !locationLoss || locationLoss.length >= 1,
    'staypoints.locationIterationLimitReached must contain at least one row'
  );
  const settledIds = locationLoss?.[0] ? undefined : staypoints.locationIds;
  const activeStaypoints = Math.min(staypoints.count[0], capacity);
  const result: CPUMobilityIndicatorsResult = {
    radiusOfGyration: new Float32Array(trackCount),
    jumpLengths: new Float32Array(capacity),
    waitingTimes: new Float32Array(capacity),
    randomEntropy: new Float32Array(trackCount),
    uncorrelatedEntropy: new Float32Array(trackCount),
    realEntropy: new Float32Array(trackCount),
    homeLocations: new Uint32Array(trackCount).fill(GPU_MOVEMENT_SENTINEL),
    workLocations: new Uint32Array(trackCount).fill(GPU_MOVEMENT_SENTINEL),
    trackingQuality: new Float32Array(trackCount),
    candidateOverflow: new Uint32Array(1),
    excludedLocationCount: Uint32Array.of(settledIds ? 0 : activeStaypoints)
  };
  if (staypoints.overflow?.[0] || staypoints.candidateOverflow?.[0]) {
    for (const values of Object.values(result))
      if (values instanceof Float32Array) values.fill(NaN);
    result.candidateOverflow[0] = 1;
    // The early return skips the per-track pass, but still reports key loss in the active prefix.
    if (settledIds) {
      let excludedLocationCount = 0;
      for (let row = 0; row < activeStaypoints; row++)
        if (settledIds[row] === GPU_MOVEMENT_SENTINEL) excludedLocationCount++;
      result.excludedLocationCount[0] = excludedLocationCount;
    }
    return result;
  }
  const triplegLoss = Boolean(triplegs?.overflow?.[0] || triplegs?.candidateOverflow?.[0]);
  const getActiveCount = (
    table: {tracks: Uint32Array; count: Uint32Array},
    name: string
  ): number => {
    if (table.count.length < 1) throw new Error(`${name}.count must contain at least one row`);
    const total = Math.min(table.count[0], table.tracks.length);
    for (let row = 0; row < total; row++)
      if (table.tracks[row] >= trackCount || (row > 0 && table.tracks[row] < table.tracks[row - 1]))
        throw new Error(`${name}.tracks must contain sorted valid track indices`);
    return total;
  };
  const total = getActiveCount(staypoints, 'staypoints');
  for (let stay = 0; stay < total; stay++)
    if (staypoints.startRows[stay] >= rowCount)
      throw new Error('staypoints.startRows must index timestamps');
  let triplegTotal = 0;
  if (triplegs) {
    const triplegCapacity = triplegs.tracks.length;
    if (triplegs.startRows.length !== triplegCapacity)
      throw new Error('triplegs.startRows length must match triplegs.tracks');
    if (triplegs.endRows.length !== triplegCapacity)
      throw new Error('triplegs.endRows length must match triplegs.tracks');
    if (triplegs.staypointFinishRows.length !== capacity)
      throw new Error('triplegs.staypointFinishRows length must match staypoints.tracks');
    triplegTotal = triplegLoss ? 0 : getActiveCount(triplegs, 'triplegs');
    for (let stay = 0; !triplegLoss && stay < total; stay++)
      if (triplegs.staypointFinishRows[stay] >= rowCount)
        throw new Error('triplegs.staypointFinishRows must index timestamps');
    for (let tripleg = 0; tripleg < triplegTotal; tripleg++) {
      if (triplegs.startRows[tripleg] >= rowCount)
        throw new Error('triplegs.startRows must index timestamps');
      if (
        triplegs.endRows[tripleg] === 0 ||
        triplegs.endRows[tripleg] > rowCount ||
        triplegs.endRows[tripleg] <= triplegs.startRows[tripleg]
      )
        throw new Error('triplegs.endRows must terminate a nonempty range within timestamps');
    }
  }
  const context = resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  validateRequirement(
    id,
    gyrationCenter !== 'spherical-mean' || radius !== undefined,
    'gyrationCenter spherical-mean requires a great-circle spatialContext; use coordinate-mean for planar positions'
  );
  let begin = 0;
  let triplegBegin = 0;
  for (let track = 0; track < trackCount; track++) {
    let end = begin;
    while (end < total && staypoints.tracks[end] === track) end++;
    let triplegEnd = triplegBegin;
    while (triplegs && triplegEnd < triplegTotal && triplegs.tracks[triplegEnd] === track)
      triplegEnd++;
    if (end > begin) {
      const anchorLongitude = staypoints.centroids[begin * 2];
      const anchorLatitude = staypoints.centroids[begin * 2 + 1];
      let longitudeSum = 0;
      let latitudeSum = 0;
      let validCount = 0;
      const sequence: number[] = [];
      const locations = new Map<number, {count: number; duration: number}>();
      for (let row = begin; row < end; row++) {
        longitudeSum += staypoints.centroids[row * 2] - anchorLongitude;
        latitudeSum += staypoints.centroids[row * 2 + 1] - anchorLatitude;
        if (row > begin) {
          result.jumpLengths[row] = getCPUTrajectoryDistance(
            staypoints.centroids,
            row - 1,
            row,
            radius
          );
          result.waitingTimes[row] = getTimeDifference(
            staypoints.startRows[row],
            staypoints.startRows[row - 1]
          );
        }
        const location = settledIds?.[row];
        if (location === undefined || location === GPU_MOVEMENT_SENTINEL) continue;
        validCount++;
        if (
          options.includeRealEntropy !== false &&
          (maximumEntropySequenceRows === undefined || validCount <= maximumEntropySequenceRows)
        )
          sequence.push(location);
        const statistics = locations.get(location) ?? {count: 0, duration: 0};
        statistics.count++;
        statistics.duration += staypoints.durations[row];
        locations.set(location, statistics);
      }
      if (settledIds) result.excludedLocationCount[0] += end - begin - validCount;
      let meanLongitude = anchorLongitude + longitudeSum / (end - begin);
      let meanLatitude = anchorLatitude + latitudeSum / (end - begin);
      if (gyrationCenter === 'spherical-mean') {
        let unitXSum = 0;
        let unitYSum = 0;
        let unitZSum = 0;
        let stationary = true;
        for (let row = begin; row < end; row++) {
          stationary =
            stationary &&
            staypoints.centroids[row * 2] === anchorLongitude &&
            staypoints.centroids[row * 2 + 1] === anchorLatitude;
          const longitude = (staypoints.centroids[row * 2] * Math.PI) / 180;
          const latitude = (staypoints.centroids[row * 2 + 1] * Math.PI) / 180;
          unitXSum += Math.cos(latitude) * Math.cos(longitude);
          unitYSum += Math.cos(latitude) * Math.sin(longitude);
          unitZSum += Math.sin(latitude);
        }
        // A mean unit-vector norm <= 1e-6 has no defined center in either backend.
        const undefinedCenter = Math.hypot(unitXSum, unitYSum, unitZSum) <= (end - begin) * 1e-6;
        meanLongitude = undefinedCenter
          ? NaN
          : stationary
            ? anchorLongitude
            : (Math.atan2(unitYSum, unitXSum) * 180) / Math.PI;
        meanLatitude = undefinedCenter
          ? NaN
          : stationary
            ? anchorLatitude
            : (Math.atan2(unitZSum, Math.hypot(unitXSum, unitYSum)) * 180) / Math.PI;
      }
      let squares = 0;
      for (let row = begin; row < end; row++) {
        // A float64 center must not be rounded into an intermediate Float32Array.
        const pair = [
          staypoints.centroids[row * 2],
          staypoints.centroids[row * 2 + 1],
          meanLongitude,
          meanLatitude
        ];
        const separation = getCPUTrajectoryDistance(pair, 0, 1, radius);
        squares += separation * separation;
      }
      result.radiusOfGyration[track] = Math.sqrt(squares / (end - begin));
      result.randomEntropy[track] = validCount ? Math.log2(locations.size) : NaN;
      let entropy = 0;
      let bestDuration = -1;
      let bestLocation = GPU_MOVEMENT_SENTINEL;
      let secondDuration = -1;
      let secondLocation = GPU_MOVEMENT_SENTINEL;
      for (const [location, statistics] of locations) {
        const share = statistics.count / validCount;
        entropy -= share * Math.log2(share);
        const duration = statistics.duration;
        if (duration > bestDuration || (duration === bestDuration && location < bestLocation)) {
          secondDuration = bestDuration;
          secondLocation = bestLocation;
          bestDuration = duration;
          bestLocation = location;
        } else if (
          duration > secondDuration ||
          (duration === secondDuration && location < secondLocation)
        ) {
          secondDuration = duration;
          secondLocation = location;
        }
      }
      result.uncorrelatedEntropy[track] = validCount ? entropy : NaN;
      result.homeLocations[track] = bestLocation;
      result.workLocations[track] = secondLocation;
      if (options.includeRealEntropy !== false) {
        if (validCount === 0) {
          result.realEntropy[track] = NaN;
        } else if (
          maximumEntropySequenceRows !== undefined &&
          validCount > maximumEntropySequenceRows
        ) {
          // Not computed: NaN, never the valid entropy 0, and the candidate flag.
          result.realEntropy[track] = NaN;
          result.candidateOverflow[0] = 1;
        } else {
          result.realEntropy[track] = computeSequenceEntropy(sequence);
        }
      }
    }
    if (triplegs && !triplegLoss) {
      let tracked = 0;
      let firstStart: number | undefined;
      let lastFinish: number | undefined;
      const includeInterval = (start: number, finish: number) => {
        const duration = getTimeDifference(finish, start);
        if (duration <= 0) return;
        tracked += duration;
        if (firstStart === undefined || getTimeDifference(start, firstStart) < 0)
          firstStart = start;
        if (lastFinish === undefined || getTimeDifference(finish, lastFinish) > 0)
          lastFinish = finish;
      };
      for (let stay = begin; stay < end; stay++)
        includeInterval(staypoints.startRows[stay], triplegs.staypointFinishRows[stay]);
      for (let tripleg = triplegBegin; tripleg < triplegEnd; tripleg++)
        includeInterval(triplegs.startRows[tripleg], triplegs.endRows[tripleg] - 1);
      if (firstStart !== undefined && lastFinish !== undefined) {
        const extent = getTimeDifference(lastFinish, firstStart);
        if (extent > 0) result.trackingQuality[track] = tracked / extent;
      }
    }
    begin = end;
    triplegBegin = triplegEnd;
  }
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const flag = triplegs?.[name];
    if (flag && flag.length < 1)
      throw new Error(`mobility-indicators triplegs.${name} must contain at least one row`);
    if (flag?.[0]) result.trackingQuality.fill(NaN);
    const stayFlag = staypoints[name];
    if (stayFlag && stayFlag.length < 1)
      throw new Error(`mobility-indicators staypoints.${name} must contain at least one row`);
    if (stayFlag?.[0]) {
      for (const values of Object.values(result)) {
        if (values instanceof Float32Array) values.fill(NaN);
      }
      result.homeLocations.fill(GPU_MOVEMENT_SENTINEL);
      result.workLocations.fill(GPU_MOVEMENT_SENTINEL);
      result.candidateOverflow[0] = 1;
    }
  }
  if (locationLoss?.[0]) {
    result.randomEntropy.fill(NaN);
    result.uncorrelatedEntropy.fill(NaN);
    result.realEntropy.fill(NaN);
    result.homeLocations.fill(GPU_MOVEMENT_SENTINEL);
    result.workLocations.fill(GPU_MOVEMENT_SENTINEL);
    result.candidateOverflow[0] = 1;
  }
  return result;
}

/** CPU mobility twin; full sequence entropy replaces the GPU's bounded estimator. */
export class CPUMobilityIndicatorsOperation extends GPUCPUOperation {
  constructor(props: MobilityIndicatorsOperationProps) {
    props = snapshotMobilityIndicatorsOperationProps(props);
    const id = props.id ?? 'mobility-indicators';
    resolveGyrationCenter(id, props.gyrationCenter, props.spatialContext);
    validateCapacity(id, 'trackCount', props.trackCount);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'staypointCapacity', props.staypointCapacity, 0);
    if (props.output.realEntropy !== undefined && props.maximumEntropySequenceRows !== undefined)
      validateCapacity(id, 'maximumEntropySequenceRows', props.maximumEntropySequenceRows);
    if (props.triplegs) validateCapacity(id, 'triplegCapacity', props.triplegCapacity!, 0);
    validateRequirement(
      id,
      Object.entries(props.output).some(
        ([port, value]) => port !== 'status' && value !== undefined
      ),
      'requires at least one output'
    );
    validateRequirement(
      id,
      props.output.trackingQuality === undefined || Boolean(props.triplegs),
      'trackingQuality requires triplegs'
    );
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        ['float32', 'uint32x2'].includes(props.timestampFormat),
      'timestampFormat must be float32 or uint32x2'
    );
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const operation = new GPUMobilityIndicatorsOperation(props);
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
        const timestamps =
          props.timestampFormat === 'uint32x2'
            ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
            : getCPUOperationFloat32Rows(id, inputs, 'timestamps');
        if (timestamps.length !== props.rowCount * (props.timestampFormat === 'uint32x2' ? 2 : 1))
          throw new Error(`${id} timestamps length must match rowCount`);
        const tracks = getCPUOperationUint32Rows(id, inputs, 'stayTracks');
        if (tracks.length !== props.staypointCapacity)
          throw new Error(`${id} staypoints.tracks length must match staypointCapacity`);
        const triplegTracks = props.triplegs
          ? getCPUOperationUint32Rows(id, inputs, 'triplegTracks')
          : undefined;
        if (triplegTracks && triplegTracks.length !== props.triplegCapacity)
          throw new Error(`${id} triplegs.tracks length must match triplegCapacity`);
        const result = computeMobilityIndicatorsOnCPU(
          {
            timestamps,
            staypoints: {
              tracks,
              centroids: getCPUOperationFloat32Rows(id, inputs, 'centroids', 'float32x2'),
              startRows: getCPUOperationUint32Rows(id, inputs, 'stayStartRows'),
              durations: getCPUOperationFloat32Rows(id, inputs, 'durations'),
              locationIds:
                props.staypoints.locationIds !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'locationIds')
                  : undefined,
              count: getCPUOperationUint32Rows(id, inputs, 'stayCount'),
              locationIterationLimitReached:
                props.staypoints.status.locationIterationLimitReached !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'locationIterationLimitReached')
                  : undefined,
              overflow:
                props.staypoints.status.overflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'stayOverflow')
                  : undefined,
              candidateOverflow:
                props.staypoints.status.candidateOverflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'stayCandidateOverflow')
                  : undefined
            },
            triplegs: triplegTracks
              ? {
                  tracks: triplegTracks,
                  startRows: getCPUOperationUint32Rows(id, inputs, 'triplegStartRows'),
                  endRows: getCPUOperationUint32Rows(id, inputs, 'triplegEndRows'),
                  count: getCPUOperationUint32Rows(id, inputs, 'triplegCount'),
                  overflow:
                    props.triplegs?.status.overflow !== undefined
                      ? getCPUOperationUint32Rows(id, inputs, 'triplegOverflow')
                      : undefined,
                  candidateOverflow:
                    props.triplegs?.status.candidateOverflow !== undefined
                      ? getCPUOperationUint32Rows(id, inputs, 'triplegCandidateOverflow')
                      : undefined,
                  staypointFinishRows: getCPUOperationUint32Rows(id, inputs, 'stayFinishRows')
                }
              : undefined
          },
          {
            trackCount: props.trackCount,
            spatialContext: props.spatialContext,
            gyrationCenter: props.gyrationCenter,
            includeRealEntropy: props.output.realEntropy !== undefined,
            maximumEntropySequenceRows: props.maximumEntropySequenceRows
          }
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUMobilityIndicatorsResult]
          ])
        );
      }
    });
  }
}
