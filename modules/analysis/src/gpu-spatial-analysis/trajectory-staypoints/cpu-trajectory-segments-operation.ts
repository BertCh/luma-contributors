// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryRows,
  type CPUTrajectoryRows
} from '../../analysis-core/operations/cpu-trajectory-rows';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {GPU_MOVEMENT_SENTINEL} from './movement-common';
import {GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH} from './gpu-trajectory-segments';
import {
  GPUTrajectorySegmentsOperation,
  snapshotTrajectorySegmentsOperationProps,
  type TrajectorySegmentsOperationProps
} from './gpu-trajectory-segments-operation';

/** Packed input; staypoint rows must be ordered by track and then start row. */
export type CPUTrajectorySegmentsInput = CPUTrajectoryRows & {
  staypointIds: Uint32Array;
  parameters: Float32Array;
  staypoints?: {
    tracks: Uint32Array;
    startRows: Uint32Array;
    finishRows: Uint32Array;
    durations: Float32Array;
    count: Uint32Array;
    overflow?: Uint32Array;
    candidateOverflow?: Uint32Array;
  };
};

/** A trip capacity requests trip construction from the capacity-clamped tripleg table. */
export type CPUTrajectorySegmentsOptions = {triplegCapacity: number; tripCapacity?: number};

/** Flat port names used by the operation, all integer rows including status. */
export type CPUTrajectorySegmentsResult = {
  triplegIds: Uint32Array;
  triplegStartRows: Uint32Array;
  triplegEndRows: Uint32Array;
  rowTriplegIds: Uint32Array;
  triplegCount: Uint32Array;
  triplegRequiredCount: Uint32Array;
  triplegOverflow: Uint32Array;
  triplegCandidateOverflow: Uint32Array;
  tripIds: Uint32Array;
  tripStartRows: Uint32Array;
  tripFinishRows: Uint32Array;
  originStaypoints: Uint32Array;
  destinationStaypoints: Uint32Array;
  triplegTripIds: Uint32Array;
  tripCount: Uint32Array;
  tripRequiredCount: Uint32Array;
  tripOverflow: Uint32Array;
  tripCandidateOverflow: Uint32Array;
};

type SegmentRecord = {track: number; start: number; finish: number};
type TripEvent = SegmentRecord & {identifier: number; isActivity: boolean; isTripleg: boolean};

/**
 * Constructs moving triplegs and the trackintel activity/gap event state machine. Trips consume
 * only the capacity-clamped tripleg table, as on GPU, so upstream tripleg overflow is kept separate
 * from trip overflow. Activity and gap comparisons are strict; staypoints win equal-start ties.
 *
 * What differs from the GPU contributor:
 * - Time differences use float64, with exact Int64 subtraction before conversion. Decisions at
 *   float32 rounding boundaries can differ.
 * - Decreasing clamped trackOffsets and inconsistent or unordered staypoint columns throw an Error.
 * - Nonfinite coordinates compare with JavaScript rules (NaN differs from itself); nonfinite
 *   timestamps use JavaScript comparisons. GPU results for these inputs are unspecified.
 *
 * @throws Error naming an inconsistent input array or missing staypoints for trip construction.
 */
export function computeTrajectorySegmentsOnCPU(
  input: CPUTrajectorySegmentsInput,
  options: CPUTrajectorySegmentsOptions
): CPUTrajectorySegmentsResult {
  const id = 'trajectory-segments';
  validateCapacity(id, 'triplegCapacity', options.triplegCapacity);
  if (options.tripCapacity !== undefined)
    validateCapacity(id, 'tripCapacity', options.tripCapacity);
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  if (input.staypointIds.length !== rowCount)
    throw new Error('staypointIds length must match positions');
  if (input.parameters.length < GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH)
    throw new Error('parameters must hold the trajectory segments parameter layout');
  if (options.tripCapacity !== undefined && !input.staypoints)
    throw new Error('staypoints are required for trips');
  const [maximumGap, activityMinimumDuration] = input.parameters;
  const trackCount = trackOffsets.length - 1;
  const staypoints = input.staypoints;
  const upstreamLoss = Number(
    Boolean(staypoints?.overflow?.[0] || staypoints?.candidateOverflow?.[0])
  );
  let stayCount = 0;
  if (staypoints) {
    for (const field of ['overflow', 'candidateOverflow'] as const) {
      if (staypoints[field] && staypoints[field]!.length === 0)
        throw new Error(`staypoints.${field} must contain one row`);
    }
    const capacity = staypoints.tracks.length;
    for (const column of ['startRows', 'finishRows', 'durations'] as const) {
      if (staypoints[column].length !== capacity)
        throw new Error(`staypoints.${column} length must match staypoints.tracks`);
    }
    if (staypoints.count.length < 1) throw new Error('staypoints.count must contain one row');
    stayCount = Math.min(staypoints.count[0], capacity);
    for (let stay = 0; stay < stayCount; stay++) {
      const track = staypoints.tracks[stay];
      const start = staypoints.startRows[stay];
      const finish = staypoints.finishRows[stay];
      if (track >= trackCount || (stay > 0 && track < staypoints.tracks[stay - 1]))
        throw new Error('staypoints.tracks must be sorted valid track indices');
      if (
        start < trackOffsets[track] ||
        start >= trackOffsets[track + 1] ||
        (stay > 0 &&
          track === staypoints.tracks[stay - 1] &&
          start < staypoints.startRows[stay - 1])
      )
        throw new Error('staypoints.startRows must be ordered within each track');
      if (finish < start || finish >= trackOffsets[track + 1])
        throw new Error('staypoints.finishRows must be within the track and at or after startRows');
    }
  }
  const createMissingRows = (length: number) => new Uint32Array(length).fill(GPU_MOVEMENT_SENTINEL);
  const tripCapacity = options.tripCapacity ?? 0;
  const result: CPUTrajectorySegmentsResult = {
    triplegIds: createMissingRows(options.triplegCapacity),
    triplegStartRows: createMissingRows(options.triplegCapacity),
    triplegEndRows: createMissingRows(options.triplegCapacity),
    rowTriplegIds: createMissingRows(rowCount),
    triplegCount: new Uint32Array(1),
    triplegRequiredCount: new Uint32Array(1),
    triplegOverflow: new Uint32Array(1),
    triplegCandidateOverflow: Uint32Array.of(upstreamLoss),
    tripIds: createMissingRows(tripCapacity),
    tripStartRows: createMissingRows(tripCapacity),
    tripFinishRows: createMissingRows(tripCapacity),
    originStaypoints: createMissingRows(tripCapacity),
    destinationStaypoints: createMissingRows(tripCapacity),
    triplegTripIds: createMissingRows(options.triplegCapacity),
    tripCount: new Uint32Array(1),
    tripRequiredCount: new Uint32Array(1),
    tripOverflow: new Uint32Array(1),
    tripCandidateOverflow: Uint32Array.of(upstreamLoss)
  };
  const triplegs: SegmentRecord[] = [];
  for (let track = 0; track < trackCount; track++) {
    const end = trackOffsets[track + 1];
    let start = trackOffsets[track];
    while (start < end) {
      if (input.staypointIds[start] !== GPU_MOVEMENT_SENTINEL) {
        start++;
        continue;
      }
      let next = start + 1;
      while (
        next < end &&
        input.staypointIds[next] === GPU_MOVEMENT_SENTINEL &&
        !(maximumGap > 0 && getTimeDifference(next, next - 1) > maximumGap)
      )
        next++;
      let moves = false;
      for (let row = start + 1; row < next; row++) {
        if (
          input.positions[row * 2] !== input.positions[start * 2] ||
          input.positions[row * 2 + 1] !== input.positions[start * 2 + 1]
        ) {
          moves = true;
          break;
        }
      }
      if (next - start >= 2 && moves) triplegs.push({track, start, finish: next - 1});
      start = next;
    }
  }
  const keptTriplegs = triplegs.slice(0, options.triplegCapacity);
  keptTriplegs.forEach((tripleg, identifier) => {
    result.triplegIds[identifier] = tripleg.track;
    result.triplegStartRows[identifier] = tripleg.start;
    result.triplegEndRows[identifier] = tripleg.finish + 1;
    result.rowTriplegIds.fill(identifier, tripleg.start, tripleg.finish + 1);
  });
  result.triplegCount[0] = keptTriplegs.length;
  result.triplegRequiredCount[0] = triplegs.length;
  result.triplegOverflow[0] = Number(triplegs.length > options.triplegCapacity);
  result.tripCandidateOverflow[0] = Number(Boolean(upstreamLoss || result.triplegOverflow[0]));
  if (options.tripCapacity === undefined) return result;
  const eventsByTrack: TripEvent[][] = Array.from({length: trackCount}, () => []);
  for (let stay = 0; stay < stayCount; stay++) {
    eventsByTrack[staypoints!.tracks[stay]].push({
      track: staypoints!.tracks[stay],
      start: staypoints!.startRows[stay],
      finish: staypoints!.finishRows[stay],
      identifier: stay,
      isActivity: staypoints!.durations[stay] > activityMinimumDuration,
      isTripleg: false
    });
  }
  keptTriplegs.forEach((tripleg, identifier) =>
    eventsByTrack[tripleg.track].push({...tripleg, identifier, isActivity: false, isTripleg: true})
  );
  let tripTotal = 0;
  for (let track = 0; track < trackCount; track++) {
    const events = eventsByTrack[track];
    events.sort(
      (first, second) =>
        first.start - second.start || Number(first.isTripleg) - Number(second.isTripleg)
    );
    let hasTripleg = false;
    let nonActivityCount = 0;
    let firstStart = GPU_MOVEMENT_SENTINEL;
    let lastFinish = GPU_MOVEMENT_SENTINEL;
    let origin = GPU_MOVEMENT_SENTINEL;
    let destination = GPU_MOVEMENT_SENTINEL;
    let openDestination = false;
    const emitTrip = () => {
      if (!hasTripleg) return;
      if (tripTotal < tripCapacity) {
        result.tripIds[tripTotal] = track;
        result.tripStartRows[tripTotal] = firstStart;
        result.tripFinishRows[tripTotal] = lastFinish;
        result.originStaypoints[tripTotal] = origin;
        result.destinationStaypoints[tripTotal] = destination;
      }
      tripTotal++;
    };
    events.forEach((event, eventIndex) => {
      const previous = events[eventIndex - 1];
      const next = events[eventIndex + 1];
      const isGapBefore =
        previous !== undefined &&
        maximumGap > 0 &&
        getTimeDifference(event.start, previous.finish) > maximumGap;
      const startsTrip = eventIndex === 0 || isGapBefore || (event.isActivity && !next?.isActivity);
      if (openDestination) {
        openDestination = false;
        destination = event.isActivity && !isGapBefore ? event.identifier : GPU_MOVEMENT_SENTINEL;
      }
      if (startsTrip && eventIndex > 0) {
        emitTrip();
        hasTripleg = false;
        nonActivityCount = 0;
        destination = GPU_MOVEMENT_SENTINEL;
        origin = GPU_MOVEMENT_SENTINEL;
      }
      if (!event.isActivity) {
        if (nonActivityCount === 0) {
          firstStart = event.start;
          origin =
            previous?.isActivity && !isGapBefore ? previous.identifier : GPU_MOVEMENT_SENTINEL;
        }
        nonActivityCount++;
        lastFinish = event.finish;
        if (event.isTripleg) hasTripleg = true;
        openDestination = true;
      }
    });
    if (openDestination) destination = GPU_MOVEMENT_SENTINEL;
    emitTrip();
  }
  const keptTripCount = Math.min(tripTotal, tripCapacity);
  result.tripCount[0] = keptTripCount;
  result.tripRequiredCount[0] = tripTotal;
  result.tripOverflow[0] = Number(tripTotal > tripCapacity);
  keptTriplegs.forEach((tripleg, identifier) => {
    let candidate = -1;
    for (let trip = 0; trip < keptTripCount; trip++) {
      if (result.tripStartRows[trip] <= tripleg.start) candidate = trip;
      else break;
    }
    if (candidate >= 0 && tripleg.start <= result.tripFinishRows[candidate])
      result.triplegTripIds[identifier] = candidate;
  });
  return result;
}

/** CPU twin, retaining tripleg capacity as the input-table bound of downstream trip assembly. */
export class CPUTrajectorySegmentsOperation extends GPUCPUOperation {
  constructor(props: TrajectorySegmentsOperationProps) {
    props = snapshotTrajectorySegmentsOperationProps(props);
    const id = props.id ?? 'trajectory-segments';
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'triplegCapacity', props.triplegCapacity);
    if (props.output.trips) {
      validateCapacity(id, 'tripCapacity', props.tripCapacity!);
      validateRequirement(id, Boolean(props.staypoints), 'staypoints are required for trips');
    }
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        ['float32', 'uint32x2'].includes(props.timestampFormat),
      'timestampFormat must be float32 or uint32x2'
    );
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: [props.timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
    const operation = new GPUTrajectorySegmentsOperation(props);
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must match rowCount`);
        const result = computeTrajectorySegmentsOnCPU(
          {
            positions,
            storage: 'resident-capacity',
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets: getCPUOperationUint32Rows(id, inputs, 'trackOffsets'),
            staypointIds: getCPUOperationUint32Rows(id, inputs, 'staypointIds'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            staypoints: props.staypoints
              ? {
                  tracks: getCPUOperationUint32Rows(id, inputs, 'stayTracks'),
                  startRows: getCPUOperationUint32Rows(id, inputs, 'stayStartRows'),
                  finishRows: getCPUOperationUint32Rows(id, inputs, 'stayFinishRows'),
                  durations: getCPUOperationFloat32Rows(id, inputs, 'stayDurations'),
                  count: getCPUOperationUint32Rows(id, inputs, 'stayCount'),
                  overflow: props.staypoints.status.overflow
                    ? getCPUOperationUint32Rows(id, inputs, 'stayOverflow')
                    : undefined,
                  candidateOverflow: props.staypoints.status.candidateOverflow
                    ? getCPUOperationUint32Rows(id, inputs, 'stayCandidateOverflow')
                    : undefined
                }
              : undefined
          },
          {
            triplegCapacity: props.triplegCapacity,
            tripCapacity: props.output.trips ? props.tripCapacity : undefined
          }
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectorySegmentsResult]
          ])
        );
      }
    });
  }
}
