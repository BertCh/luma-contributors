// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {
  getCPUTrajectoryRows,
  type CPUTrajectoryRows
} from '../../analysis-core/operations/cpu-trajectory-rows';
import {GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH} from './gpu-trajectory-staypoints';
import {GPU_MOVEMENT_SENTINEL} from './movement-common';
import {getTrajectoryVotePosition} from './geographic-vote';
import {
  GPUTrajectoryStaypointsOperation,
  snapshotTrajectoryStaypointsOperationProps,
  type TrajectoryStaypointsOperationProps
} from './gpu-trajectory-staypoints-operation';

/** Packed inputs; words are signed Int64 timestamps in little-endian low/high order. */
export type CPUTrajectoryStaypointsInput = CPUTrajectoryRows & {
  mask?: Uint32Array;
  /** Layout of getGPUTrajectoryStaypointsParameterValues. */
  parameters: Float32Array;
};

/** All supported output ports, including initialized unused capacity rows. */
export type CPUTrajectoryStaypointsResult = {
  ids: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  candidateOverflow: Uint32Array;
  missingFixCount: Uint32Array;
  drawInstanceCount: Uint32Array;
  startRows: Uint32Array;
  endRows: Uint32Array;
  finishRows: Uint32Array;
  centroids: Float32Array;
  durations: Float32Array;
  staypointIds: Uint32Array;
};

/** Wrapped degree difference, using the GPU's nearest-even half-turn convention. */
function getLongitudeDelta(source: number, destination: number): number {
  const turns = (destination - source) / 360;
  const lowerTurns = Math.floor(turns);
  const fractionalTurns = turns - lowerTurns;
  const nearestTurns =
    lowerTurns + Number(fractionalTurns > 0.5 || (fractionalTurns === 0.5 && lowerTurns % 2 !== 0));
  const halfTurn = 180 * nearestTurns;
  return destination - halfTurn - (source + halfTurn);
}

/**
 * Sequential sliding-anchor staypoints, with the contributor's inclusive distance/time tests,
 * strict gap test, distinct-coordinate centroid and exclusive end rows.
 *
 * What differs from the GPU contributor:
 * - Omitted maximumSearchRows leaves CPU searches uncapped; explicit caps split windows,
 *   report candidateOverflow and write NaN durations and centroids on the cut track, as on the GPU.
 * - Arithmetic uses float64, rounded once to float32 outputs; threshold decisions near rounding
 *   boundaries can differ. Int64 timestamps are subtracted exactly before conversion to float64.
 * - No GPU row-count or search-budget limit is imposed. Output capacity still clamps count.
 * - Compact offsets must terminate at the row count; storage: 'resident-capacity' opts into
 *   GPU-compatible clamping. The GPU does not validate ordering.
 *
 * @throws Error naming positions, timestamps, trackOffsets or parameters for inconsistent arrays.
 */
export function computeTrajectoryStaypointsOnCPU(
  input: CPUTrajectoryStaypointsInput,
  options: {
    capacity: number;
    spatialContext?: GPUSpatialContext;
    maximumSearchRows?: number;
    missingFixPolicy?: 'break';
  }
): CPUTrajectoryStaypointsResult {
  const {positions, parameters, mask} = input;
  validateRequirement(
    'trajectory-staypoints',
    options.missingFixPolicy === undefined || options.missingFixPolicy === 'break',
    'missingFixPolicy must be break'
  );
  validateRequirement(
    'trajectory-staypoints',
    !mask || mask.length === positions.length / 2,
    'mask must match positions row count'
  );
  const isObserved = (row: number) =>
    (!mask || mask[row] !== 0) &&
    Number.isFinite(positions[row * 2]) &&
    Number.isFinite(positions[row * 2 + 1]);
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  if (parameters.length < GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH) {
    throw new Error('parameters must hold the staypoint parameter layout');
  }
  validateCapacity('trajectory-staypoints', 'capacity', options.capacity);
  const maximumSearchRows = options.maximumSearchRows ?? Number.MAX_SAFE_INTEGER;
  validateCapacity(
    'trajectory-staypoints',
    'maximumSearchRows',
    maximumSearchRows,
    1,
    Number.MAX_SAFE_INTEGER
  );
  const context = resolveGPUSpatialContext('trajectory-staypoints', options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  const [distanceThreshold, timeThreshold, maximumGap, includeLast] = parameters;
  const getDistance = (first: number, second: number): number => {
    const longitudeDifference = positions[second * 2] - positions[first * 2];
    const latitudeDifference = positions[second * 2 + 1] - positions[first * 2 + 1];
    if (radius === undefined) return Math.hypot(longitudeDifference, latitudeDifference);
    const radians = Math.PI / 180;
    const firstLatitude = positions[first * 2 + 1];
    const secondLatitude = positions[second * 2 + 1];
    const longitudeAngle = getLongitudeDelta(positions[first * 2], positions[second * 2]) * radians;
    // Match geodesicCentralAngle: colatitude preserves exact zero cosines at the poles,
    // and wrapped longitude makes equivalent antimeridian coordinates exactly coincident.
    const firstSine = Math.sin(firstLatitude * radians);
    const firstCosine = Math.sin((90 - Math.abs(firstLatitude)) * radians);
    const secondCosine = Math.sin((90 - Math.abs(secondLatitude)) * radians);
    const halfVersine = 2 * Math.sin(longitudeAngle / 2) ** 2;
    const east = secondCosine * Math.sin(longitudeAngle);
    const north = Math.sin(latitudeDifference * radians) + firstSine * secondCosine * halfVersine;
    const up =
      firstSine * Math.sin(secondLatitude * radians) +
      firstCosine * secondCosine * (1 - halfVersine);
    return radius * Math.atan2(Math.hypot(east, north), up);
  };
  const {capacity} = options;
  const result: CPUTrajectoryStaypointsResult = {
    ids: new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL),
    startRows: new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL),
    endRows: new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL),
    finishRows: new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL),
    centroids: new Float32Array(capacity * 2),
    durations: new Float32Array(capacity),
    staypointIds: new Uint32Array(rowCount).fill(GPU_MOVEMENT_SENTINEL),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    candidateOverflow: new Uint32Array(1),
    missingFixCount: new Uint32Array(1),
    drawInstanceCount: new Uint32Array(1)
  };
  let requiredCount = 0;
  const emitStaypoint = (track: number, start: number, end: number, finish: number) => {
    const outputRow = requiredCount++;
    if (outputRow >= capacity) return;
    result.ids[outputRow] = track;
    result.startRows[outputRow] = start;
    result.endRows[outputRow] = end;
    result.finishRows[outputRow] = finish;
    result.durations[outputRow] = getTimeDifference(finish, start);
    const distinct = new Set<string>();
    const [anchorLongitude, anchorLatitude] = getTrajectoryVotePosition(
      positions[start * 2],
      positions[start * 2 + 1],
      radius !== undefined
    );
    let longitudeOffset = 0;
    let latitudeOffset = 0;
    for (let row = start; row < end; row++) {
      result.staypointIds[row] = outputRow;
      const [longitude, latitude] = getTrajectoryVotePosition(
        positions[row * 2],
        positions[row * 2 + 1],
        radius !== undefined
      );
      const key =
        Number.isNaN(longitude) || Number.isNaN(latitude)
          ? `nan-row-${row}`
          : `${longitude},${latitude}`;
      if (distinct.has(key)) continue;
      distinct.add(key);
      const delta = longitude - anchorLongitude;
      longitudeOffset +=
        radius === undefined ? delta : getLongitudeDelta(anchorLongitude, longitude);
      latitudeOffset += latitude - anchorLatitude;
    }
    const longitude = getTrajectoryVotePosition(
      anchorLongitude + longitudeOffset / distinct.size,
      0,
      radius !== undefined
    )[0];
    const roundedLongitude = Math.fround(longitude);
    result.centroids[outputRow * 2] =
      radius !== undefined && roundedLongitude >= 180 ? roundedLongitude - 360 : roundedLongitude;
    result.centroids[outputRow * 2 + 1] = anchorLatitude + latitudeOffset / distinct.size;
  };
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const end = trackOffsets[track + 1];
    let anchor = trackOffsets[track];
    if (anchor === end) continue;
    const firstOutputRow = requiredCount;
    let isTruncated = false;
    while (anchor < end) {
      if (!isObserved(anchor)) {
        result.missingFixCount[0]++;
        anchor++;
        continue;
      }
      const searchEnd = Math.min(end, anchor + 1 + maximumSearchRows);
      let successor = anchor;
      for (let row = anchor + 1; row < searchEnd; row++) {
        if (!isObserved(row)) {
          if (getTimeDifference(row - 1, anchor) >= timeThreshold)
            emitStaypoint(track, anchor, row, row - 1);
          successor = row;
          break;
        }
        if (maximumGap > 0 && getTimeDifference(row, row - 1) > maximumGap) {
          successor = row;
          break;
        } else if (getDistance(anchor, row) >= distanceThreshold) {
          if (getTimeDifference(row, anchor) >= timeThreshold)
            emitStaypoint(track, anchor, row, row);
          successor = row;
          break;
        }
      }
      if (successor !== anchor) {
        anchor = successor;
        continue;
      }
      if (searchEnd < end) {
        result.candidateOverflow[0] = 1;
        isTruncated = true;
        if (getTimeDifference(searchEnd, anchor) >= timeThreshold)
          emitStaypoint(track, anchor, searchEnd, searchEnd);
        anchor = searchEnd;
        continue;
      }
      if (includeLast !== 0 && getTimeDifference(end - 1, anchor) >= timeThreshold)
        emitStaypoint(track, anchor, end, end - 1);
      break;
    }
    if (isTruncated) {
      // A cut window is not a staypoint: NaN, never a plausible duration or centroid.
      const lastOutputRow = Math.min(requiredCount, capacity);
      result.durations.fill(NaN, Math.min(firstOutputRow, capacity), lastOutputRow);
      result.centroids.fill(NaN, Math.min(firstOutputRow, capacity) * 2, lastOutputRow * 2);
    }
  }
  result.requiredCount[0] = requiredCount;
  result.count[0] = Math.min(requiredCount, capacity);
  result.drawInstanceCount[0] = result.count[0];
  result.overflow[0] = Number(requiredCount > capacity);
  return result;
}

/**
 * CPU staypoints operation. Explicit search budgets publish search loss under the required
 * `output.status.candidateOverflow` graph value.
 */
export class CPUTrajectoryStaypointsOperation extends GPUCPUOperation {
  constructor(props: TrajectoryStaypointsOperationProps) {
    props = snapshotTrajectoryStaypointsOperationProps(props);
    const operation = new GPUTrajectoryStaypointsOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'capacity', props.capacity);
    if (props.maximumSearchRows !== undefined)
      validateCapacity(
        id,
        'maximumSearchRows',
        props.maximumSearchRows,
        1,
        Number.MAX_SAFE_INTEGER
      );
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        props.timestampFormat === 'float32' ||
        props.timestampFormat === 'uint32x2',
      'timestampFormat must be float32 or uint32x2'
    );
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: [props.timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
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
          throw new Error(`${id} positions must match rowCount`);
        const result = computeTrajectoryStaypointsOnCPU(
          {
            positions,
            storage: 'resident-capacity',
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets: getCPUOperationUint32Rows(id, inputs, 'trackOffsets'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            mask:
              props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectoryStaypointsResult]
          ])
        );
      }
    });
  }
}
