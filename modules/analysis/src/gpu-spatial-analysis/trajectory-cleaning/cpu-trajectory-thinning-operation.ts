// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
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
import {getCPUTrajectoryDistance} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_MOVEMENT_SENTINEL} from '../trajectory-staypoints/movement-common';
import {
  GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH,
  type GPUTrajectoryThinningMode
} from './gpu-trajectory-thinning';
import {
  GPUTrajectoryThinningOperation,
  type TrajectoryThinningOperationProps
} from './gpu-trajectory-thinning-operation';

/** Packed rows and getGPUTrajectoryThinningParameterValues layout. */
export type CPUTrajectoryThinningInput = CPUTrajectoryRows & {parameters: Float32Array};
/** An explicit search budget binds each anchor's forward search. */
export type CPUTrajectoryThinningOptions = {
  mode: GPUTrajectoryThinningMode;
  capacity: number;
  spatialContext?: GPUSpatialContext;
  maximumSearchRows?: number;
};
/** All ports, including pre-capacity keep flags and capacity-clamped per-track offsets. */
export type CPUTrajectoryThinningResult = {
  rowIds: Uint32Array;
  outputTrackOffsets: Uint32Array;
  validity: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  candidateOverflow: Uint32Array;
  outputPositions: Float32Array;
  outputTimestamps: Float32Array | Uint32Array;
};

/**
 * Follows the first later fix at or beyond the tolerance from each kept fix; always keeps both
 * track endpoints. Validity describes the full selection even when output capacity overflows.
 *
 * What differs from the GPU contributor:
 * - Omitted maximumSearchRows leaves CPU searches uncapped; explicit caps split the chain and
 *   set candidateOverflow exactly as on the GPU.
 * - Distances and time differences use float64 with one output rounding; Int64 subtraction is
 *   exact before conversion. Threshold decisions at float32 rounding boundaries can differ.
 * - Decreasing trackOffsets after clamping throw an Error.
 * - Unwritten copied geometry capacity rows are zero; the GPU leaves them untouched.
 * - Nonfinite distances or times fail the tolerance comparison when NaN; track endpoints are
 *   still kept. GPU results for nonfinite coordinates or times are unspecified.
 *
 * @throws Error naming inconsistent positions, timestamps, trackOffsets or parameters.
 */
export function computeTrajectoryThinningOnCPU(
  input: CPUTrajectoryThinningInput,
  options: CPUTrajectoryThinningOptions
): CPUTrajectoryThinningResult {
  const id = 'trajectory-thinning';
  validateCapacity(id, 'capacity', options.capacity);
  const maximumSearchRows = options.maximumSearchRows ?? Number.MAX_SAFE_INTEGER;
  validateCapacity(id, 'maximumSearchRows', maximumSearchRows, 1, Number.MAX_SAFE_INTEGER);
  validateRequirement(
    id,
    ['minimum-distance', 'minimum-time'].includes(options.mode),
    "mode must be 'minimum-distance' or 'minimum-time'"
  );
  const context = resolveGPUSpatialContext(
    id,
    options.spatialContext,
    GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
  );
  const radius = getGPUGreatCircleRadius(context);
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  if (input.parameters.length < GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH)
    throw new Error('parameters must hold the trajectory thinning parameter layout');
  const [tolerance] = input.parameters;
  const {capacity} = options;
  const result: CPUTrajectoryThinningResult = {
    rowIds: new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL),
    outputTrackOffsets: new Uint32Array(trackOffsets.length),
    validity: new Uint32Array(rowCount),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    candidateOverflow: new Uint32Array(1),
    outputPositions: new Float32Array(capacity * 2),
    outputTimestamps:
      input.timestamps instanceof Uint32Array
        ? new Uint32Array(capacity * 2)
        : new Float32Array(capacity)
  };
  let requiredCount = 0;
  const keepRow = (row: number) => {
    result.validity[row] = 1;
    if (requiredCount < capacity) {
      result.rowIds[requiredCount] = row;
      result.outputPositions.set(input.positions.subarray(row * 2, row * 2 + 2), requiredCount * 2);
      if (input.timestamps instanceof Uint32Array)
        result.outputTimestamps.set(
          input.timestamps.subarray(row * 2, row * 2 + 2),
          requiredCount * 2
        );
      else result.outputTimestamps[requiredCount] = input.timestamps[row];
    }
    requiredCount++;
  };
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const start = trackOffsets[track];
    const end = trackOffsets[track + 1];
    if (start < end) {
      let anchor = start;
      while (anchor < end) {
        keepRow(anchor);
        const searchEnd = Math.min(end, anchor + 1 + maximumSearchRows);
        let successor = anchor;
        for (let row = anchor + 1; row < searchEnd; row++) {
          const difference =
            options.mode === 'minimum-time'
              ? getTimeDifference(row, anchor)
              : getCPUTrajectoryDistance(input.positions, row, anchor, radius);
          if (difference >= tolerance) {
            successor = row;
            break;
          }
        }
        if (successor === anchor && searchEnd < end) {
          successor = searchEnd;
          result.candidateOverflow[0] = 1;
        }
        if (successor === anchor) break;
        anchor = successor;
      }
      if (result.validity[end - 1] === 0) keepRow(end - 1);
    }
    result.outputTrackOffsets[track + 1] = Math.min(requiredCount, capacity);
  }
  result.requiredCount[0] = requiredCount;
  result.count[0] = Math.min(requiredCount, capacity);
  result.overflow[0] = Number(requiredCount > capacity);
  return result;
}

/** CPU twin; follows the same explicitly bounded anchor chain as the contributor. */
export class CPUTrajectoryThinningOperation extends GPUCPUOperation {
  constructor(props: TrajectoryThinningOperationProps) {
    const id = props.id ?? 'trajectory-thinning';
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'trackCount', props.trackCount);
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
      ['minimum-distance', 'minimum-time'].includes(props.mode),
      "mode must be 'minimum-distance' or 'minimum-time'"
    );
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        ['float32', 'uint32x2'].includes(props.timestampFormat),
      'timestampFormat must be float32 or uint32x2'
    );
    resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: [props.timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
    const operation = new GPUTrajectoryThinningOperation(props);
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
        const trackOffsets = getCPUOperationUint32Rows(id, inputs, 'trackOffsets');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must match rowCount`);
        if (trackOffsets.length !== props.trackCount + 1)
          throw new Error(`${id} trackOffsets length must match trackCount + 1`);
        const result = computeTrajectoryThinningOnCPU(
          {
            positions,
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectoryThinningResult]
          ])
        );
      }
    });
  }
}
