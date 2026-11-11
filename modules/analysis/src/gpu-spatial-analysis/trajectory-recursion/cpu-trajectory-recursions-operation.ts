// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryRows,
  type CPUTrajectoryRows
} from '../../analysis-core/operations/cpu-trajectory-rows';
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
  GPUTrajectoryRecursionsOperation,
  type TrajectoryRecursionsOperationProps
} from './gpu-trajectory-recursions-operation';

/** Every supported output, in fix-major/radius-minor order. */
export type CPUTrajectoryRecursionsResult = {
  revisits: Uint32Array;
  residenceTimes: Float32Array;
  firstPassageTimes: Float32Array;
  firstPassageCensored: Uint32Array;
  candidateOverflow: Uint32Array;
};

/**
 * Counts closed-disc visits, integrates residence times and finds the forward first exit,
 * using linear interpolation between fixes of the same track. Non-positive radii produce zero
 * times/counts and censored=1; rows outside all tracks do likewise. Non-positive segment
 * durations add no residence time, matching the GPU definition.
 *
 * What differs from the GPU contributor:
 * - Whole tracks are examined: maximumTrackSegments is ignored, candidateOverflow is always zero,
 *   and forward passage is censored only at the real track end.
 * - The GPU radius-count maximum is dropped; the fix-major output layout is unchanged.
 * - Circle intersections and time sums use float64, rounded once on output. The intersection
 *   calculation uses projection onto the segment's supporting line to avoid subtracting large
 *   quadratic discriminant terms. Near-boundary decisions can differ from float32 GPU decisions.
 * - Int64 timestamps are subtracted exactly before conversion to float64.
 * - The pure function accepts empty trajectories or radii and returns empty row outputs. The CPU
 *   graph operation retains the GPU contributor's minima of one fix and one radius.
 * - Offsets are clamped to the row count, then decreasing offsets throw. The GPU does not
 *   validate ordering; decreasing offsets can produce overlapping track ranges there.
 * - Non-finite positions are not defined by the GPU contributor. In the twin, segments touching
 *   NaN coordinates have no inside interval; an outside endpoint that ends passage on such a
 *   segment uses its start time. For positions [(0, 0), (NaN, 0), (2, 0)], times [0, 1, 2], one
 *   track and radius 1, the twin returns revisits [0, 0, 0], residence [0, 0, 0], passage [1, 1, 0]
 *   and censored [0, 1, 1]. The device comparison returned GPU [2, 2, 2] where the twin returned
 *   [0, 0, 0] (measured on an Apple GPU, 2026-11-11); that undefined GPU result is not matched.
 *
 * @throws Error naming positions, timestamps or trackOffsets for inconsistent packed arrays.
 */
export function computeTrajectoryRecursionsOnCPU(
  input: CPUTrajectoryRows & {radii: Float32Array}
): CPUTrajectoryRecursionsResult {
  const {positions, radii} = input;
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  const length = rowCount * radii.length;
  const result: CPUTrajectoryRecursionsResult = {
    revisits: new Uint32Array(length),
    residenceTimes: new Float32Array(length),
    firstPassageTimes: new Float32Array(length),
    firstPassageCensored: new Uint32Array(length).fill(1),
    candidateOverflow: new Uint32Array(1)
  };
  const getDistance = (first: number, second: number) =>
    Math.hypot(
      positions[first * 2] - positions[second * 2],
      positions[first * 2 + 1] - positions[second * 2 + 1]
    );
  const getInsideInterval = (
    segment: number,
    center: number,
    radius: number
  ): [number, number] | undefined => {
    const stepX = positions[segment * 2 + 2] - positions[segment * 2];
    const stepY = positions[segment * 2 + 3] - positions[segment * 2 + 1];
    const offsetX = positions[segment * 2] - positions[center * 2];
    const offsetY = positions[segment * 2 + 1] - positions[center * 2 + 1];
    const squaredLength = stepX * stepX + stepY * stepY;
    if (squaredLength === 0) return Math.hypot(offsetX, offsetY) <= radius ? [0, 1] : undefined;
    const projection = -(offsetX * stepX + offsetY * stepY) / squaredLength;
    const closestX = offsetX + projection * stepX;
    const closestY = offsetY + projection * stepY;
    const squaredHalfWidth =
      (radius * radius - closestX * closestX - closestY * closestY) / squaredLength;
    if (!(squaredHalfWidth >= 0)) return undefined;
    const halfWidth = Math.sqrt(squaredHalfWidth);
    const first = Math.max(0, projection - halfWidth);
    const last = Math.min(1, projection + halfWidth);
    return first <= last ? [first, last] : undefined;
  };
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const start = trackOffsets[track];
    const end = trackOffsets[track + 1];
    for (let row = start; row < end; row++) {
      for (let radiusIndex = 0; radiusIndex < radii.length; radiusIndex++) {
        const radius = radii[radiusIndex];
        const slot = row * radii.length + radiusIndex;
        if (!(radius > 0)) continue;
        let visits = 0;
        let residence = 0;
        for (let segment = start; segment + 1 < end; segment++) {
          const outside = getDistance(segment, row) > radius;
          if (segment === start && !outside) visits++;
          const interval = getInsideInterval(segment, row, radius);
          if (interval) {
            if (outside) visits++;
            const duration = getTimeDifference(segment + 1, segment);
            if (duration > 0) residence += (interval[1] - interval[0]) * duration;
          }
        }
        result.revisits[slot] = Math.max(0, visits - 1);
        result.residenceTimes[slot] = residence;
        let passage = getTimeDifference(end - 1, row);
        for (let segment = row; segment + 1 < end; segment++) {
          if (getDistance(segment + 1, row) > radius) {
            const interval = getInsideInterval(segment, row, radius);
            // The path has not yet exited: its start is in the closed disc.
            passage =
              getTimeDifference(segment, row) +
              (interval?.[1] ?? 0) * getTimeDifference(segment + 1, segment);
            result.firstPassageCensored[slot] = 0;
            break;
          }
        }
        result.firstPassageTimes[slot] = passage;
      }
    }
  }
  return result;
}

/** CPU recursion operation; retains output shape constraints and removes GPU window/radius caps. */
export class CPUTrajectoryRecursionsOperation extends GPUCPUOperation {
  constructor(props: TrajectoryRecursionsOperationProps) {
    const operation = new GPUTrajectoryRecursionsOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(id, 'radiusCount', props.radiusCount);
    validateCapacity(id, 'output rows', props.rowCount * props.radiusCount);
    validateRequirement(
      id,
      Boolean(
        props.output.revisits || props.output.residenceTimes || props.output.firstPassageTimes
      ),
      'output must request revisits, residenceTimes or firstPassageTimes'
    );
    validateRequirement(
      id,
      !props.output.firstPassageCensored || Boolean(props.output.firstPassageTimes),
      'output.firstPassageCensored requires output.firstPassageTimes'
    );
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        props.timestampFormat === 'float32' ||
        props.timestampFormat === 'uint32x2',
      'timestampFormat must be float32 or uint32x2'
    );
    resolveGPUSpatialContext(id, props.spatialContext, GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT);
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
        const radii = getCPUOperationFloat32Rows(id, inputs, 'radii');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions must match rowCount`);
        if (radii.length !== props.radiusCount)
          throw new Error(`${id} radii must match radiusCount`);
        const result = computeTrajectoryRecursionsOnCPU({
          positions,
          radii,
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
              : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
          trackOffsets: getCPUOperationUint32Rows(id, inputs, 'trackOffsets')
        });
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectoryRecursionsResult]
          ])
        );
      }
    });
  }
}
