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
  GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT,
  GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH
} from './gpu-brownian-bridge-utilization';
import {
  GPUBrownianBridgeUtilizationOperation,
  type BrownianBridgeUtilizationOperationProps
} from './gpu-brownian-bridge-utilization-operation';

/** Packed views consumed by the bridge-density computation. */
export type CPUBrownianBridgeUtilizationInput = CPUTrajectoryRows & {
  motionVariances: Float32Array;
  locationErrors: Float32Array;
  /** Layout of getGPUBrownianBridgeUtilizationParameterValues: extent in float32. */
  parameters: Float32Array;
};

/**
 * Computes time-weighted Gaussian bridge density at each row-major grid cell centre. Positive
 * duration segments in each track contribute to the normalization, including zero-variance
 * segments; non-positive durations are skipped. Negative motion variances clamp to zero.
 *
 * What differs from the GPU contributor:
 * - No seven-sigma cutoff: every positive-variance midpoint contributes its full Gaussian tail.
 * - Arithmetic uses float64 with one float32 output rounding; Int64 differences are exact before
 *   float64 conversion. Comparisons use absolute/relative tolerances, rather than bitwise equality.
 * - Fix-count and subStepCount GPU maxima are dropped. The requested positive integer midpoint
 *   count still defines the integration rule; this does not replace it with continuous quadrature.
 * - The pure function accepts zero or one fix and returns a zero density grid. The CPU graph
 *   operation retains the GPU contributor's minimum of two fixes.
 * - Offsets are clamped to the row count, then decreasing offsets throw. The GPU does not
 *   validate ordering; decreasing offsets can produce overlapping track ranges there.
 * - Non-finite motion variances are not defined by the GPU contributor. The twin clamps
 *   -Infinity to zero; NaN and +Infinity contribute no density while retaining their positive
 *   segment duration in the denominator. For two stationary fixes at times [0, 1], motion
 *   variances [NaN, 0], location errors [1, 1], one midpoint and a cell centred at zero, the twin
 *   returns density 0; the GPU returned 0.31830987334251404 (measured on an Apple GPU, 2026-11-11).
 *
 * @throws Error naming any input array whose packed length or offsets are inconsistent.
 */
export function computeBrownianBridgeUtilizationOnCPU(
  input: CPUBrownianBridgeUtilizationInput,
  options: {gridSize: readonly [number, number]; subStepCount?: number}
): {density: Float32Array} {
  const {positions, motionVariances, locationErrors, parameters} = input;
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  if (motionVariances.length !== rowCount)
    throw new Error('motionVariances length must match positions');
  if (locationErrors.length !== rowCount)
    throw new Error('locationErrors length must match positions');
  if (parameters.length < GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH)
    throw new Error('parameters must contain the bridge extent layout');
  const [columns, lines] = options.gridSize;
  const subStepCount = options.subStepCount ?? GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT;
  validateCapacity('brownian-bridge-utilization', 'gridSize columns', columns);
  validateCapacity('brownian-bridge-utilization', 'gridSize rows', lines);
  validateCapacity('brownian-bridge-utilization', 'subStepCount', subStepCount);
  const [minimumX, minimumY, maximumX, maximumY] = parameters;
  const density = new Float32Array(columns * lines);
  const segments: {row: number; duration: number}[] = [];
  let totalDuration = 0;
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    for (let row = trackOffsets[track]; row + 1 < trackOffsets[track + 1]; row++) {
      const duration = getTimeDifference(row + 1, row);
      if (!(duration > 0)) continue;
      segments.push({row, duration});
      totalDuration += duration;
    }
  }
  if (!(totalDuration > 0)) return {density};
  for (let cell = 0; cell < density.length; cell++) {
    const centerX = minimumX + (((cell % columns) + 0.5) * (maximumX - minimumX)) / columns;
    const centerY = minimumY + ((Math.floor(cell / columns) + 0.5) * (maximumY - minimumY)) / lines;
    let weightedDensity = 0;
    for (const {row, duration} of segments) {
      const motion = Math.max(0, motionVariances[row]);
      const errorFrom = locationErrors[row];
      const errorTo = locationErrors[row + 1];
      let segmentDensity = 0;
      for (let step = 0; step < subStepCount; step++) {
        const fraction = (step + 0.5) / subStepCount;
        const remaining = 1 - fraction;
        const meanX = positions[row * 2] + fraction * (positions[row * 2 + 2] - positions[row * 2]);
        const meanY =
          positions[row * 2 + 1] + fraction * (positions[row * 2 + 3] - positions[row * 2 + 1]);
        const variance =
          duration * fraction * remaining * motion +
          remaining ** 2 * errorFrom ** 2 +
          fraction ** 2 * errorTo ** 2;
        if (variance > 0)
          segmentDensity +=
            Math.exp(-((centerX - meanX) ** 2 + (centerY - meanY) ** 2) / (2 * variance)) /
            (2 * Math.PI * variance);
      }
      weightedDensity += (duration * segmentDensity) / subStepCount;
    }
    density[cell] = weightedDensity / totalDuration;
  }
  return {density};
}

/** CPU bridge operation, preferred for small grids; no GPU fix-count or midpoint-count cap. */
export class CPUBrownianBridgeUtilizationOperation extends GPUCPUOperation {
  constructor(props: BrownianBridgeUtilizationOperationProps) {
    const operation = new GPUBrownianBridgeUtilizationOperation(props);
    const {id} = operation;
    validateCapacity(id, 'rowCount', props.rowCount, 2);
    validateCapacity(id, 'gridSize columns', props.gridSize[0]);
    validateCapacity(id, 'gridSize rows', props.gridSize[1]);
    validateCapacity(
      id,
      'subStepCount',
      props.subStepCount ?? GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT
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
      outputs: {
        [props.output.density]: gpuGraph.vector('float32', {
          length: props.gridSize[0] * props.gridSize[1]
        })
      },
      execute({inputs}) {
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions must match rowCount`);
        const result = computeBrownianBridgeUtilizationOnCPU(
          {
            positions,
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets: getCPUOperationUint32Rows(id, inputs, 'trackOffsets'),
            motionVariances: getCPUOperationFloat32Rows(id, inputs, 'motionVariances'),
            locationErrors: getCPUOperationFloat32Rows(id, inputs, 'locationErrors'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
        );
        return {[props.output.density]: result.density};
      }
    });
  }
}
