// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {invertCPUScaledMatrix} from '../../analysis-core/kernels/cpu-scaled-inverse';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryRows,
  type CPUTrajectoryRows
} from '../../analysis-core/operations/cpu-trajectory-rows';
import {
  GPUTrajectoryKalmanOperation,
  type TrajectoryKalmanOperationProps
} from './gpu-trajectory-kalman-operation';

/** Packed observations, timestamps (signed Int64 low/high words or float32) and parameters. */
export type CPUTrajectoryKalmanInput = {
  positions: Float32Array;
  timestamps: Float32Array | Uint32Array;
  trackOffsets: Uint32Array;
  /** Compact input must cover all observations. Defaults to GPU-compatible resident capacity. */
  storage?: CPUTrajectoryRows['storage'];
  /** Isotropic [q,r,v,maximumGap]; full [qxx,qxy,qyy,rxx,rxy,ryy,vxx,vxy,vyy,maximumGap]. */
  parameters: Float32Array;
};

/** Shape, optional smoothing, and the same coordinate/time declarations as the contributor. */
export type CPUTrajectoryKalmanOptions = {
  covarianceMode?: 'isotropic' | 'full';
  smooth?: boolean;
  spatialContext?: GPUSpatialContext;
  temporalContext?: GPUTemporalContext;
  timeUnitsPerSecond?: number;
};

/** Full covariance uses four column-major float32x4 rows per observation; isotropic uses one. */
export type CPUTrajectoryKalmanResult = {
  filteredState: Float32Array;
  filteredCovariance: Float32Array;
  status: Uint32Array;
  smoothedState?: Float32Array;
  smoothedCovariance?: Float32Array;
};

type KalmanMatrix = number[][];
const TIME_UNITS_PER_SECOND = {second: 1, millisecond: 1e3, microsecond: 1e6, nanosecond: 1e9};

function resolveKalmanOptions(
  id: string,
  options: CPUTrajectoryKalmanOptions,
  wordTime: boolean
): number {
  resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  validateRequirement(
    id,
    options.covarianceMode === undefined ||
      options.covarianceMode === 'isotropic' ||
      options.covarianceMode === 'full',
    'covarianceMode must be isotropic or full'
  );
  const context =
    options.temporalContext &&
    resolveGPUTemporalContext(id, options.temporalContext, {
      encodings: [wordTime ? 'int64-words' : 'float32-relative'],
      axisKinds: ['irregular']
    });
  const contextUnits = context && TIME_UNITS_PER_SECOND[context.unit];
  validateRequirement(
    id,
    options.timeUnitsPerSecond === undefined || context !== undefined || wordTime,
    'timeUnitsPerSecond needs a temporalContext or uint32x2 timestamps'
  );
  validateRequirement(
    id,
    options.timeUnitsPerSecond === undefined ||
      (contextUnits ?? options.timeUnitsPerSecond) === options.timeUnitsPerSecond,
    'timeUnitsPerSecond must equal the ticks per second of temporalContext.unit'
  );
  const units = contextUnits ?? options.timeUnitsPerSecond ?? 1;
  validateRequirement(
    id,
    Number.isFinite(units) && units > 0,
    'timeUnitsPerSecond must be positive and finite'
  );
  return units;
}

function createKalmanMatrix(rows: number, columns = rows): KalmanMatrix {
  return Array.from({length: rows}, () => new Array<number>(columns).fill(0));
}

function transposeKalmanMatrix(matrix: KalmanMatrix): KalmanMatrix {
  return matrix[0].map((_, column) => matrix.map(row => row[column]));
}

function multiplyKalmanMatrices(first: KalmanMatrix, second: KalmanMatrix): KalmanMatrix {
  return first.map(row =>
    second[0].map((_, column) =>
      row.reduce((sum, value, inner) => sum + value * second[inner][column], 0)
    )
  );
}

function addKalmanMatrices(first: KalmanMatrix, second: KalmanMatrix, scale = 1): KalmanMatrix {
  return first.map((row, rowIndex) =>
    row.map((value, column) => value + scale * second[rowIndex][column])
  );
}

function symmetrizeKalmanMatrix(matrix: KalmanMatrix): KalmanMatrix {
  return matrix.map((row, rowIndex) =>
    row.map((value, column) => (value + matrix[column][rowIndex]) / 2)
  );
}

function createKalmanIdentity(): KalmanMatrix {
  return Array.from({length: 4}, (_, row) =>
    Array.from({length: 4}, (_, column) => Number(row === column))
  );
}

function validateKalmanCovariance(matrix: KalmanMatrix, positive: boolean): boolean {
  const [first, cross, second] = [matrix[0][0], matrix[0][1], matrix[1][1]];
  if (![first, cross, second].every(Number.isFinite) || first < 0 || second < 0) return false;
  const scale = Math.max(first, Math.abs(cross), second);
  if (scale === 0) return !positive;
  const determinant = (first / scale) * (second / scale) - (cross / scale) ** 2;
  return positive ? determinant > 0 : determinant >= 0;
}

function invertKalmanInnovation(matrix: KalmanMatrix): KalmanMatrix {
  const scale = Math.max(...matrix.flat().map(Math.abs));
  const first = matrix[0][0] / scale;
  const cross = matrix[0][1] / scale;
  const second = matrix[1][1] / scale;
  const determinant = first * second - cross * cross;
  return [
    [second / determinant / scale, -cross / determinant / scale],
    [-cross / determinant / scale, first / determinant / scale]
  ];
}

function createKalmanTransition(delta: number): KalmanMatrix {
  const transition = createKalmanIdentity();
  transition[0][2] = delta;
  transition[1][3] = delta;
  return transition;
}

function createKalmanProcessCovariance(delta: number, noise: KalmanMatrix): KalmanMatrix {
  const covariance = createKalmanMatrix(4);
  for (let firstAxis = 0; firstAxis < 2; firstAxis++) {
    for (let secondAxis = 0; secondAxis < 2; secondAxis++) {
      const value = noise[firstAxis][secondAxis];
      covariance[firstAxis][secondAxis] = (value * delta ** 3) / 3;
      covariance[firstAxis][secondAxis + 2] = covariance[firstAxis + 2][secondAxis] =
        (value * delta ** 2) / 2;
      covariance[firstAxis + 2][secondAxis + 2] = value * delta;
    }
  }
  return covariance;
}

function predictKalmanCovariance(
  uncertainty: KalmanMatrix,
  transition: KalmanMatrix,
  process: KalmanMatrix
): KalmanMatrix {
  return addKalmanMatrices(
    multiplyKalmanMatrices(
      multiplyKalmanMatrices(transition, uncertainty),
      transposeKalmanMatrix(transition)
    ),
    process
  );
}

function packKalmanCovariance(matrix: KalmanMatrix, full: boolean): number[] {
  if (!full)
    return [
      matrix[0][0],
      (matrix[0][2] + matrix[2][0]) / 2,
      (matrix[0][2] + matrix[2][0]) / 2,
      matrix[2][2]
    ];
  return transposeKalmanMatrix(matrix).flat();
}

/**
 * Runs constant-velocity Kalman filtering and optional RTS smoothing, retaining binary64 states
 * and covariances throughout both passes. Resets, missing measurements, gap boundaries, covariance
 * layouts and status bits follow GPUTrajectoryKalman, including forward values at singular smooths.
 *
 * What differs from the GPU contributor:
 * - Arithmetic is float64 until the final float32 output, including filtered values used by the
 *   smoother. Compare well-conditioned outputs with relative tolerance 3e-4 (absolute 3e-5 at zero).
 * - Scaled covariance solves avoid GPU intermediate overflow. Invalid/smoothingSingular are set
 *   for failures in float64, so GPU float32 failures can instead produce valid CPU estimates.
 *   A finite binary64 estimate outside float32 range rounds to Infinity at output.
 * - Int64 times are differenced exactly before conversion to float64, rather than float32.
 *   maximumGap=0 means unlimited even for finite time differences above the GPU float32 maximum.
 * - By default or with `storage: 'resident-capacity'`, offsets clamp to the observation count
 *   as on GPU. Decreasing clamped offsets throw. Rows outside tracks are zero.
 * - With `storage: 'compact'`, offsets must start at zero, be nondecreasing and end at the
 *   observation count. A single `[0]` describes zero tracks over zero observations.
 * - No GPU-only limits exist to drop. Smoothing-singular rows retain their forward values, as the
 *   kernels do; an invalid parameter value returns status 4 rather than throwing.
 *
 * @throws Error naming positions, timestamps, trackOffsets or parameters for inconsistent arrays.
 */
export function computeTrajectoryKalmanOnCPU(
  input: CPUTrajectoryKalmanInput,
  options: CPUTrajectoryKalmanOptions = {}
): CPUTrajectoryKalmanResult {
  const wordTime = input.timestamps instanceof Uint32Array;
  const units = resolveKalmanOptions('computeTrajectoryKalmanOnCPU', options, wordTime);
  const {storage = 'resident-capacity'} = input;
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input, {storage});
  const full = options.covarianceMode === 'full';
  if (input.parameters.length !== (full ? 10 : 4))
    throw new Error(`parameters must have ${full ? 10 : 4} rows`);
  const parameters = input.parameters;
  const readCovariance = (offset: number) => [
    [parameters[offset], parameters[offset + 1]],
    [parameters[offset + 1], parameters[offset + 2]]
  ];
  const noise = full
    ? readCovariance(0)
    : [
        [parameters[0], 0],
        [0, parameters[0]]
      ];
  const measurement = full
    ? readCovariance(3)
    : [
        [parameters[1], 0],
        [0, parameters[1]]
      ];
  const initialVelocity = full
    ? readCovariance(6)
    : [
        [parameters[2], 0],
        [0, parameters[2]]
      ];
  const rawMaximumGap = parameters[full ? 9 : 3];
  const maximumGap = rawMaximumGap === 0 ? Infinity : rawMaximumGap;
  const validParameters =
    validateKalmanCovariance(noise, false) &&
    validateKalmanCovariance(measurement, true) &&
    validateKalmanCovariance(initialVelocity, false) &&
    maximumGap > 0;
  const covarianceWidth = full ? 16 : 4;
  const result: CPUTrajectoryKalmanResult = {
    filteredState: new Float32Array(rowCount * 4),
    filteredCovariance: new Float32Array(rowCount * covarianceWidth),
    status: new Uint32Array(rowCount),
    ...(options.smooth
      ? {
          smoothedState: new Float32Array(rowCount * 4),
          smoothedCovariance: new Float32Array(rowCount * covarianceWidth)
        }
      : {})
  };
  const states: (number[] | undefined)[] = new Array(rowCount);
  const covariances: (KalmanMatrix | undefined)[] = new Array(rowCount);
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const start = trackOffsets[track];
    const end = trackOffsets[track + 1];
    let initialized = false;
    let previousRow = start;
    let estimate = [0, 0, 0, 0];
    let uncertainty = createKalmanMatrix(4);
    for (let row = start; row < end; row++) {
      result.filteredState.fill(NaN, row * 4, (row + 1) * 4);
      result.filteredCovariance.fill(NaN, row * covarianceWidth, (row + 1) * covarianceWidth);
      const position = [input.positions[row * 2], input.positions[row * 2 + 1]];
      const measured = position.every(Number.isFinite);
      const elapsed = getTimeDifference(row, previousRow);
      const delta = elapsed / units;
      if (
        !validParameters ||
        (!wordTime && !Number.isFinite(input.timestamps[row])) ||
        (initialized && elapsed < 0)
      ) {
        result.status[row] = 4;
        initialized = false;
        continue;
      }
      if (initialized && elapsed > maximumGap) initialized = false;
      let flags = 0;
      if (!initialized) {
        if (!measured) {
          result.status[row] = 6;
          continue;
        }
        estimate = [...position, 0, 0];
        uncertainty = createKalmanMatrix(4);
        for (let firstAxis = 0; firstAxis < 2; firstAxis++) {
          for (let secondAxis = 0; secondAxis < 2; secondAxis++) {
            uncertainty[firstAxis][secondAxis] = measurement[firstAxis][secondAxis];
            uncertainty[firstAxis + 2][secondAxis + 2] = initialVelocity[firstAxis][secondAxis];
          }
        }
        initialized = true;
        flags = 1;
      } else {
        const transition = createKalmanTransition(delta);
        estimate = multiplyKalmanMatrices(
          transition,
          estimate.map(value => [value])
        ).flat();
        uncertainty = predictKalmanCovariance(
          uncertainty,
          transition,
          createKalmanProcessCovariance(delta, noise)
        );
        if (measured) {
          const innovation = addKalmanMatrices(
            uncertainty.slice(0, 2).map(values => values.slice(0, 2)),
            measurement
          );
          if (!validateKalmanCovariance(innovation, true)) {
            result.status[row] = 4;
            initialized = false;
            continue;
          }
          const gain = multiplyKalmanMatrices(
            uncertainty.map(values => values.slice(0, 2)),
            invertKalmanInnovation(innovation)
          );
          const residual = position.map((value, axis) => [value - estimate[axis]]);
          const correctionState = multiplyKalmanMatrices(gain, residual).flat();
          estimate = estimate.map((value, component) => value + correctionState[component]);
          const correction = createKalmanIdentity();
          for (let component = 0; component < 4; component++)
            for (let axis = 0; axis < 2; axis++)
              correction[component][axis] -= gain[component][axis];
          uncertainty = symmetrizeKalmanMatrix(
            addKalmanMatrices(
              multiplyKalmanMatrices(
                multiplyKalmanMatrices(correction, uncertainty),
                transposeKalmanMatrix(correction)
              ),
              multiplyKalmanMatrices(
                multiplyKalmanMatrices(gain, measurement),
                transposeKalmanMatrix(gain)
              )
            )
          );
        } else flags = 2;
      }
      if (
        !estimate.every(Number.isFinite) ||
        !uncertainty.every(values => values.every(Number.isFinite))
      ) {
        result.status[row] = 4;
        initialized = false;
        continue;
      }
      previousRow = row;
      states[row] = estimate;
      covariances[row] = uncertainty;
      result.filteredState.set(estimate, row * 4);
      result.filteredCovariance.set(packKalmanCovariance(uncertainty, full), row * covarianceWidth);
      result.status[row] = flags;
    }
    if (!options.smooth) continue;
    let nextState: number[] | undefined;
    let nextCovariance: KalmanMatrix | undefined;
    for (let row = end - 1; row >= start; row--) {
      const estimate = states[row];
      const uncertainty = covariances[row];
      if (!estimate || !uncertainty) {
        result.smoothedState!.fill(NaN, row * 4, (row + 1) * 4);
        result.smoothedCovariance!.fill(NaN, row * covarianceWidth, (row + 1) * covarianceWidth);
        nextState = undefined;
        nextCovariance = undefined;
        continue;
      }
      let correctedState = estimate;
      let correctedCovariance = uncertainty;
      if (row + 1 < end && (result.status[row + 1] & 5) === 0 && nextState && nextCovariance) {
        const delta = getTimeDifference(row + 1, row) / units;
        const transition = createKalmanTransition(delta);
        const predictedCovariance = predictKalmanCovariance(
          uncertainty,
          transition,
          createKalmanProcessCovariance(delta, noise)
        );
        const inverse = invertCPUScaledMatrix(predictedCovariance);
        if (!inverse) result.status[row] |= 8;
        else {
          const gain = multiplyKalmanMatrices(
            multiplyKalmanMatrices(uncertainty, transposeKalmanMatrix(transition)),
            inverse
          );
          const prediction = multiplyKalmanMatrices(
            transition,
            estimate.map(value => [value])
          ).flat();
          const residual = nextState.map((value, component) => [value - prediction[component]]);
          const correction = multiplyKalmanMatrices(gain, residual).flat();
          correctedState = estimate.map((value, component) => value + correction[component]);
          correctedCovariance = symmetrizeKalmanMatrix(
            addKalmanMatrices(
              uncertainty,
              multiplyKalmanMatrices(
                multiplyKalmanMatrices(
                  gain,
                  addKalmanMatrices(nextCovariance, predictedCovariance, -1)
                ),
                transposeKalmanMatrix(gain)
              )
            )
          );
          if (
            !correctedState.every(Number.isFinite) ||
            !correctedCovariance.every(values => values.every(Number.isFinite))
          ) {
            result.status[row] |= 8;
            correctedState = estimate;
            correctedCovariance = uncertainty;
          }
        }
      }
      result.smoothedState!.set(correctedState, row * 4);
      result.smoothedCovariance!.set(
        packKalmanCovariance(correctedCovariance, full),
        row * covarianceWidth
      );
      nextState = correctedState;
      nextCovariance = correctedCovariance;
    }
  }
  return result;
}

/** CPU Kalman twin; validates scalar/shape props at construction and arrays at execution. */
export class CPUTrajectoryKalmanOperation extends GPUCPUOperation {
  constructor(props: TrajectoryKalmanOperationProps) {
    const id = props.id ?? 'trajectory-kalman';
    validateCapacity(id, 'rowCount', props.rowCount);
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        props.timestampFormat === 'float32' ||
        props.timestampFormat === 'uint32x2',
      'timestampFormat must be float32 or uint32x2'
    );
    validateRequirement(
      id,
      Boolean(props.output.smoothedState) === Boolean(props.output.smoothedCovariance),
      'output.smoothedState and output.smoothedCovariance must be given together'
    );
    validateRequirement(
      id,
      Boolean(props.timestamps || props.trackOffsets) !== Boolean(props.times || props.offsets) &&
        Boolean(props.timestamps ?? props.times) &&
        Boolean(props.trackOffsets ?? props.offsets),
      'takes either timestamps and trackOffsets, or times and offsets'
    );
    resolveKalmanOptions(id, props, props.timestampFormat === 'uint32x2');
    const operation = new GPUTrajectoryKalmanOperation(props);
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
        const result = computeTrajectoryKalmanOnCPU(
          {
            positions,
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets: getCPUOperationUint32Rows(id, inputs, 'trackOffsets'),
            storage: 'resident-capacity',
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          {...props, smooth: props.output.smoothedState !== undefined}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectoryKalmanResult]!
          ])
        );
      }
    });
  }
}
