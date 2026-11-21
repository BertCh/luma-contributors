// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSinkhornTransportRadiusOption} from './sinkhorn-transport-radius';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/index';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS} from './gpu-sinkhorn-divergence';
import {GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH} from './sinkhorn-divergence-parameters';
import {
  GPUSinkhornDivergenceOperation,
  type SinkhornDivergenceOperationProps
} from './gpu-sinkhorn-divergence-operation';

export type CPUSinkhornDivergenceOptions = Pick<
  SinkhornDivergenceOperationProps,
  'grid' | 'spatialContext' | 'maximumIterations' | 'transportRadius'
>;
export type CPUSinkhornDivergenceResult = {
  divergence: Float32Array;
  components: Float32Array;
  marginalError: Float32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
  invalidCount: Uint32Array;
  approximation: Uint32Array;
};

/** Two invalid masses per grid cell must still fit the uint32 invalidCount output. */
export const CPU_SINKHORN_DIVERGENCE_MAXIMUM_CELL_COUNT = 0x7fffffff;

function validateTransportOptions(id: string, options: CPUSinkhornDivergenceOptions): void {
  const cellCount = validateRasterGrid(id, options.grid);
  validateCapacity(id, 'grid cell count', cellCount, 1, CPU_SINKHORN_DIVERGENCE_MAXIMUM_CELL_COUNT);
  resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
    measure: 'area'
  });
  validateCapacity(
    id,
    'maximumIterations',
    options.maximumIterations ?? GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS,
    1
  );
  const radius = options.transportRadius;
  if (radius !== undefined) validateCapacity(id, 'transportRadius', radius, 0);
}

// Full separable log-sum-exp: zeros are -Infinity, so no finite stand-in drops a distant term.
function computeSoftMinimum(
  potential: Float64Array,
  logMass: Float64Array,
  width: number,
  height: number,
  cellWidth: number,
  cellHeight: number,
  epsilon: number,
  radius: number,
  reachable?: Uint8Array
): Float64Array {
  const horizontal = new Float64Array(potential.length).fill(-Infinity);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      let maximum = -Infinity;
      let sum = 0;
      for (
        let other = Math.max(0, column - radius);
        other <= Math.min(width - 1, column + radius);
        other++
      ) {
        const cellIndex = row * width + other;
        if (logMass[cellIndex] === -Infinity) continue;
        const distance = (other - column) * cellWidth;
        const term =
          potential[cellIndex] / epsilon + logMass[cellIndex] - (distance * distance) / epsilon;
        if (term > maximum) {
          sum = sum * Math.exp(maximum - term) + 1;
          maximum = term;
        } else sum += Math.exp(term - maximum);
      }
      if (sum > 0) horizontal[row * width + column] = maximum + Math.log(sum);
    }
  }
  const output = new Float64Array(potential.length);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      let maximum = -Infinity;
      let sum = 0;
      for (
        let other = Math.max(0, row - radius);
        other <= Math.min(height - 1, row + radius);
        other++
      ) {
        const value = horizontal[other * width + column];
        if (value === -Infinity) continue;
        const distance = (other - row) * cellHeight;
        const term = value - (distance * distance) / epsilon;
        if (term > maximum) {
          sum = sum * Math.exp(maximum - term) + 1;
          maximum = term;
        } else sum += Math.exp(term - maximum);
      }
      if (reachable) reachable[row * width + column] = Number(sum > 0);
      output[row * width + column] = sum > 0 ? -epsilon * (maximum + Math.log(sum)) : 1e30;
    }
  }
  return output;
}

/**
 * Runs the separable log-domain Sinkhorn method in float64, including the symmetric self
 * updates, epsilon schedule, all-three-plan marginal stopping test, and the final half-update on a
 * budget-limited run. The explicit maximumIterations is a solver budget, with no GPU unroll cap.
 *
 * What differs from the GPU contributor:
 * - Arithmetic and normalization use float64 with one output rounding. Stopping near tolerance
 *   can change iterationCount and convergence flags; no universal relative bound exists near
 *   zero divergence or marginal error. Well-conditioned values agree within 3e-4 relative.
 * - Omitting transportRadius evaluates full support; an explicit transportRadius bounds each axis.
 *   No finite masked-log cutoff discards a term. The explicit solver
 *   budget is retained, including iterationLimitReached; its GPU maximum of 1000 is dropped.
 * - The GPU's grid-side, row-count and storage-binding limits are dropped; uint32 grid dimensions
 *   and diagnostic counts remain. A 1x1 grid is accepted by both routes.
 * - Subnormals and intermediates outside float32 range are retained. Invalid masses are counted
 *   and zeroed as on the GPU; zero total mass or invalid packed parameters gives NaN values.
 *
 * @throws If source/target do not match the grid or parameters is shorter than its packed layout.
 */
export function computeSinkhornDivergenceOnCPU(
  source: Float32Array,
  target: Float32Array,
  parameters: Float32Array,
  options: CPUSinkhornDivergenceOptions
): CPUSinkhornDivergenceResult {
  validateSinkhornTransportRadiusOption('computeSinkhornDivergenceOnCPU', options);
  validateTransportOptions('computeSinkhornDivergenceOnCPU', options);
  const {width, height} = options.grid;
  const count = width * height;
  if (source.length !== count) throw new Error('source length must equal grid.width * grid.height');
  if (target.length !== count) throw new Error('target length must equal grid.width * grid.height');
  if (parameters.length < GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH)
    throw new Error('parameters must hold the Sinkhorn parameter layout');
  // Layout of getGPUSinkhornDivergenceParameterValues.
  const [cellWidth, cellHeight, finalEpsilon, tolerance, epsilonStart, scalingFactor] = parameters;
  const radius = options.transportRadius ?? Math.max(width, height) - 1;
  const result: CPUSinkhornDivergenceResult = {
    divergence: Float32Array.of(NaN),
    components: new Float32Array(3).fill(NaN),
    marginalError: Float32Array.of(NaN),
    converged: Uint32Array.of(0),
    iterationLimitReached: Uint32Array.of(0),
    iterationCount: Uint32Array.of(0),
    invalidCount: Uint32Array.of(0),
    approximation: Uint32Array.of(Number(radius < Math.max(width, height) - 1))
  };
  const normalize = (masses: Float32Array) => {
    let total = 0;
    for (const mass of masses) {
      if (Number.isFinite(mass) && mass >= 0) total += mass;
      else result.invalidCount[0]++;
    }
    const logarithms = Float64Array.from(masses, mass =>
      Number.isFinite(mass) && mass > 0 && total > 0 ? Math.log(mass) - Math.log(total) : -Infinity
    );
    return {total, logarithms};
  };
  const first = normalize(source);
  const second = normalize(target);
  if (
    !(
      first.total > 0 &&
      second.total > 0 &&
      cellWidth > 0 &&
      cellHeight > 0 &&
      finalEpsilon > 0 &&
      tolerance > 0 &&
      epsilonStart >= finalEpsilon &&
      scalingFactor > 0 &&
      scalingFactor <= 1 &&
      [cellWidth, cellHeight, finalEpsilon, tolerance, epsilonStart].every(Number.isFinite)
    )
  )
    return result;
  let sourceOfCross: Float64Array = new Float64Array(count);
  let targetOfCross: Float64Array = new Float64Array(count);
  let sourceOfSelf = new Float64Array(count);
  let targetOfSelf = new Float64Array(count);
  const sourceMass = Float64Array.from(first.logarithms, Math.exp);
  const targetMass = Float64Array.from(second.logarithms, Math.exp);
  const maximumIterations =
    options.maximumIterations ?? GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS;
  const targetReachable = new Uint8Array(count);
  let epsilon = epsilonStart;
  let marginalError = NaN;
  for (let iteration = 1; iteration <= maximumIterations; iteration++) {
    const nextTarget = computeSoftMinimum(
      sourceOfCross,
      first.logarithms,
      width,
      height,
      cellWidth,
      cellHeight,
      epsilon,
      radius,
      targetReachable
    );
    const nextSourceSelf = computeSoftMinimum(
      sourceOfSelf,
      first.logarithms,
      width,
      height,
      cellWidth,
      cellHeight,
      epsilon,
      radius
    );
    const nextTargetSelf = computeSoftMinimum(
      targetOfSelf,
      second.logarithms,
      width,
      height,
      cellWidth,
      cellHeight,
      epsilon,
      radius
    );
    let error = 0;
    let sourceSelfError = 0;
    let targetSelfError = 0;
    const getMarginalResidual = (mass: number, logMass: number, delta: number): number => {
      if (mass === 0) return 0;
      // Evaluate the mass times the exponential before it can overflow on its own.
      return delta > 1
        ? Math.abs(Math.exp(logMass + delta) - mass)
        : mass * Math.abs(Math.expm1(delta));
    };
    for (let cell = 0; cell < count; cell++) {
      error += getMarginalResidual(
        targetMass[cell],
        second.logarithms[cell],
        targetReachable[cell] === 0 ? -Infinity : (targetOfCross[cell] - nextTarget[cell]) / epsilon
      );
      // The symmetric plan uses the same potential on both sides; its row and column
      // residuals agree. Measure before damping, just as for the cross-plan update.
      sourceSelfError += getMarginalResidual(
        sourceMass[cell],
        first.logarithms[cell],
        (sourceOfSelf[cell] - nextSourceSelf[cell]) / epsilon
      );
      targetSelfError += getMarginalResidual(
        targetMass[cell],
        second.logarithms[cell],
        (targetOfSelf[cell] - nextTargetSelf[cell]) / epsilon
      );
      sourceOfSelf[cell] = (sourceOfSelf[cell] + nextSourceSelf[cell]) / 2;
      targetOfSelf[cell] = (targetOfSelf[cell] + nextTargetSelf[cell]) / 2;
    }
    targetOfCross = nextTarget;
    result.iterationCount[0] = iteration;
    marginalError = error;
    const converged =
      epsilon === finalEpsilon &&
      error <= tolerance &&
      sourceSelfError <= tolerance &&
      targetSelfError <= tolerance;
    epsilon = Math.max(finalEpsilon, epsilonStart * scalingFactor ** iteration);
    if (converged) {
      result.converged[0] = 1;
      break;
    }
    if (iteration === maximumIterations) result.iterationLimitReached[0] = 1;
    sourceOfCross = computeSoftMinimum(
      targetOfCross,
      second.logarithms,
      width,
      height,
      cellWidth,
      cellHeight,
      epsilon,
      radius
    );
  }
  let cross = 0;
  let ownSource = 0;
  let ownTarget = 0;
  let divergence = 0;
  for (let cell = 0; cell < count; cell++) {
    cross += sourceMass[cell] * sourceOfCross[cell] + targetMass[cell] * targetOfCross[cell];
    ownSource += 2 * sourceMass[cell] * sourceOfSelf[cell];
    ownTarget += 2 * targetMass[cell] * targetOfSelf[cell];
    divergence +=
      sourceMass[cell] * (sourceOfCross[cell] - sourceOfSelf[cell]) +
      targetMass[cell] * (targetOfCross[cell] - targetOfSelf[cell]);
  }
  result.components.set([cross, ownSource, ownTarget]);
  result.divergence[0] = divergence;
  result.marginalError[0] = marginalError;
  return result;
}

/** CPU twin with optional transport truncation and without GPU unroll/storage caps; retains the explicit solver budget. */
export class CPUSinkhornDivergenceOperation extends GPUCPUOperation {
  constructor(props: SinkhornDivergenceOperationProps) {
    const id = props.id ?? 'sinkhorn-divergence';
    validateSinkhornTransportRadiusOption(id, props);
    props = {
      ...props,
      grid: {...props.grid},
      output: {...props.output, status: {...props.output.status}}
    };
    validateTransportOptions(id, props);
    const operation = new GPUSinkhornDivergenceOperation(props);
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
        const result = computeSinkhornDivergenceOnCPU(
          getCPUOperationFloat32Rows(id, inputs, 'source'),
          getCPUOperationFloat32Rows(id, inputs, 'target'),
          getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSinkhornDivergenceResult]
          ])
        );
      }
    });
  }
}
