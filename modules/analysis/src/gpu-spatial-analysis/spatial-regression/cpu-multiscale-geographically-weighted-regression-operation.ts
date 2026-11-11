// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUSphereRadius,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getCPUTrajectoryDistance} from '../../analysis-core/operations/cpu-trajectory-distance';
import {fitCPURegressionWithQR} from './cpu-regression-linear-algebra';
import {GPU_ITERATIVE_REGRESSION_STATUS} from './gpu-generalized-geographically-weighted-regression';
import {
  GPUMultiscaleGeographicallyWeightedRegressionOperation,
  type MultiscaleGeographicallyWeightedRegressionOperationProps
} from './gpu-multiscale-geographically-weighted-regression-operation';

// The contributor has no parameter packer/constants: its documented layout is tolerance first,
// followed by one bandwidth per coefficient, intercept first.
const CONTRIBUTION_TOLERANCE_INDEX = 0;
const COEFFICIENT_BANDWIDTH_START = 1;

/** Packed float32x2 positions, scalar design/response, and the contributor's tolerance/bandwidths. */
export type CPUMultiscaleGeographicallyWeightedRegressionInput = {
  positions: Float32Array;
  predictors: Float32Array;
  response: Float32Array;
  parameters: Float32Array;
  mask?: Uint32Array;
  bandwidthCandidates?: Float32Array;
  noiseVariance?: Float32Array;
};
/** Fixed-bandwidth backfit settings; an index changes traversal only on the GPU. */
export type CPUMultiscaleGeographicallyWeightedRegressionOptions = Pick<
  MultiscaleGeographicallyWeightedRegressionOperationProps,
  | 'predictorCount'
  | 'kernel'
  | 'maximumIterations'
  | 'spatialContext'
  | 'indexGridSize'
  | 'gaussianCutoff'
  | 'bandwidthCandidateCount'
> & {inference?: {varianceConvention?: 'residual-df' | 'smoother-df'}};
/** Same row layouts and iterative statuses as the contributor, including optional full inference. */
export type CPUMultiscaleGeographicallyWeightedRegressionResult = {
  coefficients: Float32Array;
  fitted: Float32Array;
  codes: Uint32Array;
  iterationCount: Uint32Array;
  selectedBandwidths: Float32Array;
  conditionalStandardErrors: Float32Array;
  inference?: {
    standardErrors: Float32Array;
    effectiveParameters: Float32Array;
    summary: Float32Array;
    status: Uint32Array;
  };
};

function validateOptions(
  id: string,
  options: CPUMultiscaleGeographicallyWeightedRegressionOptions
): number | undefined {
  validateCapacity(id, 'predictorCount', options.predictorCount);
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 32);
  validateRequirement(
    id,
    options.kernel === undefined || options.kernel === 'bisquare' || options.kernel === 'gaussian',
    'kernel must be gaussian or bisquare'
  );
  if (options.gaussianCutoff !== undefined) {
    validateFiniteNumber(id, 'gaussianCutoff', options.gaussianCutoff, 0);
    validateRequirement(id, options.gaussianCutoff > 0, 'gaussianCutoff must be positive');
  }
  if (options.bandwidthCandidateCount !== undefined)
    validateCapacity(id, 'bandwidthCandidateCount', options.bandwidthCandidateCount);
  validateRequirement(
    id,
    options.inference === undefined ||
      ['residual-df', 'smoother-df'].includes(
        options.inference.varianceConvention ?? 'residual-df'
      ),
    'inference.varianceConvention must be residual-df or smoother-df'
  );
  const context = resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = context.metric === 'great-circle' ? getGPUSphereRadius(context) : undefined;
  if (options.indexGridSize) {
    validateRequirement(id, radius === undefined, 'indexGridSize requires planar spatialContext');
    validateCapacity(id, 'indexGridSize[0]', options.indexGridSize[0]);
    validateCapacity(id, 'indexGridSize[1]', options.indexGridSize[1]);
    validateRequirement(
      id,
      options.indexGridSize[0] * options.indexGridSize[1] < 0xffffffff,
      'indexGridSize cell count must be less than 0xffffffff'
    );
  }
  return radius;
}

/**
 * Cyclic additive backfitting from global OLS, with per-component leave-one-out bandwidth
 * selection, known-variance conditional errors and full response-smoother inference. Each component is replaced at all
 * included foci before the next component starts; exclusions have code 1 and NaN rows.
 *
 * What differs from the GPU contributor:
 * - Float64 QR initializes global coefficients; float64 distances, weighted sums, CV scores and
 *   contributions are rounded once at output. Compare well-conditioned row outputs with 5e-3
 *   relative tolerance. Near-tied CV scores and tolerance boundaries can change selected
 *   bandwidths or completed sweep counts; exact score ties still choose the first candidate.
 * - Shader-only predictor, row, candidate and iteration caps are dropped. maximumIterations is
 *   ignored: sweeps continue to contribution tolerance, numerical failure, or a repeated complete
 *   float64 state (code 3). An exhausted GPU fit can converge here.
 * - The CPU visits all complete-case rows in input order; indexGridSize has no computational
 *   effect. Gaussian cutoff is retained; bisquare weights are zero at or above the bandwidth.
 * - Great-circle distances call the shared float64 transcription of the GPU geodesic helper,
 *   including seam wrapping. Subnormals are retained; non-finite data are excluded as on the GPU.
 * - Full inference propagates the same selected bandwidth history from zero initialization in
 *   float64; its shader row/work caps are dropped. Variance uses float64 fitted values, rather
 *   than re-reading rounded float32 predictions. Conditional errors still use supplied variance.
 * - No getGPU parameter packer exists for this contributor; parameters retain its documented
 *   [contribution tolerance, intercept bandwidth, predictor bandwidths...] layout.
 *
 * @throws If positions, predictors, mask, parameters, bandwidthCandidates or noiseVariance lengths disagree.
 */
export function computeMultiscaleGeographicallyWeightedRegressionOnCPU(
  input: CPUMultiscaleGeographicallyWeightedRegressionInput,
  options: CPUMultiscaleGeographicallyWeightedRegressionOptions
): CPUMultiscaleGeographicallyWeightedRegressionResult {
  const id = 'computeMultiscaleGeographicallyWeightedRegressionOnCPU';
  const radius = validateOptions(id, options);
  const count = input.response.length;
  const width = options.predictorCount + 1;
  if (input.positions.length !== count * 2)
    throw new Error(`${id} positions length must equal response length * 2`);
  if (input.predictors.length !== count * options.predictorCount)
    throw new Error(`${id} predictors length must equal response length * predictorCount`);
  if (input.mask && input.mask.length !== count)
    throw new Error(`${id} mask length must equal response length`);
  if (input.parameters.length < width + COEFFICIENT_BANDWIDTH_START)
    throw new Error(`${id} parameters must hold tolerance and one bandwidth per coefficient`);
  if (input.noiseVariance && input.noiseVariance.length !== 1)
    throw new Error(`${id} noiseVariance must hold one value`);
  if (
    input.bandwidthCandidates &&
    (!options.bandwidthCandidateCount ||
      input.bandwidthCandidates.length !== width * options.bandwidthCandidateCount)
  )
    throw new Error(
      `${id} bandwidthCandidates length must equal (predictorCount + 1) * bandwidthCandidateCount`
    );
  if (!input.bandwidthCandidates && options.bandwidthCandidateCount !== undefined)
    throw new Error(`${id} bandwidthCandidateCount requires bandwidthCandidates`);
  const included = Array.from(
    {length: count},
    (_, row) =>
      (!input.mask || input.mask[row] !== 0) &&
      Number.isFinite(input.response[row]) &&
      Number.isFinite(input.positions[row * 2]) &&
      Number.isFinite(input.positions[row * 2 + 1]) &&
      (radius === undefined || Math.abs(input.positions[row * 2 + 1]) <= 90) &&
      Array.from(
        input.predictors.subarray(row * options.predictorCount, (row + 1) * options.predictorCount)
      ).every(Number.isFinite)
  );
  const usedRows = Array.from({length: count}, (_, row) => row).filter(row => included[row]);
  const predictor = (row: number, column: number) =>
    column === 0 ? 1 : input.predictors[row * options.predictorCount + column - 1];
  const design = usedRows.map(row =>
    Array.from({length: width}, (_, column) => predictor(row, column))
  );
  const globalResponseOrigin = usedRows.length ? input.response[usedRows[0]] : 0;
  const means = Array.from({length: options.predictorCount}, (_, column) => {
    const origin = usedRows.length ? predictor(usedRows[0], column + 1) : 0;
    return (
      origin +
      usedRows.reduce((sum, row) => sum + (predictor(row, column + 1) - origin), 0) /
        Math.max(1, usedRows.length)
    );
  });
  const centered = design.map(row =>
    row.map((value, column) => (column === 0 ? value : value - means[column - 1]))
  );
  const global =
    !options.inference && usedRows.length >= width
      ? fitCPURegressionWithQR(
          centered,
          usedRows.map(row => input.response[row] - globalResponseOrigin)
        )
      : undefined;
  const initial = global?.estimate.slice() ?? new Array<number>(width).fill(0);
  if (global)
    initial[0] +=
      globalResponseOrigin -
      means.reduce((sum, mean, column) => sum + mean * initial[column + 1], 0);
  const current = Array.from({length: count}, () => initial.slice());
  const responseDerivatives = options.inference
    ? Array.from({length: width}, () =>
        Array.from({length: count}, () => new Array<number>(count).fill(0))
      )
    : undefined;
  const bandwidths = Array.from(
    input.parameters.subarray(COEFFICIENT_BANDWIDTH_START, COEFFICIENT_BANDWIDTH_START + width)
  );
  const tolerance = input.parameters[CONTRIBUTION_TOLERANCE_INDEX];
  const weight = (focus: number, row: number, bandwidth: number) => {
    const distance = getCPUTrajectoryDistance(input.positions, focus, row, radius);
    const ratio = distance / bandwidth;
    if (options.kernel === 'bisquare') return distance >= bandwidth ? 0 : (1 - ratio * ratio) ** 2;
    if (options.gaussianCutoff !== undefined && distance > options.gaussianCutoff * bandwidth)
      return 0;
    return Math.exp(-0.5 * ratio * ratio);
  };
  let code: number = GPU_ITERATIVE_REGRESSION_STATUS.ITERATION_LIMIT;
  let iterationCount = 0;
  const seenStates = new Set<string>();
  while (code === GPU_ITERATIVE_REGRESSION_STATUS.ITERATION_LIMIT) {
    const previous = current.map(row => row.slice());
    for (let column = 0; column < width; column++) {
      const partial = Array.from({length: count}, (_, row) => {
        let value = input.response[row];
        for (let other = 0; other < width; other++)
          if (other !== column) value -= predictor(row, other) * current[row][other];
        return value;
      });
      const fitScalar = (focus: number, bandwidth: number, leaveOneOut: boolean) => {
        if (!(bandwidth > 0) || !Number.isFinite(bandwidth)) return NaN;
        let numerator = 0;
        let denominator = 0;
        for (const row of usedRows) {
          if (leaveOneOut && row === focus) continue;
          const spatialWeight = weight(focus, row, bandwidth);
          if (!(spatialWeight > 0)) continue;
          const value = predictor(row, column);
          numerator += spatialWeight * value * partial[row];
          denominator += spatialWeight * value * value;
        }
        return denominator > 0 && Number.isFinite(denominator) ? numerator / denominator : NaN;
      };
      if (input.bandwidthCandidates) {
        let bestScore = Infinity;
        let best = NaN;
        for (let candidate = 0; candidate < options.bandwidthCandidateCount!; candidate++) {
          const bandwidth =
            input.bandwidthCandidates[column * options.bandwidthCandidateCount! + candidate];
          let score = 0;
          for (const focus of usedRows) {
            const coefficient = fitScalar(focus, bandwidth, true);
            const residual = partial[focus] - predictor(focus, column) * coefficient;
            score += residual * residual;
          }
          if (Number.isFinite(score) && score < bestScore) {
            best = bandwidth;
            bestScore = score;
          }
        }
        bandwidths[column] = best;
      }
      for (let focus = 0; focus < count; focus++)
        current[focus][column] = included[focus]
          ? fitScalar(focus, bandwidths[column], false)
          : NaN;
      if (responseDerivatives) {
        const next = Array.from({length: count}, () => new Array<number>(count).fill(0));
        for (const focus of usedRows) {
          const weights = usedRows.map(row => weight(focus, row, bandwidths[column]));
          const denominator = usedRows.reduce(
            (sum, row, position) => sum + weights[position] * predictor(row, column) ** 2,
            0
          );
          for (const responseRow of usedRows) {
            let numerator = 0;
            for (const [position, row] of usedRows.entries()) {
              let partialDerivative = Number(row === responseRow);
              for (let other = 0; other < width; other++)
                if (other !== column)
                  partialDerivative -=
                    predictor(row, other) * responseDerivatives[other][row][responseRow];
              numerator += weights[position] * predictor(row, column) * partialDerivative;
            }
            next[focus][responseRow] = denominator > 0 ? numerator / denominator : NaN;
          }
        }
        responseDerivatives[column] = next;
      }
    }
    iterationCount++;
    if (!(tolerance > 0) || !Number.isFinite(tolerance)) {
      code = GPU_ITERATIVE_REGRESSION_STATUS.NUMERICAL_FAILURE;
      break;
    }
    let maximumChange = 0;
    let failed = false;
    for (const row of usedRows) {
      for (let column = 0; column < width; column++) {
        const coefficient = current[row][column];
        if (!Number.isFinite(coefficient)) {
          code = GPU_ITERATIVE_REGRESSION_STATUS.SINGULAR;
          failed = true;
          break;
        }
        const contribution = predictor(row, column) * coefficient;
        const before = predictor(row, column) * previous[row][column];
        if (!Number.isFinite(contribution) || !Number.isFinite(before)) {
          code = GPU_ITERATIVE_REGRESSION_STATUS.NUMERICAL_FAILURE;
          failed = true;
          break;
        }
        maximumChange = Math.max(
          maximumChange,
          Math.abs(contribution - before) / Math.max(1, Math.abs(before))
        );
      }
      if (failed) break;
    }
    if (failed) break;
    if (usedRows.length === 0) {
      code = GPU_ITERATIVE_REGRESSION_STATUS.EXCLUDED;
      break;
    }
    if (maximumChange <= tolerance) {
      code = GPU_ITERATIVE_REGRESSION_STATUS.CONVERGED;
      break;
    }
    const state = [...bandwidths, ...usedRows.flatMap(row => current[row])].join(',');
    if (seenStates.has(state)) break;
    seenStates.add(state);
  }
  const result: CPUMultiscaleGeographicallyWeightedRegressionResult = {
    coefficients: new Float32Array(count * width).fill(NaN),
    fitted: new Float32Array(count).fill(NaN),
    codes: Uint32Array.from(included, keep =>
      keep ? code : GPU_ITERATIVE_REGRESSION_STATUS.EXCLUDED
    ),
    iterationCount: Uint32Array.of(iterationCount),
    selectedBandwidths: Float32Array.from(bandwidths),
    conditionalStandardErrors: new Float32Array(count * width).fill(NaN)
  };
  if (code === GPU_ITERATIVE_REGRESSION_STATUS.CONVERGED) {
    const noise = input.noiseVariance?.[0];
    for (const focus of usedRows) {
      result.coefficients.set(current[focus], focus * width);
      result.fitted[focus] = current[focus].reduce(
        (sum, coefficient, column) => sum + predictor(focus, column) * coefficient,
        0
      );
      if (noise !== undefined && noise >= 0 && Number.isFinite(noise))
        for (let column = 0; column < width; column++) {
          let numerator = 0;
          let denominator = 0;
          for (const row of usedRows) {
            const spatialWeight = weight(focus, row, bandwidths[column]);
            const value = predictor(row, column);
            denominator += spatialWeight * value * value;
            numerator += spatialWeight * spatialWeight * value * value;
          }
          const standardError = denominator > 0 ? Math.sqrt(noise * numerator) / denominator : NaN;
          if (Number.isFinite(standardError))
            result.conditionalStandardErrors[focus * width + column] = standardError;
        }
    }
  }
  if (responseDerivatives) {
    const inference = {
      standardErrors: new Float32Array(count * width).fill(NaN),
      effectiveParameters: new Float32Array(width).fill(NaN),
      summary: new Float32Array(4).fill(NaN),
      status: Uint32Array.of(1)
    };
    result.inference = inference;
    if (code === GPU_ITERATIVE_REGRESSION_STATUS.CONVERGED || usedRows.length === 0) {
      let trace = 0;
      let traceSquared = 0;
      let residualSquares = 0;
      const effectiveParameters = new Array<number>(width).fill(0);
      for (const focus of usedRows) {
        const fitted = current[focus].reduce(
          (sum, coefficient, column) => sum + predictor(focus, column) * coefficient,
          0
        );
        residualSquares += (input.response[focus] - fitted) ** 2;
        for (const responseRow of usedRows) {
          let influence = 0;
          for (let column = 0; column < width; column++)
            influence += predictor(focus, column) * responseDerivatives[column][focus][responseRow];
          traceSquared += influence * influence;
          if (focus === responseRow) trace += influence;
        }
        for (let column = 0; column < width; column++)
          effectiveParameters[column] +=
            predictor(focus, column) * responseDerivatives[column][focus][focus];
      }
      const degrees =
        usedRows.length -
        (options.inference?.varianceConvention === 'smoother-df'
          ? 2 * trace - traceSquared
          : trace);
      const variance = residualSquares / degrees;
      inference.summary.set([trace, traceSquared, variance, degrees]);
      inference.status[0] =
        usedRows.length > 0 &&
        degrees > 0 &&
        variance >= 0 &&
        Number.isFinite(variance) &&
        Number.isFinite(traceSquared)
          ? 0
          : 2;
      if (inference.status[0] === 0) {
        inference.effectiveParameters.set(effectiveParameters);
        for (const focus of usedRows)
          for (let column = 0; column < width; column++) {
            const squares = responseDerivatives[column][focus].reduce(
              (sum, derivative) => sum + derivative * derivative,
              0
            );
            inference.standardErrors[focus * width + column] = Math.sqrt(
              Math.max(0, squares * variance)
            );
          }
      }
    }
  }
  return result;
}

/** CPU MGWR, including full inference, without shader size/sweep/work caps. */
export class CPUMultiscaleGeographicallyWeightedRegressionOperation extends GPUCPUOperation {
  constructor(props: MultiscaleGeographicallyWeightedRegressionOperationProps) {
    const id = props.id ?? 'multiscale-geographically-weighted-regression';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount);
    validateRequirement(
      id,
      !props.output.conditionalStandardErrors || props.noiseVariance !== undefined,
      'output.conditionalStandardErrors requires noiseVariance'
    );
    validateRequirement(
      id,
      (props.bandwidthCandidates !== undefined) === (props.bandwidthCandidateCount !== undefined),
      'bandwidthCandidates and bandwidthCandidateCount must be supplied together'
    );
    const operation = new GPUMultiscaleGeographicallyWeightedRegressionOperation(props);
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
        const response = getCPUOperationFloat32Rows(id, inputs, 'response');
        if (response.length !== props.rowCount)
          throw new Error(`${id} response length must equal rowCount`);
        const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            predictors: getCPUOperationFloat32Rows(id, inputs, 'predictors'),
            response,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            mask: props.mask ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined,
            bandwidthCandidates: props.bandwidthCandidates
              ? getCPUOperationFloat32Rows(id, inputs, 'bandwidthCandidates')
              : undefined,
            noiseVariance: props.noiseVariance
              ? getCPUOperationFloat32Rows(id, inputs, 'noiseVariance')
              : undefined
          },
          props
        );
        const values = {
          ...result,
          inferenceStandardErrors: result.inference?.standardErrors,
          inferenceEffectiveParameters: result.inference?.effectiveParameters,
          inferenceSummary: result.inference?.summary,
          inferenceStatus: result.inference?.status
        };
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            values[port as keyof typeof values] as Float32Array | Uint32Array
          ])
        );
      }
    });
  }
}
