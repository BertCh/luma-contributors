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
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSpatialInteractionOperation,
  type SpatialInteractionOperationProps
} from './gpu-spatial-interaction-operation';

/** One packed row per explicit OD cell; duplicate cells retain independent Poisson contributions. */
export type CPUSpatialInteractionInput = {
  origins: Uint32Array;
  destinations: Uint32Array;
  flows: Float32Array;
  covariate: Float32Array;
  pairState?: Uint32Array;
};
/** Model settings. GPU iteration budgets are accepted but do not stop the float64 solve. */
export type CPUSpatialInteractionOptions = Pick<
  SpatialInteractionOperationProps,
  | 'originCount'
  | 'destinationCount'
  | 'constraint'
  | 'coefficients'
  | 'calibration'
  | 'maximumCalibrationIterations'
  | 'coefficientTolerance'
  | 'maximumIterations'
  | 'tolerance'
>;
/** The contributor's scalar layouts. Optional output columns are always returned by the function. */
export type CPUSpatialInteractionResult = {
  predictions: Float32Array;
  summary: Float32Array;
  status: Uint32Array;
  candidates: Float32Array;
  calibration?: Float32Array;
  originResiduals: Float32Array;
  destinationResiduals: Float32Array;
};

type GravityFit = {predictions: number[]; diagnostics: [number, number, number, number]};

function validateOptions(id: string, options: CPUSpatialInteractionOptions): void {
  validateCapacity(id, 'originCount', options.originCount);
  validateCapacity(id, 'destinationCount', options.destinationCount);
  validateCapacity(id, 'coefficients.length', options.coefficients.length);
  for (const [index, coefficient] of options.coefficients.entries())
    validateRequirement(
      id,
      Number.isFinite(coefficient),
      `coefficients[${index}] must be a finite number`
    );
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 256);
  validateCapacity(id, 'maximumCalibrationIterations', options.maximumCalibrationIterations ?? 32);
  validateFiniteNumber(id, 'tolerance', options.tolerance ?? 1e-5, 0);
  validateRequirement(id, (options.tolerance ?? 1e-5) > 0, 'tolerance must be positive');
  validateFiniteNumber(id, 'coefficientTolerance', options.coefficientTolerance ?? 1e-5, 0);
  validateRequirement(
    id,
    (options.coefficientTolerance ?? 1e-5) > 0,
    'coefficientTolerance must be positive'
  );
  validateRequirement(
    id,
    ['unconstrained', 'origin', 'destination', 'doubly'].includes(options.constraint),
    'constraint must be unconstrained, origin, destination or doubly'
  );
  validateRequirement(
    id,
    ['grid', 'bounded'].includes(options.calibration ?? 'grid'),
    'calibration must be grid or bounded'
  );
  validateRequirement(
    id,
    options.calibration !== 'bounded' ||
      (options.coefficients.length === 2 && options.coefficients[0] < options.coefficients[1]),
    'bounded coefficients must hold exactly two increasing endpoints'
  );
}

function getLogSum(values: readonly number[]): number {
  let largest = -Infinity;
  for (const value of values) largest = Math.max(largest, value);
  if (largest === -Infinity) return -Infinity;
  let total = 0;
  for (const value of values) total += Math.exp(value - largest);
  return largest + Math.log(total);
}

/**
 * Calibrates the four Poisson gravity constraints, with log-domain proportional fitting and
 * float64 candidate selection. Missing pairs are excluded before reading IDs/flow/covariate;
 * structural-zero covariates are ignored, as on the GPU.
 *
 * What differs from the GPU contributor:
 * - Float64 log-sum-exp balancing removes the GPU's exponent-span >80 rejection (candidate code 4
 *   is never set). Predictions outside the float64 exponential range become zero internally;
 *   their finite log means still contribute to deviance. Final float32 rounding can underflow
 *   additional predictions. Subnormals are retained internally.
 * - Pair/group/candidate caps and both GPU iteration caps are dropped. Balancing continues to
 *   tolerance or float64 stagnation; bounded calibration continues to coefficientTolerance or
 *   an unrepresentable narrower bracket. Candidate code 1/status 3 can therefore describe
 *   stagnation, and status 4 only a bracket that float64 cannot narrow. Exhausted GPU fits may
 *   succeed; iteration counts may differ at tolerance boundaries.
 * - Plain numeric props stay float64. Coefficients need not be representable as float32 shader
 *   literals. A non-finite float64 coefficient-times-covariate product reports invalid input.
 * - Output arithmetic is rounded once. Compare predictions/deviance with 5e-4 relative tolerance;
 *   marginal errors/residuals near numerical zero need absolute tolerance relative to mass.
 *   Grid ties choose the first candidate; bounded ties retain the lower bracket, as on the GPU.
 * - A candidate with finite float64 deviance remains successful even if publishing that deviance
 *   as float32 gives Infinity. The GPU's finite-deviance gate may instead reject it (candidate
 *   code 1/status 3); the target-device outcome is covered by a pending headless case.
 * - The pure function accepts empty pair arrays (status 1); an operation needs positive pairCount.
 *
 * @throws If origins, destinations, flows, covariate or pairState lengths disagree.
 */
export function computeSpatialInteractionOnCPU(
  input: CPUSpatialInteractionInput,
  options: CPUSpatialInteractionOptions
): CPUSpatialInteractionResult {
  const id = 'computeSpatialInteractionOnCPU';
  validateOptions(id, options);
  const count = input.flows.length;
  for (const name of ['origins', 'destinations', 'covariate', 'pairState'] as const)
    if (input[name] && input[name].length !== count)
      throw new Error(`${id} ${name} length must equal flows length`);
  const originTotals = new Array<number>(options.originCount).fill(0);
  const destinationTotals = new Array<number>(options.destinationCount).fill(0);
  const originPairs = originTotals.map(() => [] as number[]);
  const destinationPairs = destinationTotals.map(() => [] as number[]);
  const active: number[] = [];
  const structural: number[] = [];
  let invalid = false;
  let observed = 0;
  for (let pair = 0; pair < count; pair++) {
    const state = input.pairState?.[pair] ?? 1;
    if (state === 2) continue;
    const origin = input.origins[pair];
    const destination = input.destinations[pair];
    const flow = input.flows[pair];
    if (
      state > 2 ||
      origin >= options.originCount ||
      destination >= options.destinationCount ||
      !Number.isFinite(flow) ||
      flow < 0 ||
      (state === 0 && flow !== 0)
    ) {
      invalid = true;
      continue;
    }
    if (state === 0) {
      structural.push(pair);
      continue;
    }
    active.push(pair);
    observed += flow;
    originTotals[origin] += flow;
    destinationTotals[destination] += flow;
    originPairs[origin].push(pair);
    destinationPairs[destination].push(pair);
  }
  const fitCoefficient = (coefficient: number): GravityFit => {
    const predictions = new Array<number>(count).fill(NaN);
    const logMeans = new Array<number>(count).fill(-Infinity);
    let invalidCandidate = invalid;
    for (const pair of active) {
      logMeans[pair] = coefficient * input.covariate[pair];
      invalidCandidate ||= !Number.isFinite(logMeans[pair]);
    }
    if (invalidCandidate || !(observed > 0))
      return {predictions, diagnostics: [NaN, NaN, 0, invalidCandidate ? 3 : 2]};
    for (const pair of structural) predictions[pair] = 0;
    let iterations = 0;
    let error = 0;
    let converged = options.constraint === 'unconstrained';
    const normalize = (pairs: number[], total: number) => {
      if (pairs.length === 0) return;
      let largest = -Infinity;
      for (const pair of pairs) largest = Math.max(largest, logMeans[pair]);
      const adjustment =
        total > 0
          ? Math.log(total) - getLogSum(pairs.map(pair => logMeans[pair] - largest))
          : -Infinity;
      for (const pair of pairs)
        logMeans[pair] =
          adjustment === -Infinity ? -Infinity : logMeans[pair] - largest + adjustment;
    };
    const updatePredictions = () => {
      for (const pair of active) predictions[pair] = Math.exp(logMeans[pair]);
    };
    if (converged) {
      normalize(active, observed);
      updatePredictions();
    } else {
      const seenStates = new Set<string>();
      while (true) {
        const previous = active.map(pair => logMeans[pair]);
        if (options.constraint !== 'destination')
          originPairs.forEach((pairs, group) => normalize(pairs, originTotals[group]));
        if (options.constraint !== 'origin')
          destinationPairs.forEach((pairs, group) => normalize(pairs, destinationTotals[group]));
        iterations++;
        updatePredictions();
        error = 0;
        if (options.constraint !== 'destination')
          originPairs.forEach((pairs, group) => {
            error = Math.max(
              error,
              Math.abs(
                pairs.reduce((sum, pair) => sum + predictions[pair], 0) - originTotals[group]
              ) / Math.max(1, originTotals[group])
            );
          });
        if (options.constraint !== 'origin')
          destinationPairs.forEach((pairs, group) => {
            error = Math.max(
              error,
              Math.abs(
                pairs.reduce((sum, pair) => sum + predictions[pair], 0) - destinationTotals[group]
              ) / Math.max(1, destinationTotals[group])
            );
          });
        converged = error <= (options.tolerance ?? 1e-5);
        if (
          converged ||
          active.every((pair, index) => logMeans[pair] === previous[index]) ||
          !Number.isFinite(error)
        )
          break;
        const state = active.map(pair => logMeans[pair]).join(',');
        if (seenStates.has(state)) break;
        seenStates.add(state);
      }
    }
    let deviance = 0;
    for (const pair of active) {
      const flow = input.flows[pair];
      const prediction = predictions[pair];
      // expm1 avoids cancellation when observed and fitted counts nearly agree.
      if (flow > 0) {
        const difference = logMeans[pair] - Math.log(flow);
        deviance +=
          2 *
          flow *
          (Math.abs(difference) < 0.5
            ? Math.expm1(difference) - difference
            : prediction / flow - 1 - difference);
      } else deviance += 2 * prediction;
    }
    if (!Number.isFinite(deviance)) converged = false;
    return {
      predictions,
      diagnostics: [Math.max(0, deviance), error, iterations, converged ? 0 : 1]
    };
  };
  let coefficients = Array.from(options.coefficients);
  let fits: GravityFit[];
  let calibration: Float32Array | undefined;
  let bracketConverged = true;
  if (options.calibration === 'bounded') {
    let lower = coefficients[0];
    let upper = coefficients[1];
    const ratio = (Math.sqrt(5) - 1) / 2;
    while (true) {
      coefficients = [ratio * lower + (1 - ratio) * upper, (1 - ratio) * lower + ratio * upper];
      fits = coefficients.map(fitCoefficient);
      const firstValid = fits[0].diagnostics[3] === 0;
      const secondValid = fits[1].diagnostics[3] === 0;
      if (upper - lower <= (options.coefficientTolerance ?? 1e-5) || (!firstValid && !secondValid))
        break;
      const chooseFirst =
        firstValid && (!secondValid || fits[0].diagnostics[0] <= fits[1].diagnostics[0]);
      const nextLower = chooseFirst ? lower : coefficients[0];
      const nextUpper = chooseFirst ? coefficients[1] : upper;
      if (nextLower === lower && nextUpper === upper) break;
      lower = nextLower;
      upper = nextUpper;
    }
    bracketConverged = upper - lower <= (options.coefficientTolerance ?? 1e-5);
    calibration = Float32Array.of(lower, upper, upper - lower, Number(bracketConverged));
  } else fits = coefficients.map(fitCoefficient);
  let chosen = -1;
  let best = Infinity;
  let code = 3;
  fits.forEach((fit, candidate) => {
    if (fit.diagnostics[3] === 3) code = 2;
    else if (fit.diagnostics[3] === 2 && code !== 2) code = 1;
    if (fit.diagnostics[3] === 0 && fit.diagnostics[0] < best) {
      best = fit.diagnostics[0];
      chosen = candidate;
    }
  });
  const result: CPUSpatialInteractionResult = {
    predictions: new Float32Array(count).fill(NaN),
    summary: new Float32Array(4).fill(NaN),
    status: Uint32Array.of(code),
    candidates: Float32Array.from(fits.flatMap(fit => fit.diagnostics)),
    calibration,
    originResiduals: new Float32Array(options.originCount).fill(NaN),
    destinationResiduals: new Float32Array(options.destinationCount).fill(NaN)
  };
  if (chosen >= 0) {
    const fit = fits[chosen];
    result.predictions.set(fit.predictions);
    result.summary.set([coefficients[chosen], ...fit.diagnostics.slice(0, 3)]);
    result.status[0] = bracketConverged ? 0 : 4;
    if (result.status[0] === 0) {
      result.originResiduals.set(
        originPairs.map(pairs =>
          pairs.reduce((sum, pair) => sum + input.flows[pair] - fit.predictions[pair], 0)
        )
      );
      result.destinationResiduals.set(
        destinationPairs.map(pairs =>
          pairs.reduce((sum, pair) => sum + input.flows[pair] - fit.predictions[pair], 0)
        )
      );
    }
  }
  return result;
}

/** CPU gravity operation; GPU size and iteration caps are dropped. */
export class CPUSpatialInteractionOperation extends GPUCPUOperation {
  constructor(props: SpatialInteractionOperationProps) {
    const id = props.id ?? 'spatial-interaction';
    validateOptions(id, props);
    validateCapacity(id, 'pairCount', props.pairCount);
    validateRequirement(
      id,
      !props.output.calibration || props.calibration === 'bounded',
      'output.calibration requires bounded calibration'
    );
    const operation = new GPUSpatialInteractionOperation(props);
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
        const flows = getCPUOperationFloat32Rows(id, inputs, 'flows');
        if (flows.length !== props.pairCount)
          throw new Error(`${id} flows length must equal pairCount`);
        const result = computeSpatialInteractionOnCPU(
          {
            origins: getCPUOperationUint32Rows(id, inputs, 'origins'),
            destinations: getCPUOperationUint32Rows(id, inputs, 'destinations'),
            flows,
            covariate: getCPUOperationFloat32Rows(id, inputs, 'covariate'),
            pairState: props.pairState
              ? getCPUOperationUint32Rows(id, inputs, 'pairState')
              : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSpatialInteractionResult]!
          ])
        );
      }
    });
  }
}
