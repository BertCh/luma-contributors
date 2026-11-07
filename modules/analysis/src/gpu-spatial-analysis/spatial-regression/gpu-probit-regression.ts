// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  ERFC_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights,
  getGPUSpatialWeightsViews
} from '../spatial-weights/spatial-weights';
import {getCholeskyWGSL} from './spatial-regression-solve';

export type GPUProbitRegressionProps = {
  id?: string;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  maximumIterations?: number;
  tolerance?: number;
  /** Pinkse, Kelejian-Prucha and Pinkse-Slade diagnostics under independent binary probit.
   * Supplied square weights have zero diagonal; complete cases induce the retained graph.
   * These are asymptotic model diagnostics, not latent spatial-lag coefficient estimates.
   */
  spatialDiagnostics?: {
    weights: GPUSpatialWeights;
    output: {
      /** [Pinkse chi², p, KP normal z, two-sided p, Pinkse-Slade chi², p]. */
      tests: GraphDataView<'float32'>;
      /** 0 valid; 1 failed model/no cases; 2 invalid weights; 3 no positive diagnostic variance. */
      status: GraphDataView<'uint32'>;
    };
  };
  output: {
    /** Intercept first. Nonconverged finite iterates are available only with status 3. */
    coefficients: GraphDataView<'float32'>;
    /** Model-based expected-information standard errors, NaN unless converged. */
    standardErrors: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
    /** Average marginal effect per predictor (not an effect at the means). */
    marginalEffects: GraphDataView<'float32'>;
    /** Complete rows, log likelihood, deviance, iteration count. */
    summary: GraphDataView<'float32'>;
    /** 0 converged, 1 insufficient, 2 singular, 3 iteration/line-search exhaustion, 4 nonbinary or separated. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Nonspatial binary probit with bounded Fisher scoring and monotone likelihood backtracking.
 * Complete cases, centered/scaled predictors, implicit intercept. Gaussian CDF/derivatives use
 * f32; probabilities are floored at 1e-7 for numerical tails. The log likelihood and score use
 * compensated (Neumaier) summation and the line search tolerates likelihood noise relative to
 * |loglik| and sqrt(rows). Trial steps are halved until the likelihood does not decrease and
 * |linear predictor| <= 12, so an early overshoot is damped rather than rejected. All-one/all-zero
 * samples, and fits that stay unconverged while pinned against that bound, are reported as
 * separated (status 4), never successful estimates.
 *
 * Capacity: the fit kernel is a single invocation that loops over all rows (the small solve is
 * serial), so rows are capped at 65536 (enforced in the constructor). The row count is a pipeline
 * override, so models that differ only in row count share one shader module.
 * This is not a latent spatial-lag probit. Spatial residual diagnostics require explicit downstream
 * residual construction and spatial-statistic inference with the appropriate model assumptions.
 */
export class GPUProbitRegression implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUProbitRegressionProps) {
    this.id = props.id ?? 'probit-regression';
    const rows = props.response.length;
    const columns = props.predictorCount;
    if (props.spatialDiagnostics) {
      const diagnostics = props.spatialDiagnostics;
      if (
        validateGPUSpatialWeights(this.id, diagnostics.weights) !== rows ||
        diagnostics.output.tests.length !== 6 ||
        diagnostics.output.status.length !== 1
      )
        throw new Error(`${this.id}: diagnostic shape mismatch`);
      validatePackedView(diagnostics.output.tests, ['float32'], this.id);
      validatePackedView(diagnostics.output.status, ['uint32'], this.id);
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [...Object.values(props.output), ...Object.values(diagnostics.output)],
        [
          props.predictors,
          props.response,
          props.mask,
          ...getGPUSpatialWeightsViews(diagnostics.weights)
        ]
      );
    }
    validateCapacity(this.id, 'rows', rows, 1, 65536);
    validateCapacity(this.id, 'predictorCount', columns, 1, 11);
    validateCapacity(this.id, 'maximumIterations', props.maximumIterations ?? 64, 1, 256);
    if (!Number.isFinite(props.tolerance ?? 1e-5) || (props.tolerance ?? 1e-5) <= 0)
      throw new Error(`${this.id}: invalid tolerance`);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * columns],
      [props.response, 'float32', rows],
      [props.mask, 'uint32', rows],
      [props.output.coefficients, 'float32', columns + 1],
      [props.output.standardErrors, 'float32', columns + 1],
      [props.output.fitted, 'float32', rows],
      [props.output.marginalEffects, 'float32', columns],
      [props.output.summary, 'float32', 4],
      [props.output.status, 'uint32', 1]
    ] as const)
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.predictors,
      props.response,
      props.mask
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.response.length;
    const columns = props.predictorCount;
    const size = columns + 1;
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'f32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const inputBindings = [
      binding('predictors', props.predictors),
      binding('response', props.response),
      ...(props.mask ? [binding('mask', props.mask, 'read', 'u32')] : [])
    ];
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ERFC_VALUE_WGSL}
fn included(row: u32) -> bool {
 if (${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteBits(response[responseOffset + row])) { return false; }
 for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; } } return true;
}
fn probability(value: f32) -> f32 { return clamp(0.5 * erfcValue(-value * 0.7071067811865475), 1e-7, 1.0 - 1e-7); }
fn density(value: f32) -> f32 { return 0.3989422804014327 * exp(-0.5 * value * value); }
`;
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.mask,
      ...Object.values(props.output)
    ]);
    const nodes = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: 'GPUProbitRegression',
        variant: 'fit',
        invocationCount: 1,
        constants: {ROW_COUNT: rows},
        bindings: [
          ...inputBindings,
          binding('coefficients', props.output.coefficients, 'read_write'),
          binding('errors', props.output.standardErrors, 'read_write'),
          binding('summary', props.output.summary, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32')
        ],
        declarations: `${common}\noverride ROW_COUNT: u32;\n${getCholeskyWGSL(size)}
fn getDesign(row: u32, means: ptr<function, array<f32, ${columns}>>, scales: ptr<function, array<f32, ${columns}>>) -> array<f32, ${size}> { var design: array<f32, ${size}>; design[0] = 1.0;
 for (var column = 0u; column < ${columns}u; column++) { design[column + 1u] = (predictors[predictorsOffset + row * ${columns}u + column] - (*means)[column]) * (*scales)[column]; } return design; }
var<private> lastLargestLinear: f32;
fn likelihood(coefficients: ptr<function, array<f32, ${size}>>, means: ptr<function, array<f32, ${columns}>>, scales: ptr<function, array<f32, ${columns}>>) -> f32 { var value = 0.0; var compensation = 0.0; var largestLinear = 0.0;
for (var row = 0u; row < ROW_COUNT; row++) { if (!included(row)) { continue; } let design = getDesign(row, means, scales); var linear = 0.0; for (var column = 0u; column < ${size}u; column++) { linear += design[column] * (*coefficients)[column]; } largestLinear = max(largestLinear, abs(linear)); let fitted = probability(linear); let outcome = response[responseOffset + row]; let term = outcome * log(fitted) + (1.0 - outcome) * log(1.0 - fitted);
 // Neumaier compensated sum: a plain f32 sum loses the low bits of the log likelihood for large row counts.
 let total = value + term; if (abs(value) >= abs(term)) { compensation += (value - total) + term; } else { compensation += (term - total) + value; } value = total; }
lastLargestLinear = largestLinear; return value + compensation; }
`,
        body: `status[statusOffset] = 1u;
for (var column = 0u; column < ${size}u; column++) { coefficients[coefficientsOffset + column] = getNaN(); errors[errorsOffset + column] = getNaN(); }
for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); }
var count = 0u; var successes = 0.0; var means: array<f32, ${columns}>; var scales: array<f32, ${columns}>;
for (var row = 0u; row < ROW_COUNT; row++) { if (!included(row)) { continue; } let outcome = response[responseOffset + row]; if (outcome != 0.0 && outcome != 1.0) { status[statusOffset] = 4u; return; } count++; successes += outcome;
for (var column = 0u; column < ${columns}u; column++) { let value = predictors[predictorsOffset + row * ${columns}u + column]; let difference = value - means[column]; means[column] += difference / f32(count); scales[column] += difference * (value - means[column]); }
}
summary[summaryOffset] = f32(count); summary[summaryOffset + 3u] = 0.0;
if (count <= ${size}u) { return; } if (successes == 0.0 || successes == f32(count)) { status[statusOffset] = 4u; return; }
for (var column = 0u; column < ${columns}u; column++) { if (!(scales[column] > 0.0)) { status[statusOffset] = 2u; return; } scales[column] = sqrt(f32(count) / scales[column]); }
var estimate: array<f32, ${size}>; var information: array<f32, ${size * size}>; var code = 3u; var iterations = 0u; var current = likelihood(&estimate, &means, &scales); var largestLinear = 0.0;
// Likelihood noise floor: the compensated sum is exact to about eps * |loglik|, the per-row log/erfc errors add roughly sqrt(count) of that.
let noiseScale = 4.0 * 1.1920929e-7 * max(1.0, sqrt(f32(count)));
for (var iteration = 0u; iteration <= ${props.maximumIterations ?? 64}u; iteration++) {
 information = array<f32, ${size * size}>(); var gradient: array<f32, ${size}>; var gradientCompensation: array<f32, ${size}>; largestLinear = 0.0;
 for (var row = 0u; row < ROW_COUNT; row++) { if (!included(row)) { continue; } let design = getDesign(row, &means, &scales); var linear = 0.0; for (var column = 0u; column < ${size}u; column++) { linear += design[column] * estimate[column]; }
 if (!isFiniteBits(linear)) { status[statusOffset] = 4u; return; } largestLinear = max(largestLinear, abs(linear));
 let fitted = probability(linear); let derivative = density(linear); let denominator = fitted * (1.0 - fitted); let weight = derivative * derivative / denominator; let score = (response[responseOffset + row] - fitted) * derivative / denominator;
 for (var first = 0u; first < ${size}u; first++) { let term = design[first] * score; let total = gradient[first] + term; if (abs(gradient[first]) >= abs(term)) { gradientCompensation[first] += (gradient[first] - total) + term; } else { gradientCompensation[first] += (term - total) + gradient[first]; } gradient[first] = total; for (var second = 0u; second < ${size}u; second++) { information[first * ${size}u + second] += design[first] * design[second] * weight; } }
 }
 for (var first = 0u; first < ${size}u; first++) { gradient[first] += gradientCompensation[first]; }
 if (!choleskyFactor_${size}(&information)) { status[statusOffset] = 2u; return; }
 if (code == 0u || iteration == ${props.maximumIterations ?? 64}u) { break; }
 choleskySolve_${size}(&information, &gradient); var scale = 1.0; var candidate = estimate; var candidateLikelihood = current; var accepted = false;
 for (var attempt = 0u; attempt < 24u; attempt++) { for (var column = 0u; column < ${size}u; column++) { candidate[column] = estimate[column] + scale * gradient[column]; }
 candidateLikelihood = likelihood(&candidate, &means, &scales);
 // Halve the step until the likelihood does not decrease beyond its noise and the linear predictor stays bounded (|eta| <= 12); an early overshoot is damped, not reported as separation.
 if (candidateLikelihood >= current - noiseScale * (abs(current) + 1.0) && isFiniteBits(candidateLikelihood) && lastLargestLinear <= 12.0) { accepted = true; break; } scale *= 0.5; }
 if (!accepted) { break; } var largest = 0.0; var magnitude = 0.0;
 for (var column = 0u; column < ${size}u; column++) { largest = max(largest, abs(scale * gradient[column])); magnitude = max(magnitude, abs(candidate[column])); }
 estimate = candidate; current = candidateLikelihood; iterations++;
 if (largest <= ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)} * (1.0 + magnitude)) { code = 0u; }
}
// Separation: the fit did not converge and is pinned against the bounded linear predictor.
if (code != 0u && largestLinear > 8.0) { status[statusOffset] = 4u; return; }
var intercept = estimate[0]; var interceptTransform: array<f32, ${size}>; interceptTransform[0] = 1.0;
for (var column = 0u; column < ${columns}u; column++) { let coefficient = estimate[column + 1u] * scales[column]; coefficients[coefficientsOffset + column + 1u] = coefficient; intercept -= means[column] * coefficient; interceptTransform[column + 1u] = -means[column] * scales[column]; if (code == 0u) { errors[errorsOffset + column + 1u] = scales[column] * sqrt(choleskyInverseDiagonal_${size}(&information, column + 1u)); } }
coefficients[coefficientsOffset] = intercept;
if (code == 0u) { var solved = interceptTransform; choleskySolve_${size}(&information, &solved); var variance = 0.0; for (var column = 0u; column < ${size}u; column++) { variance += solved[column] * interceptTransform[column]; } errors[errorsOffset] = sqrt(max(variance, 0.0)); }
summary[summaryOffset + 1u] = current; summary[summaryOffset + 2u] = -2.0 * current; summary[summaryOffset + 3u] = f32(iterations); status[statusOffset] = code;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-predict`,
        operation: 'GPUProbitRegression',
        variant: 'predict',
        invocationCount: rows,
        bindings: [
          ...inputBindings,
          binding('coefficients', props.output.coefficients),
          binding('fitted', props.output.fitted, 'read_write')
        ],
        declarations: common,
        body: `var prediction = getNaN(); if (included(index)) { var linear = coefficients[coefficientsOffset]; for (var column = 0u; column < ${columns}u; column++) { linear += predictors[predictorsOffset + index * ${columns}u + column] * coefficients[coefficientsOffset + column + 1u]; } if (isFiniteBits(linear)) { prediction = probability(linear); } } fitted[fittedOffset + index] = prediction;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-effects`,
        operation: 'GPUProbitRegression',
        variant: 'average-marginal-effects',
        invocationCount: columns,
        constants: {ROW_COUNT: rows},
        bindings: [
          ...inputBindings,
          binding('coefficients', props.output.coefficients),
          binding('effects', props.output.marginalEffects, 'read_write')
        ],
        declarations: `${common}\noverride ROW_COUNT: u32;`,
        body: `var derivative = 0.0; var count = 0u; for (var row = 0u; row < ROW_COUNT; row++) { if (!included(row)) { continue; } var linear = coefficients[coefficientsOffset]; for (var column = 0u; column < ${columns}u; column++) { linear += predictors[predictorsOffset + row * ${columns}u + column] * coefficients[coefficientsOffset + column + 1u]; } derivative += density(linear); count++; } effects[effectsOffset + index] = derivative / f32(count) * coefficients[coefficientsOffset + index + 1u];`
      })
    ];

    if (props.spatialDiagnostics) {
      const diagnostics = props.spatialDiagnostics;
      const residuals = createTransientView(
        graph,
        `${id}-diagnostic-residuals`,
        'float32',
        rows * 5
      );
      validateGraphViewsBelongToGraph(id, graph, [
        ...Object.values(diagnostics.output),
        ...getGPUSpatialWeightsViews(diagnostics.weights)
      ]);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-diagnostic-residuals`,
          operation: 'GPUProbitRegression',
          variant: 'generalized-residuals',
          invocationCount: rows,
          bindings: [
            binding('predictors', props.predictors),
            binding('response', props.response),
            binding('coefficients', props.output.coefficients),
            binding('fitted', props.output.fitted),
            binding('residuals', residuals, 'read_write'),
            ...(props.mask ? [binding('mask', props.mask, 'read', 'u32')] : [])
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
          body: `for (var slot = 0u; slot < 5u; slot++) { residuals[residualsOffset + index * 5u + slot] = getNaN(); }
let probability = fitted[fittedOffset + index]; let observed = response[responseOffset + index];
if (!isFiniteBits(observed) || !isFiniteBits(probability) || !(probability > 0.0 && probability < 1.0) ${props.mask ? '|| mask[maskOffset + index] == 0u' : ''}) { return; }
var linear = coefficients[coefficientsOffset]; for (var column = 0u; column < ${columns}u; column++) { linear += predictors[predictorsOffset + index * ${columns}u + column] * coefficients[coefficientsOffset + column + 1u]; }
let density = 0.3989422804014327 * exp(-0.5 * linear * linear); let variance = probability * (1.0 - probability); let naive = observed - probability;
residuals[residualsOffset + index * 5u] = naive; residuals[residualsOffset + index * 5u + 1u] = density * naive / variance; residuals[residualsOffset + index * 5u + 2u] = naive / sqrt(variance); residuals[residualsOffset + index * 5u + 3u] = variance; residuals[residualsOffset + index * 5u + 4u] = density * density / variance;`
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-spatial-diagnostics`,
          operation: 'GPUProbitRegression',
          variant: 'pinkse-kp-pinkse-slade',
          invocationCount: 1,
          bindings: [
            binding('offsets', diagnostics.weights.offsets, 'read', 'u32'),
            binding('neighbors', diagnostics.weights.neighbors, 'read', 'u32'),
            binding('weights', diagnostics.weights.weights),
            ...(diagnostics.weights.status?.overflow
              ? [binding('overflow', diagnostics.weights.status.overflow, 'read', 'u32')]
              : []),
            binding('residuals', residuals),
            binding('modelStatus', props.output.status, 'read', 'u32'),
            binding('tests', diagnostics.output.tests, 'read_write'),
            binding('diagnosticStatus', diagnostics.output.status, 'read_write', 'u32')
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ERFC_VALUE_WGSL}`,
          body: `diagnosticStatus[diagnosticStatusOffset] = 1u; for (var slot = 0u; slot < 6u; slot++) { tests[testsOffset + slot] = getNaN(); } if (modelStatus[modelStatusOffset] != 0u) { return; }
${diagnostics.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }' : ''}
var count = 0.0; var information = 0.0; var naiveProduct = 0.0; var generalizedProduct = 0.0; var standardizedProduct = 0.0; var trace = 0.0; var varianceTrace = 0.0;
for (var row = 0u; row < ${rows}u; row++) { if (!isFiniteBits(residuals[residualsOffset + row * 5u])) { continue; } count += 1.0; information += residuals[residualsOffset + row * 5u + 4u];
let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u]; if (end < begin || end > ${diagnostics.weights.neighbors.length}u) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }
for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ${rows}u || neighbor == row || !isFiniteBits(weight) || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }
if (!isFiniteBits(residuals[residualsOffset + neighbor * 5u])) { continue; }
var reciprocal = 0.0; for (var other = offsets[offsetsOffset + neighbor]; other < min(offsets[offsetsOffset + neighbor + 1u], ${diagnostics.weights.neighbors.length}u); other++) { if (neighbors[neighborsOffset + other] == row) { reciprocal = weights[weightsOffset + other]; } }
let contribution = weight * (weight + reciprocal); trace += contribution; varianceTrace += contribution * residuals[residualsOffset + row * 5u + 3u] * residuals[residualsOffset + neighbor * 5u + 3u];
naiveProduct += weight * residuals[residualsOffset + row * 5u] * residuals[residualsOffset + neighbor * 5u]; generalizedProduct += weight * residuals[residualsOffset + row * 5u + 1u] * residuals[residualsOffset + neighbor * 5u + 1u]; standardizedProduct += weight * residuals[residualsOffset + row * 5u + 2u] * residuals[residualsOffset + neighbor * 5u + 2u]; } }
if (!(count > 0.0)) { return; } diagnosticStatus[diagnosticStatusOffset] = 3u; information /= count; if (!(trace > 0.0 && varianceTrace > 0.0 && information > 0.0)) { return; }
let pinkse = generalizedProduct * generalizedProduct / (information * information * trace); let kelejian = naiveProduct / sqrt(varianceTrace); let slade = standardizedProduct * standardizedProduct / trace;
tests[testsOffset] = pinkse; tests[testsOffset + 1u] = erfcValue(sqrt(pinkse * 0.5)); tests[testsOffset + 2u] = kelejian; tests[testsOffset + 3u] = erfcValue(abs(kelejian) * 0.7071067811865476); tests[testsOffset + 4u] = slade; tests[testsOffset + 5u] = erfcValue(sqrt(slade * 0.5)); diagnosticStatus[diagnosticStatusOffset] = 0u;`
        })
      );
    }
    return nodes;
  }
}
