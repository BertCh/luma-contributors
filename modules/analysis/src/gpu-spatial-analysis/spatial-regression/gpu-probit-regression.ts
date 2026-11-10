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
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateCommandNodeBudget,
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
import {
  getRegressionTileLayout,
  getRowTileSumDispatchShapes,
  getRowTileSumNodes,
  REGRESSION_TILE_MAXIMUM_ROW_COUNT
} from './regression-row-tiles';

const OPERATION = 'GPUProbitRegression';
/** Largest predictor count: the 12 x 12 information matrix is factored in one invocation. */
const MAXIMUM_PREDICTOR_COUNT = 11;
/** Largest `maximumIterations`: bounds the unrolled Fisher-scoring nodes (6 per iteration). */
const MAXIMUM_ITERATION_COUNT = 256;
const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_TOLERANCE = 1e-5;
/** Step halvings evaluated by the line search: steps 1, 1/2, ..., 2^-23 (as the serial search). */
const LINE_SEARCH_STEP_COUNT = 24;
/** Float32 words of one diagnostic residual row: naive, generalized, standardized, variance, weight. */
const DIAGNOSTIC_RESIDUAL_STRIDE = 5;

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
 * f32; probabilities are floored at 1e-7 for numerical tails. The log likelihood and score are
 * row-tile sums merged with compensated (Neumaier) summation, and the line search tolerates
 * likelihood noise relative to |loglik| and sqrt(rows). Trial steps are halved until the
 * likelihood does not decrease and |linear predictor| <= 12, so an early overshoot is damped
 * rather than rejected. All-one/all-zero samples, and fits that stay unconverged while pinned
 * against that bound, are reported as separated (status 4), never successful estimates.
 *
 * Every row pass is parallel (the row-tile pattern of `GPUOrdinaryLeastSquares`: one invocation
 * per tile of consecutive rows, then a compensated merge in tile order, so results are bitwise
 * reproducible). Each Fisher-scoring iteration is unrolled into six nodes:
 * 1. tiles and merge of the score `X'(y - p) phi / (p (1 - p))` and the expected information
 *    `X' W X` at the current coefficients (and the largest |linear predictor|);
 * 2. one invocation that factors the information, tests convergence and the iteration limit, and
 *    solves for the Fisher direction (a small `p x p` solve, no row loop);
 * 3. tiles and merge of the log likelihood at all 24 halved steps `1, 1/2, ..., 2^-23` in one pass;
 * 4. one invocation that accepts the largest acceptable step, exactly as the serial backtracking
 *    did, and updates the coefficients.
 * The one-invocation nodes own a GPU dispatch gate (`createWGSLDispatchGate`): once the fit has
 * converged, failed or exhausted its iterations, every remaining iteration launches nothing.
 * `estimatedCommandNodeCount` is `6 * maximumIterations` plus a fixed prologue and epilogue.
 * Rows are limited by float32 counts (`2^24`) and by the predictor (and, with diagnostics, the
 * diagnostic residual) bindings of the device. Row and tile counts are pipeline overrides, so the
 * row passes of models that differ only in row count share shader modules.
 * This is not a latent spatial-lag probit. Spatial residual diagnostics require explicit downstream
 * residual construction and spatial-statistic inference with the appropriate model assumptions.
 */
export class GPUProbitRegression implements GPUCommandNodeProducer {
  readonly id: string;
  /** Number of command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;
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
    validateCapacity(this.id, 'rows', rows, 1, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
    validateCapacity(this.id, 'predictorCount', columns, 1, MAXIMUM_PREDICTOR_COUNT);
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateCapacity(this.id, 'maximumIterations', maximumIterations, 1, MAXIMUM_ITERATION_COUNT);
    if (
      !Number.isFinite(props.tolerance ?? DEFAULT_TOLERANCE) ||
      (props.tolerance ?? DEFAULT_TOLERANCE) <= 0
    )
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
    // Prologue: two tile passes (4 nodes), initialization, initial likelihood (2). Per iteration:
    // information tiles and merge, solve, line-search tiles and merge, decide; the last iteration
    // stops after its solve. Epilogue: finish, predict, effect tiles (2), effects, and with
    // diagnostics the residual, tile (2) and finish nodes.
    this.estimatedCommandNodeCount =
      7 + 6 * maximumIterations + 3 + 5 + (props.spatialDiagnostics ? 4 : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.response.length;
    const columns = props.predictorCount;
    const size = columns + 1;
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    // The predictor view (and the diagnostic residual rows) are the largest bindings.
    validateCapacity(
      id,
      'rows',
      rows,
      1,
      Math.min(
        REGRESSION_TILE_MAXIMUM_ROW_COUNT,
        getStorageBindingElementLimit(
          graph.device,
          4 * Math.max(columns, props.spatialDiagnostics ? DIAGNOSTIC_RESIDUAL_STRIDE : 1)
        )
      )
    );
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.mask,
      ...Object.values(props.output)
    ]);
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
    const layout = getProbitStateLayout(columns);
    const tileLayout = getRegressionTileLayout(rows);
    const state = createTransientView(graph, `${id}-state`, 'float32', layout.length);
    const informationSumCount = size + (size * (size + 1)) / 2;
    const [tileShape, informationMergeShape] = getRowTileSumDispatchShapes(
      tileLayout,
      informationSumCount + 2
    );
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: OPERATION,
      shapes: [tileShape, informationMergeShape, tileShape, [2 * LINE_SEARCH_STEP_COUNT]]
    });
    const partial = (name: string, termCount: number) =>
      createTransientView(
        graph,
        `${id}-${name}-partials`,
        'float32',
        tileLayout.tileCount * termCount
      );
    const informationPartial = partial('information', informationSumCount + 2);
    const lineSearchPartial = partial('line-search', 2 * LINE_SEARCH_STEP_COUNT);
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ERFC_VALUE_WGSL}
fn included(row: u32) -> bool {
 if (${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteBits(response[responseOffset + row])) { return false; }
 for (var column = 0u; column < ${columns}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${columns}u + column])) { return false; } } return true;
}
fn probability(value: f32) -> f32 { return clamp(0.5 * erfcValue(-value * 0.7071067811865475), 1e-7, 1.0 - 1e-7); }
fn density(value: f32) -> f32 { return 0.3989422804014327 * exp(-0.5 * value * value); }
`;
    const stateWGSL = `const C: u32 = ${columns}u;
const P: u32 = ${size}u;
${Object.entries(layout)
  .map(
    ([name, slot]) =>
      `const S_${name.replace(/[A-Z]/g, letter => `_${letter}`).toUpperCase()}: u32 = ${slot}u;`
  )
  .join('\n')}
fn stateAt(slot: u32) -> f32 { return state[stateOffset + slot]; }`;
    // Centered, scaled design row with the intercept first.
    const designWGSL = `fn getDesign(row: u32) -> array<f32, ${size}> { var design: array<f32, ${size}>; design[0] = 1.0;
 for (var column = 0u; column < C; column++) { design[column + 1u] = (predictors[predictorsOffset + row * C + column] - stateAt(S_MEANS + column)) * stateAt(S_SCALES + column); } return design; }`;
    const rowDeclarations = `${common}\n${stateWGSL}\n${designWGSL}`;
    const stateBindings = [...inputBindings, binding('state', state)];
    // Log likelihood at `stepCount` halved steps from the current coefficients along the stored
    // direction (largest step first), and the largest |linear predictor| of each step.
    const likelihoodWGSL = (stepCount: number) => `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; }
 let design = getDesign(row); let outcome = response[responseOffset + row];
 var estimate: array<f32, ${size}>; var direction: array<f32, ${size}>;
 for (var column = 0u; column < P; column++) { estimate[column] = stateAt(S_ESTIMATE + column); direction[column] = stateAt(S_DIRECTION + column); }
 var scale = 1.0;
 for (var step = 0u; step < ${stepCount}u; step++) {
  var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * (estimate[column] + scale * direction[column]); }
  (*terms)[${stepCount}u + step] = max((*terms)[${stepCount}u + step], abs(linear));
  let fitted = probability(linear);
  (*terms)[step] += outcome * log(fitted) + (1.0 - outcome) * log(1.0 - fitted);
  scale *= 0.5;
 }
}`;
    const tileSums = (
      name: string,
      sumCount: number,
      maximumCount: number,
      declarations: string,
      outputSlot: number,
      partialView: GraphDataView<'float32'>,
      bindings: readonly WGSLKernelBinding[] = stateBindings,
      gated = false
    ) =>
      getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name.replace(/-\d+/g, ''),
        rowCount: rows,
        layout: tileLayout,
        sumCount,
        maximumCount,
        bindings,
        declarations,
        partial: partialView,
        output: state,
        outputSlot,
        ...(gated
          ? {
              tileGate: gate.getGate(
                name.startsWith('iteration') && name.endsWith('information') ? 0 : 2
              ),
              mergeGate: gate.getGate(name.endsWith('information') ? 1 : 3)
            }
          : {})
      });
    const singleNode = (
      name: string,
      bindings: WGSLKernelBinding[],
      declarations: string,
      body: string
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name.replace(/iteration-\d+-/, ''),
        invocationCount: 1,
        bindings,
        declarations,
        body
      });
    const tolerance = getWGSLFloatLiteral(props.tolerance ?? DEFAULT_TOLERANCE);

    const nodes: GPUCommandNode<Parameters>[] = [
      // Complete-row count, successes, column sums and a nonbinary-response flag.
      ...tileSums(
        'prepare',
        columns + 2,
        1,
        `${common}
const C: u32 = ${columns}u;
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let outcome = response[responseOffset + row];
 if (outcome != 0.0 && outcome != 1.0) { (*terms)[SUM_COUNT] = 1.0; }
 (*terms)[0] += 1.0; (*terms)[1] += outcome;
 for (var column = 0u; column < C; column++) { (*terms)[2u + column] += predictors[predictorsOffset + row * C + column]; }
}`,
        layout.prepare,
        partial('prepare', columns + 3),
        inputBindings
      ),
      // Centered sums of squares about the column means (two-pass variance).
      ...tileSums(
        'variance',
        columns,
        0,
        `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let count = stateAt(S_PREPARE);
 for (var column = 0u; column < C; column++) { let difference = predictors[predictorsOffset + row * C + column] - stateAt(S_PREPARE + 2u + column) / count; (*terms)[column] += difference * difference; }
}`,
        layout.variance,
        partial('variance', columns),
        stateBindings
      ),
      singleNode(
        'initialize',
        [binding('state', state, 'read_write'), gate.binding],
        stateWGSL,
        `${gate.disarmWGSL}
state[stateOffset + S_DONE] = 1.0; state[stateOffset + S_CODE] = 1.0; state[stateOffset + S_COUNTED] = 0.0;
state[stateOffset + S_ITERATIONS] = 0.0; state[stateOffset + S_ITERATION] = 0.0; state[stateOffset + S_LARGEST] = 0.0;
for (var column = 0u; column < P; column++) { state[stateOffset + S_ESTIMATE + column] = 0.0; state[stateOffset + S_DIRECTION + column] = 0.0; }
let count = stateAt(S_PREPARE); let successes = stateAt(S_PREPARE + 1u);
for (var column = 0u; column < C; column++) { state[stateOffset + S_MEANS + column] = stateAt(S_PREPARE + 2u + column) / count; state[stateOffset + S_SCALES + column] = 0.0; }
if (stateAt(S_PREPARE + C + 2u) != 0.0) { state[stateOffset + S_CODE] = 4.0; return; }
state[stateOffset + S_COUNTED] = 1.0;
if (count <= f32(P)) { return; } if (successes == 0.0 || successes == count) { state[stateOffset + S_CODE] = 4.0; return; }
for (var column = 0u; column < C; column++) { let variance = stateAt(S_VARIANCE + column); if (!(variance > 0.0)) { state[stateOffset + S_CODE] = 2.0; return; } state[stateOffset + S_SCALES + column] = sqrt(count / variance); }
// Likelihood noise floor: the compensated sum is exact to about eps * |loglik|, the per-row log/erfc errors add roughly sqrt(count) of that.
state[stateOffset + S_NOISE] = 4.0 * 1.1920929e-7 * max(1.0, sqrt(count));
state[stateOffset + S_CODE] = 3.0; state[stateOffset + S_DONE] = 0.0;
${gate.armWGSL}`
      ),
      // Log likelihood at the zero start (one step of the zero direction).
      ...tileSums(
        'initial-likelihood',
        1,
        1,
        likelihoodWGSL(1),
        layout.current,
        partial('initial-likelihood', 2)
      )
    ];

    for (let iteration = 0; iteration <= maximumIterations; iteration++) {
      const prefix = `iteration-${iteration}`;
      nodes.push(
        ...tileSums(
          `${prefix}-information`,
          informationSumCount,
          2,
          `${rowDeclarations}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } let design = getDesign(row);
 var linear = 0.0; for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); }
 if (!isFiniteBits(linear)) { (*terms)[SUM_COUNT + 1u] = 1.0; return; }
 (*terms)[SUM_COUNT] = max((*terms)[SUM_COUNT], abs(linear));
 let fitted = probability(linear); let derivative = density(linear); let denominator = fitted * (1.0 - fitted); let weight = derivative * derivative / denominator; let score = (response[responseOffset + row] - fitted) * derivative / denominator;
 var slot = P;
 for (var first = 0u; first < P; first++) { (*terms)[first] += design[first] * score; for (var second = first; second < P; second++) { (*terms)[slot] += design[first] * design[second] * weight; slot++; } }
}`,
          layout.information,
          informationPartial,
          stateBindings,
          true
        ),
        singleNode(
          `${prefix}-solve`,
          [binding('state', state, 'read_write'), gate.binding],
          `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${stateWGSL}\n${getCholeskyWGSL(size)}`,
          `if (stateAt(S_DONE) != 0.0) { return; }
let iteration = u32(stateAt(S_ITERATION)); state[stateOffset + S_ITERATION] = f32(iteration + 1u);
let informationSums = ${informationSumCount}u;
if (stateAt(S_INFORMATION + informationSums + 1u) != 0.0) { state[stateOffset + S_CODE] = 4.0; state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
state[stateOffset + S_LARGEST] = stateAt(S_INFORMATION + informationSums);
var information: array<f32, ${size * size}>; var gradient: array<f32, ${size}>; var slot = P;
for (var first = 0u; first < P; first++) { gradient[first] = stateAt(S_INFORMATION + first); for (var second = first; second < P; second++) { let value = stateAt(S_INFORMATION + slot); slot++; information[first * P + second] = value; information[second * P + first] = value; } }
if (!choleskyFactor_${size}(&information)) { state[stateOffset + S_CODE] = 2.0; state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
for (var entry = 0u; entry < ${size * size}u; entry++) { state[stateOffset + S_FACTOR + entry] = information[entry]; }
if (stateAt(S_CODE) == 0.0 || iteration == ${maximumIterations}u) { state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
choleskySolve_${size}(&information, &gradient);
for (var column = 0u; column < P; column++) { state[stateOffset + S_DIRECTION + column] = gradient[column]; }`
        )
      );
      if (iteration === maximumIterations) {
        break;
      }
      nodes.push(
        ...tileSums(
          `${prefix}-line-search`,
          LINE_SEARCH_STEP_COUNT,
          LINE_SEARCH_STEP_COUNT,
          likelihoodWGSL(LINE_SEARCH_STEP_COUNT),
          layout.lineSearch,
          lineSearchPartial,
          stateBindings,
          true
        ),
        singleNode(
          `${prefix}-decide`,
          [binding('state', state, 'read_write'), gate.binding],
          `${IS_FINITE_BITS_WGSL}\n${stateWGSL}`,
          `if (stateAt(S_DONE) != 0.0) { return; }
let current = stateAt(S_CURRENT); let noiseScale = stateAt(S_NOISE);
var scale = 1.0; var accepted = false; var chosen = 0u;
for (var attempt = 0u; attempt < ${LINE_SEARCH_STEP_COUNT}u; attempt++) {
 let candidateLikelihood = stateAt(S_LINE_SEARCH + attempt); let lastLargestLinear = stateAt(S_LINE_SEARCH + ${LINE_SEARCH_STEP_COUNT}u + attempt);
 // Halve the step until the likelihood does not decrease beyond its noise and the linear predictor stays bounded (|eta| <= 12); an early overshoot is damped, not reported as separation.
 if (candidateLikelihood >= current - noiseScale * (abs(current) + 1.0) && isFiniteBits(candidateLikelihood) && lastLargestLinear <= 12.0) { accepted = true; chosen = attempt; break; } scale *= 0.5; }
if (!accepted) { state[stateOffset + S_DONE] = 1.0; ${gate.disarmWGSL} return; }
var largest = 0.0; var magnitude = 0.0;
for (var column = 0u; column < P; column++) { let step = scale * stateAt(S_DIRECTION + column); let candidate = stateAt(S_ESTIMATE + column) + step; largest = max(largest, abs(step)); magnitude = max(magnitude, abs(candidate)); state[stateOffset + S_ESTIMATE + column] = candidate; }
state[stateOffset + S_CURRENT] = stateAt(S_LINE_SEARCH + chosen); state[stateOffset + S_ITERATIONS] = stateAt(S_ITERATIONS) + 1.0;
if (largest <= ${tolerance} * (1.0 + magnitude)) { state[stateOffset + S_CODE] = 0.0; }`
        )
      );
    }

    nodes.push(
      singleNode(
        'finish',
        [
          binding('state', state),
          binding('coefficients', props.output.coefficients, 'read_write'),
          binding('errors', props.output.standardErrors, 'read_write'),
          binding('summary', props.output.summary, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32')
        ],
        `${GET_NAN_WGSL}\n${stateWGSL}\n${getCholeskyWGSL(size)}`,
        `for (var column = 0u; column < P; column++) { coefficients[coefficientsOffset + column] = getNaN(); errors[errorsOffset + column] = getNaN(); }
for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); }
let code = u32(stateAt(S_CODE)); status[statusOffset] = code;
if (stateAt(S_COUNTED) == 0.0) { return; }
summary[summaryOffset] = stateAt(S_PREPARE); summary[summaryOffset + 3u] = 0.0;
if (code == 1u || code == 2u || code == 4u) { return; }
// Separation: the fit did not converge and is pinned against the bounded linear predictor.
let largestLinear = stateAt(S_LARGEST);
if (code != 0u && largestLinear > 8.0) { status[statusOffset] = 4u; return; }
var information: array<f32, ${size * size}>; for (var entry = 0u; entry < ${size * size}u; entry++) { information[entry] = stateAt(S_FACTOR + entry); }
var intercept = stateAt(S_ESTIMATE); var interceptTransform: array<f32, ${size}>; interceptTransform[0] = 1.0;
for (var column = 0u; column < C; column++) { let scale = stateAt(S_SCALES + column); let mean = stateAt(S_MEANS + column); let coefficient = stateAt(S_ESTIMATE + column + 1u) * scale; coefficients[coefficientsOffset + column + 1u] = coefficient; intercept -= mean * coefficient; interceptTransform[column + 1u] = -mean * scale; if (code == 0u) { errors[errorsOffset + column + 1u] = scale * sqrt(choleskyInverseDiagonal_${size}(&information, column + 1u)); } }
coefficients[coefficientsOffset] = intercept;
if (code == 0u) { var solved = interceptTransform; choleskySolve_${size}(&information, &solved); var variance = 0.0; for (var column = 0u; column < P; column++) { variance += solved[column] * interceptTransform[column]; } errors[errorsOffset] = sqrt(max(variance, 0.0)); }
let current = stateAt(S_CURRENT);
summary[summaryOffset + 1u] = current; summary[summaryOffset + 2u] = -2.0 * current; summary[summaryOffset + 3u] = stateAt(S_ITERATIONS);`
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-predict`,
        operation: OPERATION,
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
      // Average marginal effects: mean density of the linear predictor times each slope.
      ...tileSums(
        'effects',
        2,
        0,
        `${common}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!included(row)) { return; } var linear = coefficients[coefficientsOffset];
 for (var column = 0u; column < ${columns}u; column++) { linear += predictors[predictorsOffset + row * ${columns}u + column] * coefficients[coefficientsOffset + column + 1u]; }
 (*terms)[0] += density(linear); (*terms)[1] += 1.0;
}`,
        layout.effects,
        partial('effects', 2),
        [...inputBindings, binding('coefficients', props.output.coefficients)]
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-effects`,
        operation: OPERATION,
        variant: 'average-marginal-effects',
        invocationCount: columns,
        bindings: [
          binding('state', state),
          binding('coefficients', props.output.coefficients),
          binding('effects', props.output.marginalEffects, 'read_write')
        ],
        declarations: stateWGSL,
        body: `effects[effectsOffset + index] = stateAt(S_EFFECTS) / stateAt(S_EFFECTS + 1u) * coefficients[coefficientsOffset + index + 1u];`
      })
    );

    if (props.spatialDiagnostics) {
      const diagnostics = props.spatialDiagnostics;
      const neighborCount = diagnostics.weights.neighbors.length;
      const residuals = createTransientView(
        graph,
        `${id}-diagnostic-residuals`,
        'float32',
        rows * DIAGNOSTIC_RESIDUAL_STRIDE
      );
      validateGraphViewsBelongToGraph(id, graph, [
        ...Object.values(diagnostics.output),
        ...getGPUSpatialWeightsViews(diagnostics.weights)
      ]);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-diagnostic-residuals`,
          operation: OPERATION,
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
        }),
        // One invocation per row visits its own CSR neighbors: O(nnz) work in row tiles.
        // Sums: count, information, trace, variance trace, naive, generalized and standardized
        // cross-products; maximum: the invalid-weights flag.
        ...tileSums(
          'diagnostics',
          7,
          1,
          `${IS_FINITE_BITS_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
 if (!isFiniteBits(residuals[residualsOffset + row * 5u])) { return; }
 (*terms)[0] += 1.0; (*terms)[1] += residuals[residualsOffset + row * 5u + 4u];
 let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u]; if (end < begin || end > ${neighborCount}u) { (*terms)[SUM_COUNT] = 1.0; return; }
 for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ROW_COUNT || neighbor == row || !isFiniteBits(weight) || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { (*terms)[SUM_COUNT] = 1.0; return; }
 if (!isFiniteBits(residuals[residualsOffset + neighbor * 5u])) { continue; }
 var reciprocal = 0.0; for (var other = offsets[offsetsOffset + neighbor]; other < min(offsets[offsetsOffset + neighbor + 1u], ${neighborCount}u); other++) { if (neighbors[neighborsOffset + other] == row) { reciprocal = weights[weightsOffset + other]; } }
 let contribution = weight * (weight + reciprocal); (*terms)[2] += contribution; (*terms)[3] += contribution * residuals[residualsOffset + row * 5u + 3u] * residuals[residualsOffset + neighbor * 5u + 3u];
 (*terms)[4] += weight * residuals[residualsOffset + row * 5u] * residuals[residualsOffset + neighbor * 5u]; (*terms)[5] += weight * residuals[residualsOffset + row * 5u + 1u] * residuals[residualsOffset + neighbor * 5u + 1u]; (*terms)[6] += weight * residuals[residualsOffset + row * 5u + 2u] * residuals[residualsOffset + neighbor * 5u + 2u]; }
}`,
          layout.diagnostics,
          partial('diagnostics', 8),
          [
            binding('offsets', diagnostics.weights.offsets, 'read', 'u32'),
            binding('neighbors', diagnostics.weights.neighbors, 'read', 'u32'),
            binding('weights', diagnostics.weights.weights),
            binding('residuals', residuals)
          ]
        ),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-spatial-diagnostics`,
          operation: OPERATION,
          variant: 'pinkse-kp-pinkse-slade',
          invocationCount: 1,
          bindings: [
            binding('state', state),
            ...(diagnostics.weights.status?.overflow
              ? [binding('overflow', diagnostics.weights.status.overflow, 'read', 'u32')]
              : []),
            binding('modelStatus', props.output.status, 'read', 'u32'),
            binding('tests', diagnostics.output.tests, 'read_write'),
            binding('diagnosticStatus', diagnostics.output.status, 'read_write', 'u32')
          ],
          declarations: `${GET_NAN_WGSL}\n${ERFC_VALUE_WGSL}\n${stateWGSL}`,
          body: `diagnosticStatus[diagnosticStatusOffset] = 1u; for (var slot = 0u; slot < 6u; slot++) { tests[testsOffset + slot] = getNaN(); } if (modelStatus[modelStatusOffset] != 0u) { return; }
${diagnostics.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }' : ''}
if (stateAt(S_DIAGNOSTICS + 7u) != 0.0) { diagnosticStatus[diagnosticStatusOffset] = 2u; return; }
let count = stateAt(S_DIAGNOSTICS); var information = stateAt(S_DIAGNOSTICS + 1u); let trace = stateAt(S_DIAGNOSTICS + 2u); let varianceTrace = stateAt(S_DIAGNOSTICS + 3u);
let naiveProduct = stateAt(S_DIAGNOSTICS + 4u); let generalizedProduct = stateAt(S_DIAGNOSTICS + 5u); let standardizedProduct = stateAt(S_DIAGNOSTICS + 6u);
if (!(count > 0.0)) { return; } diagnosticStatus[diagnosticStatusOffset] = 3u; information /= count; if (!(trace > 0.0 && varianceTrace > 0.0 && information > 0.0)) { return; }
let pinkse = generalizedProduct * generalizedProduct / (information * information * trace); let kelejian = naiveProduct / sqrt(varianceTrace); let slade = standardizedProduct * standardizedProduct / trace;
tests[testsOffset] = pinkse; tests[testsOffset + 1u] = erfcValue(sqrt(pinkse * 0.5)); tests[testsOffset + 2u] = kelejian; tests[testsOffset + 3u] = erfcValue(abs(kelejian) * 0.7071067811865476); tests[testsOffset + 4u] = slade; tests[testsOffset + 5u] = erfcValue(sqrt(slade * 0.5)); diagnosticStatus[diagnosticStatusOffset] = 0u;`
        })
      );
    }
    return nodes;
  }
}

/** Float32 word offsets of the probit state buffer. */
function getProbitStateLayout(columns: number) {
  const size = columns + 1;
  let length = 0;
  const take = (count: number) => {
    const slot = length;
    length += count;
    return slot;
  };
  return {
    /** Complete-row count, successes, column sums, nonbinary flag. */
    prepare: take(columns + 3),
    /** Centered column sums of squares. */
    variance: take(columns),
    means: take(columns),
    scales: take(columns),
    /** Status code in progress (as the serial fit), done flag, whether the count is reported. */
    code: take(1),
    done: take(1),
    counted: take(1),
    /** Accepted steps, unrolled iterations entered, current log likelihood and its scratch maximum. */
    iterations: take(1),
    iteration: take(1),
    current: take(2),
    noise: take(1),
    /** Largest |linear predictor| of the latest information pass. */
    largest: take(1),
    estimate: take(size),
    direction: take(size),
    factor: take(size * size),
    /** Score, packed information, largest |linear predictor|, nonfinite flag. */
    information: take(size + (size * (size + 1)) / 2 + 2),
    lineSearch: take(2 * LINE_SEARCH_STEP_COUNT),
    effects: take(2),
    diagnostics: take(8),
    length
  };
}
