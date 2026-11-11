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
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

export type GPUSpatialInteractionProps = {
  id?: string;
  origins: GraphDataView<'uint32'>;
  destinations: GraphDataView<'uint32'>;
  originCount: number;
  destinationCount: number;
  /** Observed nonnegative flows, one per explicitly supplied OD pair. */
  flows: GraphDataView<'float32'>;
  /** One explicit cost/log-distance covariate per pair; exponent is coefficient * covariate. */
  covariate: GraphDataView<'float32'>;
  /** 0 structural zero; 1 observed; 2 missing (excluded, prediction NaN). Omitted means observed. */
  pairState?: GraphDataView<'uint32'>;
  constraint: 'unconstrained' | 'origin' | 'destination' | 'doubly';
  /** Explicit coefficient grid. Best converged candidate by Poisson deviance; ties choose first. */
  coefficients: readonly number[];
  /** grid selects supplied candidates. bounded continuously profiles between exactly two supplied endpoints. */
  calibration?: 'grid' | 'bounded';
  /** Golden-section iterations for bounded profile calibration, 1–64; default 32. */
  maximumCalibrationIterations?: number;
  /** Absolute coefficient bracket width required for convergence; default 1e-5. */
  coefficientTolerance?: number;
  maximumIterations?: number;
  /** Relative marginal tolerance, scaled by max(1, observed marginal). */
  tolerance?: number;
  output: {
    predictions: GraphDataView<'float32'>;
    /** Selected coefficient, deviance, maximum relative marginal error, balancing iterations. */
    summary: GraphDataView<'float32'>;
    /** 0 converged grid optimum; 1 no observed mass; 2 invalid input; 3 no converged candidate; 4 bounded coefficient search exhausted. */
    status: GraphDataView<'uint32'>;
    /** Per candidate: deviance, marginal error, iterations, code (0 converged, 1 exhausted, 2 no mass, 3 invalid, 4 exponent span >80). */
    candidates: GraphDataView<'float32'>;
    /** Optional bounded-search [lower endpoint, upper endpoint, bracket width, converged 0/1]. */
    calibration?: GraphDataView<'float32'>;
    /** Optional signed observed-minus-predicted origin/destination marginal residuals. */
    originResiduals?: GraphDataView<'float32'>;
    destinationResiduals?: GraphDataView<'float32'>;
  };
};

/**
 * Calibrated one-covariate Poisson gravity models over a grid or bounded continuous coefficient interval.
 * All four constraint modes profile nuisance intercepts/marginals; doubly constrained means use
 * bounded iterative proportional fitting. Missing OD pairs are excluded from support and margins,
 * structural zeros predict exactly zero and must have observed flow zero. Each supplied observed
 * pair is an independent count cell; callers must aggregate duplicate OD records explicitly.
 * Golden-section profile search exposes its final bracket and exhaustion; uncertainty is not
 * provided. Candidate exponent spans above 80 are rejected explicitly. Zero marginals remain zero. A failed candidate is never selected as a converged fit.
 */
export class GPUSpatialInteraction implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSpatialInteractionProps) {
    this.id = props.id ?? 'spatial-interaction';
    const pairs = props.flows.length;
    validateCapacity(
      this.id,
      'maximumCalibrationIterations',
      props.maximumCalibrationIterations ?? 32,
      1,
      64
    );
    if (props.output.calibration && props.calibration !== 'bounded')
      throw new Error(`${this.id}: calibration output requires bounded calibration`);
    if (
      !['grid', 'bounded'].includes(props.calibration ?? 'grid') ||
      !(
        Number.isFinite(props.coefficientTolerance ?? 1e-5) &&
        (props.coefficientTolerance ?? 1e-5) > 0
      ) ||
      (props.calibration === 'bounded' &&
        (props.coefficients.length !== 2 || !(props.coefficients[0] < props.coefficients[1])))
    )
      throw new Error(`${this.id}: invalid calibration settings`);
    validateCapacity(this.id, 'pairs', pairs, 1, 4096);
    validateCapacity(this.id, 'originCount', props.originCount, 1, 128);
    validateCapacity(this.id, 'destinationCount', props.destinationCount, 1, 128);
    validateCapacity(this.id, 'coefficient candidates', props.coefficients.length, 1, 128);
    validateCapacity(this.id, 'maximumIterations', props.maximumIterations ?? 256, 1, 4096);
    if (
      !props.coefficients.every(Number.isFinite) ||
      !['unconstrained', 'origin', 'destination', 'doubly'].includes(props.constraint) ||
      !Number.isFinite(props.tolerance ?? 1e-5) ||
      (props.tolerance ?? 1e-5) <= 0
    )
      throw new Error(`${this.id}: invalid model settings`);
    for (const [view, format, length] of [
      [props.origins, 'uint32', pairs],
      [props.destinations, 'uint32', pairs],
      [props.flows, 'float32', pairs],
      [props.covariate, 'float32', pairs],
      [props.pairState, 'uint32', pairs],
      [props.output.predictions, 'float32', pairs],
      [props.output.summary, 'float32', 4],
      [props.output.status, 'uint32', 1],
      [props.output.candidates, 'float32', props.coefficients.length * 4],
      [props.output.calibration, 'float32', 4],
      [props.output.originResiduals, 'float32', props.originCount],
      [props.output.destinationResiduals, 'float32', props.destinationCount]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.origins,
      props.destinations,
      props.flows,
      props.covariate,
      props.pairState
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const pairs = props.flows.length;
    const candidates = props.coefficients.length;
    const bounded = props.calibration === 'bounded';
    const coefficientValues = bounded
      ? createTransientView(graph, `${id}-coefficient-values`, 'float32', 2)
      : undefined;
    const bracket = bounded ? createTransientView(graph, `${id}-bracket`, 'float32', 2) : undefined;
    const predictions = createTransientView(
      graph,
      `${id}-candidate-predictions`,
      'float32',
      pairs * candidates
    );
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'f32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const inputBindings = [
      binding('origins', props.origins, 'read', 'u32'),
      binding('destinations', props.destinations, 'read', 'u32'),
      binding('flows', props.flows),
      binding('covariate', props.covariate),
      ...(props.pairState ? [binding('pairState', props.pairState, 'read', 'u32')] : [])
    ];
    const constants = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\nconst COEFFICIENTS = array<f32, ${candidates}>(${props.coefficients.map(getWGSLFloatLiteral).join(', ')});`;
    validateGraphViewsBelongToGraph(id, graph, [
      props.origins,
      props.destinations,
      props.flows,
      props.covariate,
      props.pairState,
      ...Object.values(props.output)
    ]);
    const createFitNode = (nodeId: string) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: 'GPUSpatialInteraction',
        variant: props.constraint,
        invocationCount: candidates,
        bindings: [
          ...inputBindings,
          binding('predictions', predictions, 'read_write'),
          binding('diagnostics', props.output.candidates, 'read_write'),
          ...(coefficientValues ? [binding('coefficientValues', coefficientValues)] : [])
        ],
        declarations: constants,
        body: `var originTotals: array<f32, ${props.originCount}>; var destinationTotals: array<f32, ${props.destinationCount}>; var observed = 0.0; var largest = -3.402823e38; var smallest = 3.402823e38; var invalid = false;
for (var pair = 0u; pair < ${pairs}u; pair++) {
 let state = ${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'}; let flow = flows[flowsOffset + pair]; let origin = origins[originsOffset + pair]; let destination = destinations[destinationsOffset + pair];
 predictions[predictionsOffset + index * ${pairs}u + pair] = getNaN();
 if (state == 2u) { continue; } if (state > 2u || origin >= ${props.originCount}u || destination >= ${props.destinationCount}u || !isFiniteBits(flow) || flow < 0.0 || (state == 0u && flow != 0.0)) { invalid = true; continue; }
 if (state == 0u) { continue; } let exponent = ${coefficientValues ? 'coefficientValues[coefficientValuesOffset + index]' : 'COEFFICIENTS[index]'} * covariate[covariateOffset + pair]; if (!isFiniteBits(exponent)) { invalid = true; continue; }
 largest = max(largest, exponent); smallest = min(smallest, exponent); observed += flow; originTotals[origin] += flow; destinationTotals[destination] += flow;
}
var code = select(0u, 2u, invalid); if (!(observed > 0.0)) { code = select(1u, 2u, invalid); }
if (code != 0u) { diagnostics[diagnosticsOffset + index * 4u] = getNaN(); diagnostics[diagnosticsOffset + index * 4u + 1u] = getNaN(); diagnostics[diagnosticsOffset + index * 4u + 2u] = 0.0; diagnostics[diagnosticsOffset + index * 4u + 3u] = f32(code + 1u); return; }
if (largest - smallest > 80.0) { diagnostics[diagnosticsOffset + index * 4u] = getNaN(); diagnostics[diagnosticsOffset + index * 4u + 1u] = getNaN(); diagnostics[diagnosticsOffset + index * 4u + 2u] = 0.0; diagnostics[diagnosticsOffset + index * 4u + 3u] = 4.0; return; }
// Each candidate invocation owns one disjoint storage slice; no pair-sized private array.
for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 0u) { predictions[predictionsOffset + index * ${pairs}u + pair] = 0.0; } }
var sum = 0.0;
for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { predictions[predictionsOffset + index * ${pairs}u + pair] = exp(${coefficientValues ? 'coefficientValues[coefficientValuesOffset + index]' : 'COEFFICIENTS[index]'} * covariate[covariateOffset + pair] - largest); sum += predictions[predictionsOffset + index * ${pairs}u + pair]; } }
${props.constraint === 'unconstrained' ? `for (var pair = 0u; pair < ${pairs}u; pair++) { predictions[predictionsOffset + index * ${pairs}u + pair] *= observed / sum; }` : ''}
var error = 0.0; var iterations = 0u; var converged = ${props.constraint === 'unconstrained' ? 'true' : 'false'};
${
  props.constraint === 'unconstrained'
    ? ''
    : `for (var iteration = 0u; iteration < ${props.maximumIterations ?? 256}u; iteration++) {
${
  props.constraint !== 'destination'
    ? `var sums: array<f32, ${props.originCount}>; for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { sums[origins[originsOffset + pair]] += predictions[predictionsOffset + index * ${pairs}u + pair]; } }
for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { let origin = origins[originsOffset + pair]; predictions[predictionsOffset + index * ${pairs}u + pair] *= select(0.0, originTotals[origin] / sums[origin], sums[origin] > 0.0); } }`
    : ''
}
${
  props.constraint !== 'origin'
    ? `var destinationSums: array<f32, ${props.destinationCount}>; for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { destinationSums[destinations[destinationsOffset + pair]] += predictions[predictionsOffset + index * ${pairs}u + pair]; } }
for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { let destination = destinations[destinationsOffset + pair]; predictions[predictionsOffset + index * ${pairs}u + pair] *= select(0.0, destinationTotals[destination] / destinationSums[destination], destinationSums[destination] > 0.0); } }`
    : ''
}
var fittedOrigins: array<f32, ${props.originCount}>; var fittedDestinations: array<f32, ${props.destinationCount}>;
for (var pair = 0u; pair < ${pairs}u; pair++) { if (${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { fittedOrigins[origins[originsOffset + pair]] += predictions[predictionsOffset + index * ${pairs}u + pair]; fittedDestinations[destinations[destinationsOffset + pair]] += predictions[predictionsOffset + index * ${pairs}u + pair]; } }
error = 0.0;
${props.constraint !== 'destination' ? `for (var origin = 0u; origin < ${props.originCount}u; origin++) { error = max(error, abs(fittedOrigins[origin] - originTotals[origin]) / max(1.0, originTotals[origin])); }` : ''}
${props.constraint !== 'origin' ? `for (var destination = 0u; destination < ${props.destinationCount}u; destination++) { error = max(error, abs(fittedDestinations[destination] - destinationTotals[destination]) / max(1.0, destinationTotals[destination])); }` : ''}
iterations++; if (error <= ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)}) { converged = true; break; }
}`
}
var deviance = 0.0;
for (var pair = 0u; pair < ${pairs}u; pair++) { let state = ${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'};
if (state == 2u) { continue; }
if (state == 1u) { let flow = flows[flowsOffset + pair]; let prediction = predictions[predictionsOffset + index * ${pairs}u + pair]; if (!isFiniteBits(prediction) || prediction < 0.0 || (flow > 0.0 && !(prediction > 0.0))) { converged = false; }
var term = prediction; if (flow > 0.0) { term = flow * (log(flow) - log(prediction)) - flow + prediction; } deviance += 2.0 * term; }
}
diagnostics[diagnosticsOffset + index * 4u] = max(0.0, deviance);
diagnostics[diagnosticsOffset + index * 4u + 1u] = error;
diagnostics[diagnosticsOffset + index * 4u + 2u] = f32(iterations);
diagnostics[diagnosticsOffset + index * 4u + 3u] = select(1.0, 0.0, converged && isFiniteBits(deviance));`
      });
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (coefficientValues && bracket) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-initialize-calibration`,
          operation: 'GPUSpatialInteraction',
          variant: 'initialize-bounded-profile',
          invocationCount: 1,
          bindings: [
            binding('coefficients', coefficientValues, 'read_write'),
            binding('bracket', bracket, 'read_write')
          ],
          body: `let lower = ${getWGSLFloatLiteral(props.coefficients[0])}; let upper = ${getWGSLFloatLiteral(props.coefficients[1])}; bracket[bracketOffset] = lower; bracket[bracketOffset + 1u] = upper; coefficients[coefficientsOffset] = lower + 0.3819660112501051 * (upper - lower); coefficients[coefficientsOffset + 1u] = lower + 0.6180339887498949 * (upper - lower);`
        })
      );
      for (let iteration = 0; iteration < (props.maximumCalibrationIterations ?? 32); iteration++) {
        nodes.push(createFitNode(`${id}-profile-${iteration}`));
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-refine-${iteration}`,
            operation: 'GPUSpatialInteraction',
            variant: 'golden-section-profile',
            invocationCount: 1,
            bindings: [
              binding('coefficients', coefficientValues, 'read_write'),
              binding('bracket', bracket, 'read_write'),
              binding('candidates', props.output.candidates)
            ],
            body: `var lower = bracket[bracketOffset]; var upper = bracket[bracketOffset + 1u]; if (upper - lower <= ${getWGSLFloatLiteral(props.coefficientTolerance ?? 1e-5)}) { return; }
let firstValid = candidates[candidatesOffset + 3u] == 0.0; let secondValid = candidates[candidatesOffset + 7u] == 0.0; if (!firstValid && !secondValid) { return; }
if (firstValid && (!secondValid || candidates[candidatesOffset] <= candidates[candidatesOffset + 4u])) { upper = coefficients[coefficientsOffset + 1u]; } else { lower = coefficients[coefficientsOffset]; }
bracket[bracketOffset] = lower; bracket[bracketOffset + 1u] = upper; coefficients[coefficientsOffset] = lower + 0.3819660112501051 * (upper - lower); coefficients[coefficientsOffset + 1u] = lower + 0.6180339887498949 * (upper - lower);`
          })
        );
      }
    }
    nodes.push(createFitNode(`${id}-fit`));
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: 'GPUSpatialInteraction',
        variant: 'select-grid-optimum',
        invocationCount: 1,
        bindings: [
          binding('predictions', predictions),
          binding('diagnostics', props.output.candidates),
          binding('output', props.output.predictions, 'read_write'),
          binding('summary', props.output.summary, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32'),
          ...(coefficientValues ? [binding('coefficientValues', coefficientValues)] : [])
        ],
        declarations: constants,
        body: `var chosen = 0xffffffffu; var best = 3.402823e38; var code = 3u;
for (var candidate = 0u; candidate < ${candidates}u; candidate++) { let candidateCode = diagnostics[diagnosticsOffset + candidate * 4u + 3u]; let deviance = diagnostics[diagnosticsOffset + candidate * 4u]; if (candidateCode == 3.0) { code = 2u; } else if (candidateCode == 2.0 && code != 2u) { code = 1u; }
if (candidateCode == 0.0 && deviance < best) { best = deviance; chosen = candidate; } }
for (var pair = 0u; pair < ${pairs}u; pair++) { var value = getNaN(); if (chosen != 0xffffffffu) { value = predictions[predictionsOffset + chosen * ${pairs}u + pair]; } output[outputOffset + pair] = value; }
for (var slot = 0u; slot < 4u; slot++) { summary[summaryOffset + slot] = getNaN(); }
if (chosen != 0xffffffffu) { code = 0u; summary[summaryOffset] = ${coefficientValues ? 'coefficientValues[coefficientValuesOffset + chosen]' : 'COEFFICIENTS[chosen]'}; summary[summaryOffset + 1u] = best; summary[summaryOffset + 2u] = diagnostics[diagnosticsOffset + chosen * 4u + 1u]; summary[summaryOffset + 3u] = diagnostics[diagnosticsOffset + chosen * 4u + 2u]; }
status[statusOffset] = code;`
      })
    );
    if (bracket) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-calibration-status`,
          operation: 'GPUSpatialInteraction',
          variant: 'bounded-profile-status',
          invocationCount: 1,
          bindings: [
            binding('bracket', bracket),
            binding('status', props.output.status, 'read_write', 'u32'),
            ...(props.output.calibration
              ? [binding('calibration', props.output.calibration, 'read_write')]
              : [])
          ],
          body: `let lower = bracket[bracketOffset]; let upper = bracket[bracketOffset + 1u]; let converged = upper - lower <= ${getWGSLFloatLiteral(props.coefficientTolerance ?? 1e-5)};
if (!converged && status[statusOffset] == 0u) { status[statusOffset] = 4u; }
${props.output.calibration ? 'calibration[calibrationOffset] = lower; calibration[calibrationOffset + 1u] = upper; calibration[calibrationOffset + 2u] = upper - lower; calibration[calibrationOffset + 3u] = select(0.0, 1.0, converged);' : ''}`
        })
      );
    }
    for (const [name, view, groups, ids] of [
      ['origin', props.output.originResiduals, props.originCount, props.origins],
      ['destination', props.output.destinationResiduals, props.destinationCount, props.destinations]
    ] as const)
      if (view)
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-${name}-residuals`,
            operation: 'GPUSpatialInteraction',
            variant: `${name}-residuals`,
            invocationCount: groups,
            bindings: [
              binding('ids', ids, 'read', 'u32'),
              binding('flows', props.flows),
              binding('predictions', props.output.predictions),
              ...(props.pairState ? [binding('pairState', props.pairState, 'read', 'u32')] : []),
              binding('residuals', view, 'read_write'),
              binding('status', props.output.status, 'read', 'u32')
            ],
            declarations: GET_NAN_WGSL,
            body: `var residual = 0.0; for (var pair = 0u; pair < ${pairs}u; pair++) { if (ids[idsOffset + pair] == index && ${props.pairState ? 'pairState[pairStateOffset + pair]' : '1u'} == 1u) { residual += flows[flowsOffset + pair] - predictions[predictionsOffset + pair]; } }
residuals[residualsOffset + index] = select(getNaN(), residual, status[statusOffset] == 0u);`
          })
        );
    return nodes;
  }
}
