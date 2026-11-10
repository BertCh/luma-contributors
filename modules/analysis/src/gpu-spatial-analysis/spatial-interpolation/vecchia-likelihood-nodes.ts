// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFixedPointClearNode,
  createFixedPointScaleNodes,
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPackedPredictionAccessors} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Output ports of one Vecchia log-likelihood evaluation, shared by the spatial and space-time forms. */
export type GPUVecchiaLikelihoodOutput = {
  /** One row per site: `[conditional mean, conditional variance d_i, pivot ratio, retained]`. */
  conditionals: GraphDataView<'float32x4'>;
  /** One word per site: solve status, 0 success, 1 invalid or insufficient data, 2 singular. */
  conditionalStatus: GraphDataView<'uint32'>;
  /** One row per site: `-1/2 (log 2 pi + log d_i + r_i^2 / d_i)`; NaN for an excluded or failed site. */
  siteTerms: GraphDataView<'float32'>;
  /** One word per site: 0 used, 1 excluded, 2 solve failed, 3 variance nonpositive or non-finite. */
  siteStatus: GraphDataView<'uint32'>;
  /**
   * Two elements `[high, low]`, an unevaluated float32 sum holding the log-likelihood total.
   * Both are NaN when any site failed or no site was used.
   */
  logLikelihood: GraphDataView<'float32'>;
  /** Three words `[failedSiteCount, excludedSiteCount, usedSiteCount]`. Any failure fails the likelihood. */
  status: GraphDataView<'uint32'>;
};

/**
 * Sums `cellCount` flat f32 terms per site (read from `terms`, `rowCount * cellCount` rows, non-finite
 * entries ignored) into `totals`, two elements `[high, low]` per cell, an unevaluated float32 sum.
 * Every cell is NaN when `counters[0]` (failed count) is nonzero or `counters[2]` (used count) is
 * zero. The fixed-point sum is exact for the float32 terms, and order independent.
 *
 * @internal
 */
export function createVecchiaFixedPointSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rowCount: number;
    cellCount: number;
    terms: GraphDataView<'float32'>;
    counters: GraphDataView<'uint32'>;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, rowCount, cellCount, terms, counters, totals} = props;
  const scale = createTransientView(graph, `${id}-scale-word`, 'uint32', 1);
  const words = createTransientView(
    graph,
    `${id}-words`,
    'uint32',
    cellCount * FIXED_POINT_WORD_COUNT
  );
  const headroomBits = getFixedPointHeadroomBits(rowCount);
  return [
    ...createFixedPointScaleNodes<Parameters>(graph, {
      id: `${id}-scale`,
      operation,
      values: terms,
      scale
    }),
    createFixedPointClearNode<Parameters>(graph, {
      id: `${id}-words-clear`,
      operation,
      words
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-accumulate`,
      operation,
      variant: 'accumulate',
      bindings: [
        {name: 'terms', view: terms, type: 'f32', access: 'read'},
        {name: 'scale', view: scale, type: 'u32', access: 'read'},
        {name: 'fixedWords', view: words, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rowCount * cellCount,
      workgroupSize: 64,
      declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('fixedWords')}`,
      constants: {FIXED_POINT_HEADROOM_BITS: headroomBits},
      body: `let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], FIXED_POINT_HEADROOM_BITS);
addFixedPointToFixedWords(index % ${cellCount}u, quantizeFixedPoint(terms[termsOffset + index], unitExponent));`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-decode`,
      operation,
      variant: 'decode',
      bindings: [
        {name: 'fixedWords', view: words, type: 'u32', access: 'read'},
        {name: 'scale', view: scale, type: 'u32', access: 'read'},
        {name: 'counters', view: counters, type: 'u32', access: 'read'},
        {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
      ],
      invocationCount: cellCount,
      workgroupSize: 16,
      declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}
${GET_NAN_WGSL}`,
      constants: {FIXED_POINT_HEADROOM_BITS: headroomBits},
      body: `if (counters[countersOffset] != 0u || counters[countersOffset + 2u] == 0u) {
  totals[totalsOffset + index * 2u] = getNaN();
  totals[totalsOffset + index * 2u + 1u] = getNaN();
  return;
}
let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], FIXED_POINT_HEADROOM_BITS);
let word = fixedWordsOffset + index * ${FIXED_POINT_WORD_COUNT}u;
let exact = vec3<u32>(fixedWords[word], fixedWords[word + 1u], fixedWords[word + 2u]);
let high = getFixedPointFloat(exact, unitExponent);
let remainder = addFixedPoint(exact, negateFixedPoint(quantizeFixedPoint(high, unitExponent)));
totals[totalsOffset + index * 2u] = high;
totals[totalsOffset + index * 2u + 1u] = getFixedPointFloat(remainder, unitExponent);`
    })
  ];
}

/**
 * Per-site likelihood terms from solved conditionals, counts, and the sum nodes: run after the solve
 * that wrote `output.conditionals` and `output.conditionalStatus`. The marginal variance of a site
 * with an empty conditioning set is `params[0]`, the covariance scale `C(0, 0)` of both the
 * spatial and space-time parameter layouts.
 *
 * @internal
 */
export function createVecchiaLogLikelihoodSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rowCount: number;
    observations: GraphDataView<'float32x4'>;
    neighborhoodOffsets: GraphDataView<'uint32'>;
    parameters: GraphDataView<'float32'>;
    output: GPUVecchiaLikelihoodOutput;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, rowCount, observations, neighborhoodOffsets, parameters, output} = props;
  validateGraphViewsBelongToGraph(id, graph, [
    observations,
    neighborhoodOffsets,
    parameters,
    output.conditionals,
    output.conditionalStatus,
    output.siteTerms,
    output.siteStatus,
    output.logLikelihood,
    output.status
  ]);
  const logTwoPi = getWGSLFloatLiteral(Math.log(2 * Math.PI));
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-clear-counts`,
      operation,
      variant: 'clear-counts',
      bindings: [{name: 'counters', view: output.status, type: 'u32', access: 'read_write'}],
      invocationCount: 3,
      workgroupSize: 4,
      body: 'counters[countersOffset + index] = 0u;'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-site-terms`,
      operation,
      variant: 'site-terms',
      bindings: [
        {name: 'conditionals', view: output.conditionals, type: 'f32', access: 'read'},
        {name: 'conditionalStatus', view: output.conditionalStatus, type: 'u32', access: 'read'},
        {name: 'observations', view: observations, type: 'f32', access: 'read'},
        {name: 'offsets', view: neighborhoodOffsets, type: 'u32', access: 'read'},
        {name: 'params', view: parameters, type: 'f32', access: 'read'},
        {name: 'siteTerms', view: output.siteTerms, type: 'f32', access: 'read_write'},
        {name: 'siteStatus', view: output.siteStatus, type: 'u32', access: 'read_write'},
        {name: 'counters', view: output.status, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rowCount,
      workgroupSize: 64,
      declarations: `${getPackedPredictionAccessors({conditionals: 4, observations: 4})}
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
const LOG_TWO_PI: f32 = ${logTwoPi};`,
      body: `siteTerms[siteTermsOffset + index] = getNaN();
let observation = read_observations(index);
if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0) {
  siteStatus[siteStatusOffset + index] = 1u;
  atomicAdd(&counters[countersOffset + 1u], 1u);
  return;
}
let isMarginal = offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u];
var mean = 0.0;
var variance = 0.0;
if (isMarginal) {
  if (!(params[paramsOffset] > 0.0)) {
    siteStatus[siteStatusOffset + index] = 2u;
    atomicAdd(&counters[countersOffset], 1u);
    return;
  }
  variance = params[paramsOffset];
} else {
  if (conditionalStatus[conditionalStatusOffset + index] != 0u) {
    siteStatus[siteStatusOffset + index] = 2u;
    atomicAdd(&counters[countersOffset], 1u);
    return;
  }
  let conditional = read_conditionals(index);
  mean = conditional.x;
  variance = conditional.y;
}
variance += observation.y;
let residual = observation.x - mean;
let quadratic = residual * residual / variance;
let term = -0.5 * (LOG_TWO_PI + log(variance) + quadratic);
if (!(variance > 0.0) || !isFiniteValue(variance) || !isFiniteValue(term)) {
  siteStatus[siteStatusOffset + index] = 3u;
  atomicAdd(&counters[countersOffset], 1u);
  return;
}
siteTerms[siteTermsOffset + index] = term;
siteStatus[siteStatusOffset + index] = 0u;
atomicAdd(&counters[countersOffset + 2u], 1u);`
    }),
    ...createVecchiaFixedPointSumNodes<Parameters>(graph, {
      id,
      operation,
      rowCount,
      cellCount: 1,
      terms: output.siteTerms,
      counters: output.status,
      totals: output.logLikelihood
    })
  ];
}
