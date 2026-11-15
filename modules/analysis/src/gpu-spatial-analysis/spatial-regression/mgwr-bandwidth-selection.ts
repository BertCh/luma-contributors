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
  createWGSLKernelNode,
  createWGSLDispatchGate,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUMultiscaleGeographicallyWeightedRegressionProps} from './gpu-multiscale-geographically-weighted-regression';
import {createGoldenSectionSearchNodes} from './gwr-golden-section-nodes';
import {getCandidateLoopWGSL} from './local-weighted-least-squares-wgsl';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_MAXIMUM_SEARCH_ITERATIONS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH
} from './geographically-weighted-regression-parameters';

type Selection = GPUMultiscaleGeographicallyWeightedRegressionProps['bandwidthSelection'];
const OPERATION = 'GPUMultiscaleGeographicallyWeightedRegression';
// Four row terms: residual squares (scaled for AICc), smoother diagonal, included, failed.
const SCORE_STRIDE = 4;
// Shared GWR golden-section parameter header and two distance bounds/interior points.
// Two interior points plus the upper-bound feasibility probe.
const SEARCH_PARAMETER_LENGTH = 7;
const TABLE_STRIDE = 3;

/** Construction-time selection settings shared by the GPU and CPU twin. @internal */
export function validateMultiscaleBandwidthSelection(
  id: string,
  selection: Selection,
  hasCandidates: boolean,
  candidateCount?: number
): void {
  validateRequirement(
    id,
    !selection || hasCandidates,
    'bandwidthSelection requires bandwidthCandidates'
  );
  validateRequirement(
    id,
    selection?.criterion === undefined || ['leave-one-out', 'AICc'].includes(selection.criterion),
    'bandwidthSelection.criterion must be leave-one-out or AICc'
  );
  validateRequirement(
    id,
    selection?.searchMethod === undefined ||
      ['finite-grid', 'golden-section'].includes(selection.searchMethod),
    'bandwidthSelection.searchMethod must be finite-grid or golden-section'
  );
  const golden = selection?.searchMethod === 'golden-section';
  validateRequirement(
    id,
    !golden || candidateCount === 2,
    'golden-section bandwidthCandidateCount must be 2 (lower and upper bounds)'
  );
  if (selection?.maximumSearchIterations !== undefined) {
    validateRequirement(id, golden, 'maximumSearchIterations requires golden-section searchMethod');
    validateCapacity(id, 'maximumSearchIterations', selection.maximumSearchIterations);
    validateMaximumCount(
      id,
      'maximumSearchIterations',
      selection.maximumSearchIterations,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS
    );
  }
}

/** Shared across all columns and sweeps: selection never allocates scratch per iteration. @internal */
export function createMultiscaleBandwidthWorkspace<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rows: number,
  candidateCount: number,
  selection: Selection
) {
  const golden = selection?.searchMethod === 'golden-section';
  const width = golden ? 3 : candidateCount;
  const maximumIterations =
    selection?.maximumSearchIterations ??
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_MAXIMUM_SEARCH_ITERATIONS;
  const tableRows = golden ? maximumIterations + 2 : candidateCount;
  const layout = getRegressionTileLayout(rows);
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'bandwidth scores',
    rows * width * SCORE_STRIDE * 4,
    'reduce rows or bandwidthCandidateCount'
  );
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-selection-${name}`, 'float32', length);
  return {
    golden,
    width,
    maximumIterations,
    tableRows,
    layout,
    scores: transient('scores', rows * width * SCORE_STRIDE),
    partials: transient('partials', layout.tileCount * width * SCORE_STRIDE),
    totals: transient('totals', width * SCORE_STRIDE),
    table: transient('table', tableRows * TABLE_STRIDE),
    bounds: transient('bounds', SEARCH_PARAMETER_LENGTH),
    ladders: transient('ladders', 1 + maximumIterations * SEARCH_PARAMETER_LENGTH),
    searchState: transient('state', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH)
  };
}

/** Partial-response scalar selection, matching mgwr multi_bw's per-variable GWR. @internal */
export function createMultiscaleBandwidthSelection<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    pairs: GraphDataView<'float32'>;
    gridBindings: WGSLKernelBinding[];
    common: string;
    hasGrid: boolean;
    rows: number;
    column: number;
    candidates: GraphDataView<'float32'>;
    candidateCount: number;
    selection: Selection;
    parameters: GraphDataView<'float32'>;
    state: GraphDataView<'uint32'>;
    workspace: ReturnType<typeof createMultiscaleBandwidthWorkspace>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, workspace, rows, column} = props;
  const {
    golden,
    width,
    maximumIterations,
    tableRows,
    layout,
    scores,
    partials,
    totals,
    table,
    bounds,
    ladders,
    searchState
  } = workspace;
  const usesAICcCriterion = props.selection?.criterion === 'AICc';
  // Scale by an exact power of two before squaring. The divisor is at least the row capacity,
  // so RSS / divisor cannot overflow when RSS / includedCount is representable. Recover the
  // variance with the actual included count; masks must not change the statistical divisor.
  const residualScale = 2 ** -Math.ceil(Math.log2(rows) / 2);
  const residualSquareDivisor = 1 / (residualScale * residualScale);
  const read = (
    name: string,
    view: GraphDataView,
    type: 'f32' | 'u32' = 'f32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read'});
  const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
    name,
    view,
    type: 'f32',
    access: 'read_write'
  });
  const nodes: GPUCommandNode<Parameters>[] = [];
  const scalarCommon = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`;
  if (golden)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-bounds`,
        operation: OPERATION,
        variant: 'selection-bounds',
        invocationCount: SEARCH_PARAMETER_LENGTH,
        bindings: [
          read('candidates', props.candidates),
          write('bounds', bounds),
          write('ladders', ladders)
        ],
        declarations: scalarCommon,
        body: `let lower = candidates[candidatesOffset + ${column * 2}u];
let upper = candidates[candidatesOffset + ${column * 2 + 1}u];
let valid = lower > 0.0 && upper >= lower && isFiniteBits(lower) && isFiniteBits(upper);
let field = index;
var value = 0.0;
  if (field == 2u) { value = 2.0; }
  if (field == 4u) { value = select(getNaN(), lower, valid); }
  if (field == 5u) { value = select(getNaN(), upper, valid); }
  bounds[boundsOffset + field] = value;
  ladders[laddersOffset + 1u + field] = value;
if (index == 0u) { ladders[laddersOffset] = 0.0; }`
      })
    );
  const search = golden
    ? createGoldenSectionSearchNodes(graph, {
        id,
        operation: OPERATION,
        rowCount: rows,
        tileRows: layout.tileRowCount,
        tileCount: layout.tileCount,
        candidateWidth: width,
        maximumIterations,
        tableRowCount: tableRows,
        callerBoundsOnly: true,
        probeUpperBound: true,
        runningState: props.state,
        dispatchShapes: [[rows], [layout.tileCount, 64], [width], [width * SCORE_STRIDE]],
        common: `${scalarCommon}
const ROW_COUNT: u32 = ${rows}u;
const TILE_ROWS: u32 = ${layout.tileRowCount}u;
const TILE_COUNT: u32 = ${layout.tileCount}u;
const TABLE_ROWS: u32 = ${tableRows}u;
const TABLE_STRIDE: u32 = ${TABLE_STRIDE}u;
const HEADER: u32 = 4u;
const PARAMETER_LENGTH: u32 = ${SEARCH_PARAMETER_LENGTH}u;
const MAXIMUM_NEIGHBORS: u32 = 1u;
const P: u32 = 1u;
const SENTINEL: f32 = 3.402823e38;`,
        candidateDeclarations: '',
        focalRowWGSL: '',
        hasGrid: false,
        candidateBindings: [],
        // Distance-bound passes are omitted: both bounds are supplied per component.
        rowValid: props.state,
        ladders,
        parameters: bounds,
        candidateScratch: scores,
        tileCandidates: partials,
        candidateTable: table,
        searchState
      })
    : undefined;
  const finiteGate = search
    ? undefined
    : createWGSLDispatchGate(graph, {
        id: `${id}-backfit-dispatch`,
        operation: OPERATION,
        shapes: [[rows], [layout.tileCount, 64], [width], [width * SCORE_STRIDE]]
      });
  if (search) nodes.push(...search.setupNodes);
  if (finiteGate)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-backfit-gate`,
        operation: OPERATION,
        variant: 'selection-backfit-gate',
        invocationCount: 1,
        bindings: [read('state', props.state, 'u32'), finiteGate.binding],
        body: `if (state[stateOffset] == 3u) { ${finiteGate.armWGSL} } else { ${finiteGate.disarmWGSL} }`
      })
    );
  for (let pass = 0; pass < (golden ? maximumIterations : 1); pass++) {
    const prefix = `${id}-pass-${pass}`;
    const gated = search?.gate ?? finiteGate!;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${prefix}-evaluate`,
        operation: OPERATION,
        variant: 'selection-evaluate',
        invocationCount: rows,
        ...gated?.getGate(0),
        bindings: [
          read('positions', props.positions),
          read('pairs', props.pairs),
          read('candidates', golden ? ladders : props.candidates),
          write('scores', scores),
          read('state', props.state, 'u32'),
          ...props.gridBindings
        ],
        declarations: props.common,
        body: `for (var candidate = 0u; candidate < ${width}u; candidate++) {
  let slot = scoresOffset + (index * ${width}u + candidate) * 4u;
  for (var field = 0u; field < 4u; field++) { scores[slot + field] = 0.0; }
  ${
    golden
      ? `let candidateBlock = candidatesOffset + 1u + u32(candidates[candidatesOffset]) * ${SEARCH_PARAMETER_LENGTH}u;
  if (candidate >= u32(candidates[candidateBlock + 2u])) { continue; }
  let bandwidth = candidates[candidateBlock + 4u + candidate];`
      : `let bandwidth = candidates[candidatesOffset + ${column * props.candidateCount}u + candidate];`
  }
  if (state[stateOffset] != 3u || !isActive(index)) { continue; }
  scores[slot + 2u] = 1.0;
  scores[slot + 3u] = 1.0;
  if (!(bandwidth > 0.0) || !isFiniteBits(bandwidth)) { continue; }
  var numerator = 0.0; var denominator = 0.0;
  let focus = index;
  let origin = getPosition(focus);
  ${getCandidateLoopWGSL(
    `${usesAICcCriterion ? '' : 'if (row == focus) { continue; }'}
    let weight = spatialWeight(focus, row, bandwidth);
    if (!(weight > 0.0)) { continue; }
    let value = pairs[pairsOffset + 2u * row + 1u];
    numerator += weight * value * pairs[pairsOffset + 2u * row];
    denominator += weight * value * value;`,
    props.hasGrid,
    'isActive(row)'
  )}
  if (!(denominator > 0.0) || !isFiniteBits(denominator)) { continue; }
  let value = pairs[pairsOffset + 2u * focus + 1u];
  // Form the coefficient first: value * numerator can overflow while both the coefficient
  // and the fitted contribution remain finite.
  let coefficient = numerator / denominator;
  let residual = pairs[pairsOffset + 2u * focus] - value * coefficient;
  if (!isFiniteBits(residual)) { continue; }
  ${
    usesAICcCriterion
      ? `let scaledResidual = residual * f32(${residualScale});
  scores[slot] = scaledResidual * scaledResidual;`
      : 'scores[slot] = residual * residual;'
  }
  scores[slot + 1u] = ${usesAICcCriterion ? 'value * value / denominator' : '0.0'};
  scores[slot + 3u] = 0.0;
}`
      })
    );
    nodes.push(
      ...getRowTileSumNodes(graph, {
        id: prefix,
        operation: OPERATION,
        variant: 'selection-sum',
        rowCount: rows,
        layout,
        sumCount: width * SCORE_STRIDE,
        bindings: [read('scores', scores)],
        partial: partials,
        output: totals,
        tileGate: gated?.getGate(1),
        mergeGate: gated?.getGate(3),
        declarations: `fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  for (var term = 0u; term < TERM_COUNT; term++) { (*terms)[term] += scores[scoresOffset + row * TERM_COUNT + term]; }
}`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${prefix}-score`,
        operation: OPERATION,
        variant: 'selection-score',
        invocationCount: width,
        ...gated?.getGate(2),
        bindings: [
          read('totals', totals),
          read('candidates', golden ? ladders : props.candidates),
          write('table', table)
        ],
        declarations: scalarCommon,
        body: `${
          golden
            ? `let cursor = u32(candidates[candidatesOffset]);
let candidateBlock = candidatesOffset + 1u + cursor * ${SEARCH_PARAMETER_LENGTH}u;
if (index >= u32(candidates[candidateBlock + 2u])) { return; }
let bandwidth = candidates[candidateBlock + 4u + index];
let tableRow = select(cursor + 2u, index, cursor == 0u);`
            : `let bandwidth = candidates[candidatesOffset + ${column * props.candidateCount}u + index];
let tableRow = index;`
        }
let base = totalsOffset + index * 4u;
let residualSquares = totals[base]; let trace = totals[base + 1u]; let count = totals[base + 2u];
var score = getNaN();
if (count > 0.0 && totals[base + 3u] == 0.0 && isFiniteBits(residualSquares)) {
  ${
    usesAICcCriterion
      ? `let degrees = count - trace - 2.0;
  let variance = residualSquares * (f32(${residualSquareDivisor}) / count);
  if (degrees > 0.0 && isFiniteBits(trace) && isFiniteBits(variance)) {
    score = count * log(max(variance, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE})) + count * log(6.283185307179586) + count * (count + trace) / degrees;
  }`
      : 'score = residualSquares;'
  }
}
let slot = tableOffset + tableRow * 3u;
table[slot] = bandwidth; table[slot + 1u] = score; table[slot + 2u] = 0.0;`
      })
    );
    if (search) nodes.push(search.createIterationNode(pass + 1));
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-publish`,
      operation: OPERATION,
      variant: 'selection-publish',
      // contract-allow: single-invocation: finite-grid comparison is bounded by 32 candidates.
      invocationCount: 1,
      bindings: [
        read('table', table),
        read('searchState', searchState),
        write('parameters', props.parameters),
        read('state', props.state, 'u32')
      ],
      declarations: scalarCommon,
      body: `if (state[stateOffset] != 3u) { return; }
var best = getNaN();
${
  golden
    ? `let row = i32(searchState[searchStateOffset + ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE.OPTIMUM_ROW}u]);
if (row >= 0 && row < ${tableRows}) {
  let slot = tableOffset + u32(row) * 3u;
  if (isFiniteBits(table[slot + 1u])) { best = table[slot]; }
}`
    : `var bestScore = 3.402823e38;
for (var candidate = 0u; candidate < ${props.candidateCount}u; candidate++) {
  let slot = tableOffset + candidate * 3u;
  let score = table[slot + 1u];
  if (isFiniteBits(score) && score < bestScore) { bestScore = score; best = table[slot]; }
}`
}
parameters[parametersOffset + ${column + 1}u] = best;`
    })
  );
  return nodes;
}
