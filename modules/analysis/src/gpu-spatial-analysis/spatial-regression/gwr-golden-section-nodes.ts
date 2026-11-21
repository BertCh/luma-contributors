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
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE
} from './geographically-weighted-regression-parameters';
import {getRingNeighborDistancesWGSL} from './gwr-grid-index';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Upper bound for the generic interval search, including both boundary evaluations. @internal */
export function getGoldenSectionIntervalNodeCount(
  iterations: number,
  evaluationNodes: number
): number {
  return 1 + (2 * iterations + 2) * (evaluationNodes + 2) + iterations;
}

/**
 * Scalar golden-section minimization shared by the space-time profile searches. Both endpoints
 * are tested; ties keep the smaller value. `score` is [objective, auxiliary value, converged,
 * iterationLimitReached]. `result` is [value, objective, auxiliary, converged, iterations, active,
 * iterationLimitReached]. A parent search enables a nested search through result[5].
 * The callback gates its data work with `enabled`; state nodes remain ungated. @internal
 */
export function createGoldenSectionIntervalNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    bounds: GraphDataView<'float32'>;
    axis: 0 | 1;
    integer: boolean;
    maximumValue?: number;
    maximumIterations: number;
    parent?: GraphDataView<'float32'>;
    evaluate: (
      value: GraphDataView<'float32'>,
      score: GraphDataView<'float32'>,
      enabled: GraphDataView<'float32'>,
      id: string
    ) => readonly GPUCommandNode<Parameters>[];
  }
): {nodes: GPUCommandNode<Parameters>[]; result: GraphDataView<'float32'>} {
  const {id, maximumIterations} = props;
  const state = createTransientView(graph, `${id}-interval`, 'float32', 8);
  const result = createTransientView(graph, `${id}-result`, 'float32', 7);
  const value = createTransientView(graph, `${id}-value`, 'float32', 1);
  const score = createTransientView(graph, `${id}-score`, 'float32', 4);
  const read = (name: string, view: GraphDataView<'float32'>): WGSLKernelBinding => ({
    name,
    view,
    type: 'f32',
    access: 'read'
  });
  const write = (name: string, view: GraphDataView<'float32'>): WGSLKernelBinding => ({
    name,
    view,
    type: 'f32',
    access: 'read_write'
  });
  const node = (suffix: string, bindings: WGSLKernelBinding[], body: string) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${suffix}`,
      operation: 'GTWRBandwidthSearch',
      variant: suffix.replace(/\d+/g, ''),
      invocationCount: 1,
      bindings,
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body
    });
  const nodes = [
    node(
      'initialize',
      [
        read('bounds', props.bounds),
        write('state', state),
        write('result', result),
        ...(props.parent ? [read('parent', props.parent)] : [])
      ],
      `
  let lower = bounds[boundsOffset + ${props.axis * 2}u];
  let upper = bounds[boundsOffset + ${props.axis * 2 + 1}u];
  let absoluteTolerance = bounds[boundsOffset + 4u];
  let relativeTolerance = bounds[boundsOffset + 5u];
  var valid = isFiniteBits(lower) && isFiniteBits(upper) && lower <= upper
    && lower ${props.axis === 0 ? '> 0.0' : '>= 0.0'}
    && isFiniteBits(absoluteTolerance) && absoluteTolerance > 0.0
    && isFiniteBits(relativeTolerance) && relativeTolerance >= 0.0;
  ${props.integer ? 'valid = valid && lower == round(lower) && upper == round(upper);' : ''}
  ${props.maximumValue === undefined ? '' : `valid = valid && upper <= ${props.maximumValue}.0;`}
  ${props.parent ? 'valid = valid && parent[parentOffset + 5u] > 0.5;' : ''}
  state[stateOffset] = lower;
  state[stateOffset + 1u] = upper;
  state[stateOffset + 2u] = 0.0;
  state[stateOffset + 3u] = 0.0;
  state[stateOffset + 4u] = 0.0;
  state[stateOffset + 5u] = 0.0;
  state[stateOffset + 6u] = 0.0;
  state[stateOffset + 7u] = 0.0;
  for (var field = 0u; field < 7u; field++) { result[resultOffset + field] = 0.0; }
  result[resultOffset] = getNaN();
  result[resultOffset + 1u] = getNaN();
  result[resultOffset + 2u] = getNaN();
  result[resultOffset + 5u] = select(0.0, 1.0, valid);`
    )
  ];
  // Every round re-evaluates the interior pair, so nested objectives need no cached side effects.
  for (let evaluation = 0; evaluation < 2 + 2 * maximumIterations; evaluation++) {
    const boundary = evaluation < 2;
    const side = evaluation % 2;
    nodes.push(
      node(
        `prepare-${evaluation}`,
        [
          write('state', state),
          read('result', result),
          write('value', value),
          write('score', score)
        ],
        `
  if (result[resultOffset + 5u] < 0.5) { return; }
  let lower = state[stateOffset];
  let upper = state[stateOffset + 1u];
  var candidate = ${
    boundary
      ? side === 0
        ? 'lower'
        : 'upper'
      : side === 0
        ? `lower + ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA} * (upper - lower)`
        : `upper - ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA} * (upper - lower)`
  };
  ${props.integer ? 'candidate = round(candidate);' : ''}
  ${
    props.integer && !boundary && side === 1
      ? `// Rounded interior points must compare different counts while the bracket has room.
  // In a three-count bracket this visits the middle and upper counts; the lower was tested
  // when it became a bracket endpoint. The iteration can then certify all three counts.
  candidate = min(upper, max(candidate, state[stateOffset + 2u] + 1.0));`
      : ''
  }
  value[valueOffset] = candidate;
  state[stateOffset + ${2 + side}u] = candidate;
  for (var field = 0u; field < 4u; field++) { score[scoreOffset + field] = getNaN(); }`
      )
    );
    nodes.push(...props.evaluate(value, score, result, `${id}-evaluate-${evaluation}`));
    nodes.push(
      node(
        `collect-${evaluation}`,
        [
          read('value', value),
          read('score', score),
          write('state', state),
          write('result', result)
        ],
        `
  if (result[resultOffset + 5u] < 0.5) { return; }
  let objective = score[scoreOffset];
  let candidate = value[valueOffset];
  state[stateOffset + ${4 + side}u] = objective;
  let best = result[resultOffset + 1u];
  if (isFiniteBits(objective) && (!isFiniteBits(best) || objective < best
      || (objective == best && candidate < result[resultOffset]))) {
    result[resultOffset] = candidate;
    result[resultOffset + 1u] = objective;
    result[resultOffset + 2u] = score[scoreOffset + 1u];
    state[stateOffset + 6u] = score[scoreOffset + 2u];
    state[stateOffset + 7u] = score[scoreOffset + 3u];
  }`
      )
    );
    if (!boundary && side === 1) {
      nodes.push(
        node(
          `iterate-${evaluation}`,
          [read('bounds', props.bounds), write('state', state), write('result', result)],
          `
  if (result[resultOffset + 5u] < 0.5) { return; }
  let firstScore = state[stateOffset + 4u];
  let secondScore = state[stateOffset + 5u];
  let firstWins = isFiniteBits(firstScore) && (!isFiniteBits(secondScore) || firstScore <= secondScore);
  ${props.integer ? 'let resolvedIntegerInterval = state[stateOffset + 1u] - state[stateOffset] <= 2.0;' : ''}
  if (firstWins) { state[stateOffset + 1u] = state[stateOffset + 3u]; }
  else { state[stateOffset] = state[stateOffset + 2u]; }
  let lower = state[stateOffset];
  let upper = state[stateOffset + 1u];
  let tolerance = max(bounds[boundsOffset + 4u], bounds[boundsOffset + 5u] * max(abs(lower), abs(upper)));
  let finished = ${props.integer ? 'resolvedIntegerInterval || ' : ''}upper - lower <= ${props.integer ? 'max(1.0, tolerance)' : 'tolerance'};
  result[resultOffset + 4u] += 1.0;
  if (finished || result[resultOffset + 4u] >= ${maximumIterations}.0) {
    result[resultOffset + 5u] = 0.0;
    result[resultOffset + 3u] = select(0.0, state[stateOffset + 6u], finished && isFiniteBits(result[resultOffset + 1u]));
    result[resultOffset + 6u] = max(state[stateOffset + 7u], select(0.0, 1.0, !finished));
  }`
        )
      );
    }
  }
  return {nodes, result};
}

/** Inputs of {@link createGoldenSectionSearchNodes}. @internal */
export type GoldenSectionSearchProps = {
  /** Prefix of the node and transient IDs. */
  id: string;
  /** Skip distance-derived bounds when the caller supplies both positive bounds. */
  callerBoundsOnly?: boolean;
  /** Evaluate the upper endpoint too, retaining it when the two interior scores are invalid. */
  probeUpperBound?: boolean;
  /** Override dispatch shapes for callers with different reduction passes. */
  dispatchShapes?: readonly (readonly [number, number?])[];
  /** Optional outer backfit state: only code 3 starts another search. */
  runningState?: GraphDataView<'uint32'>;
  /** Operation name of the generated kernels. */
  operation: string;
  rowCount: number;
  /** Rows summed by one thread of a tile pass. */
  tileRows: number;
  tileCount: number;
  /** Initial candidates: two interior points, optionally also the upper endpoint. */
  candidateWidth: number;
  /** Compile-time `maximumSearchIterations`. */
  maximumIterations: number;
  /** Rows of the candidate table: maximumIterations + 1, or + 2 with the endpoint probe. */
  tableRowCount: number;
  /** Declarations shared by every GWR kernel (the constants and the NaN helpers). */
  common: string;
  /** Candidate declarations: the pass parameter reader and the local fit helpers. */
  candidateDeclarations: string;
  /** WGSL that maps the invocation index to the focal row (grid order or row order). */
  focalRowWGSL: string;
  /** True when a `GPUGridIndex` is bound (its bindings are in `candidateBindings`). */
  hasGrid: boolean;
  /** Bindings of the candidate-evaluation kernel except `candidateScratch`, in its order. */
  candidateBindings: WGSLKernelBinding[];
  /** Row validity flags (the head of the shared row-state buffer). */
  rowValid: GraphDataView<'uint32'>;
  /** `[cursor, block 0, block 1, ...]` ladders: block `i` holds the candidates of pass `i`. */
  ladders: GraphDataView<'float32'>;
  /** The caller's parameters (bounds in the first two ladder values). */
  parameters: GraphDataView<'float32'>;
  candidateScratch: GraphDataView<'float32'>;
  tileCandidates: GraphDataView<'float32'>;
  candidateTable: GraphDataView<'float32'>;
  searchState: GraphDataView<'float32'>;
};

/** Nodes and gate of the golden-section search. @internal */
export type GoldenSectionSearchNodes<Parameters> = {
  /** Bound computation and search initialization, in order, before the first candidate pass. */
  setupNodes: GPUCommandNode<Parameters>[];
  /**
   * Gate for the candidate passes after the first: slot 0 candidates (`rowCount` invocations),
   * slot 1 tile sums (`tileCount * candidateWidth`), slot 2 selection (`candidateWidth`).
   */
  gate: WGSLDispatchGate<Parameters>;
  /** The one-thread kernel that compares the two interior scores after pass `iteration - 1`. */
  createIterationNode: (iteration: number) => GPUCommandNode<Parameters>;
};

const STATE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE;

/**
 * Creates the golden-section bandwidth search (mgwr `golden_section`) as GPU nodes: the default
 * bound computation, the initialization, and the per-iteration state update. The caller wires the
 * candidate passes between them (pass `i` reads block `i` of `ladders`), gated by `gate` after
 * pass 0.
 *
 * State words (float32, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE`): the bracket
 * `a, b, c, d`, the table rows of the scores of `b` and `d`, the iteration count, the armed flag,
 * the table row of the last winner, the evaluation count and the converged flag. The loop index
 * is the cursor word of `ladders`, never WGSL text (rule 12), and a converged search zeroes the
 * dispatch triples of the later passes (rule 13).
 *
 * @internal
 */
export function createGoldenSectionSearchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GoldenSectionSearchProps
): GoldenSectionSearchNodes<Parameters> {
  const {id, operation, rowCount, tileCount, candidateWidth} = props;
  const gate = createWGSLDispatchGate<Parameters>(graph, {
    id: `${id}-search-dispatch`,
    operation,
    shapes: props.dispatchShapes ?? [[rowCount], [tileCount * candidateWidth], [candidateWidth]]
  });
  const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
    name,
    view,
    type,
    access: 'read'
  });
  const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
    name,
    view,
    type,
    access: 'read_write'
  });
  const stateConstants = Object.entries(STATE)
    .map(([name, slot]) => `const STATE_${name}: u32 = ${slot}u;`)
    .join('\n');
  const searchCommon = `${props.common}
const MAXIMUM_ITERATIONS: u32 = ${props.maximumIterations}u;
const GOLDEN_DELTA: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA};
const TOLERANCE_ABSOLUTE: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE};
const TOLERANCE_RELATIVE: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE};
${stateConstants}`;
  // Scratch per row: [nearest positive distance, farthest distance]; SENTINEL / 0 when unused.
  const boundsRowsBody = `${props.focalRowWGSL}
  let slot = candidateScratchOffset + focal * 2u;
  candidateScratch[slot] = SENTINEL;
  candidateScratch[slot + 1u] = 0.0;
  if (getIsAdaptive() || rowValid[rowValidOffset + focal] == 0u) {
    return;
  }
  let origin = getPosition(focal);
  var nearest = SENTINEL;
  var farthest = 0.0;
  ${
    props.hasGrid
      ? `// Nearest strictly positive distance, as for the scan below: rows coincident with the focal
  // row (repeated measures per site) are skipped, not counted as the nearest neighbour.
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  selectPositiveNeighborDistances(focal, 1u, &neighbors);
  if (neighbors[0] < SENTINEL) {
    nearest = sqrt(neighbors[0]);
  }
  // The farthest row is no farther than the farthest corner of the grid extent. This upper bound
  // is looser than the scan's exact farthest row, so the grid path brackets the search with a
  // wider upper bound than the scan path; the bracket is documented, not exact.
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  farthest = max(
    max(getDistance(origin, minimum), getDistance(origin, maximum)),
    max(getDistance(origin, vec2f(minimum.x, maximum.y)), getDistance(origin, vec2f(maximum.x, minimum.y)))
  );`
      : `for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let distance = getDistance(origin, getPosition(row));
    if (distance > 0.0) {
      nearest = min(nearest, distance);
    }
    farthest = max(farthest, distance);
  }`
  }
  candidateScratch[slot] = nearest;
  candidateScratch[slot + 1u] = farthest;`;
  const setupNodes: GPUCommandNode<Parameters>[] = [
    // Per row: nearest positive distance and farthest distance (fixed bandwidths only).
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-bounds-rows`,
      operation,
      variant: 'search-bounds-rows',
      bindings: [
        ...props.candidateBindings,
        write('candidateScratch', props.candidateScratch, 'f32')
      ],
      invocationCount: rowCount,
      declarations: props.hasGrid
        ? `${props.candidateDeclarations}\n${getRingNeighborDistancesWGSL('selectPositiveNeighborDistances', true)}`
        : props.candidateDeclarations,
      body: boundsRowsBody
    }),
    // Per tile: smallest nearest distance, largest farthest distance, included rows.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-bounds-tiles`,
      operation,
      variant: 'search-bounds-tiles',
      bindings: [
        read('candidateScratch', props.candidateScratch, 'f32'),
        read('rowValid', props.rowValid, 'u32'),
        write('tileCandidates', props.tileCandidates, 'f32')
      ],
      invocationCount: tileCount,
      declarations: searchCommon,
      body: `let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    nearest = min(nearest, candidateScratch[candidateScratchOffset + row * 2u]);
    farthest = max(farthest, candidateScratch[candidateScratchOffset + row * 2u + 1u]);
  }
  let base = tileCandidatesOffset + index * 3u;
  tileCandidates[base] = nearest;
  tileCandidates[base + 1u] = farthest;
  tileCandidates[base + 2u] = included;`
    }),
    // Bounds, interior points, block 0 of the ladders, the state and the gate.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-init`,
      operation,
      variant: 'search-init',
      bindings: [
        read('params', props.parameters, 'f32'),
        write('ladders', props.ladders, 'f32'),
        read('tileCandidates', props.tileCandidates, 'f32'),
        write('table', props.candidateTable, 'f32'),
        write('state', props.searchState, 'f32'),
        gate.binding,
        ...(props.runningState ? [read('runningState', props.runningState, 'u32')] : [])
      ],
      // contract-allow: single-invocation -- GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS = 64 bounds 66 table rows including two endpoints, each with seven fields; MAXIMUM_DEFAULT_TILE_COUNT = 4096 bounds default-layout GWR partials (large route pre-reduces to one); MGWR supplies bounds.
      invocationCount: 1,
      declarations: searchCommon,
      body: `let nan = getNaN();
  for (var row = 0u; row < TABLE_ROWS; row++) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[tableOffset + row * TABLE_STRIDE + field] = nan;
    }
  }
  let isAdaptive = params[paramsOffset + 1u] > 0.5;
  let callerCount = u32(max(params[paramsOffset + 2u], 0.0));
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < ${props.callerBoundsOnly ? '0u' : 'TILE_COUNT'}; tile++) {
    let base = tileCandidatesOffset + tile * 3u;
    nearest = min(nearest, tileCandidates[base]);
    farthest = max(farthest, tileCandidates[base + 1u]);
    included = included + tileCandidates[base + 2u];
  }
  // Adaptive [min(40 + 2 P, n), n], n capped at the neighbour list; fixed
  // [min_dist / 2, 2 max_dist]; a positive caller bound replaces its default.
  var lower = 0.0;
  var upper = 0.0;
  if (isAdaptive) {
    upper = min(included, f32(MAXIMUM_NEIGHBORS));
    lower = min(f32(40u + 2u * P), upper);
  } else if (included > 0.0 && nearest < SENTINEL) {
    lower = 0.5 * nearest;
    upper = 2.0 * farthest;
  }
  let isSearch = callerCount >= 2u${props.runningState ? ' && runningState[runningStateOffset] == 3u' : ''};
  if (isSearch) {
    let givenLower = params[paramsOffset + HEADER];
    let givenUpper = params[paramsOffset + HEADER + 1u];
    if (givenLower > 0.0 && isFiniteBits(givenLower)) {
      lower = givenLower;
    }
    if (givenUpper > 0.0 && isFiniteBits(givenUpper)) {
      upper = givenUpper;
    }
  }
  let low = min(lower, upper);
  let high = max(lower, upper);
  var first = low + GOLDEN_DELTA * (high - low);
  var second = high - GOLDEN_DELTA * (high - low);
  if (isAdaptive) {
    // numpy round: half to even, as WGSL round.
    first = round(first);
    second = round(second);
  }
  if (isSearch) {
    // Block 0 holds the caller header; it now carries the two interior points of iteration 1.
    ladders[laddersOffset + 3u] = ${props.probeUpperBound ? '3.0' : '2.0'};
    ladders[laddersOffset + 1u + HEADER] = first;
    ladders[laddersOffset + 2u + HEADER] = second;
    ${props.probeUpperBound ? 'ladders[laddersOffset + 3u + HEADER] = high;' : ''}
  }
  state[stateOffset + STATE_LOWER] = low;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = high;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = 0.0;
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = 1.0;
  state[stateOffset + STATE_ITERATION_COUNT] = 0.0;
  state[stateOffset + STATE_ACTIVE] = select(0.0, 1.0, isSearch);
  // A single caller value is a fixed bandwidth: its evaluation is table row 0.
  state[stateOffset + STATE_OPTIMUM_ROW] = select(0.0, -1.0, isSearch);
  state[stateOffset + STATE_EVALUATION_COUNT] = select(1.0, ${props.probeUpperBound ? '3.0' : '2.0'}, isSearch);
  state[stateOffset + STATE_CONVERGED] = 0.0;
  state[stateOffset + STATE_RESERVED] = 0.0;
  if (isSearch) {
    ${gate.armWGSL}
  } else {
    ${gate.disarmWGSL}
  }`
    })
  ];

  const createIterationNode = (iteration: number): GPUCommandNode<Parameters> =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-${iteration}-iterate`,
      operation,
      variant: 'search-iterate',
      bindings: [
        write('ladders', props.ladders, 'f32'),
        read('table', props.candidateTable, 'f32'),
        write('state', props.searchState, 'f32'),
        gate.binding
      ],
      invocationCount: 1,
      declarations: `${searchCommon}
fn getRowScore(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 1u];
}
fn getRowSingular(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 2u];
}`,
      // The iteration index is the state word, not this node's position: one pipeline for all.
      body: `if (state[stateOffset + STATE_ACTIVE] < 0.5) {
    return;
  }
  let iteration = u32(state[stateOffset + STATE_ITERATION_COUNT]) + 1u;
  let firstRow = u32(state[stateOffset + STATE_LOWER_INTERIOR_ROW]);
  let secondRow = u32(state[stateOffset + STATE_UPPER_INTERIOR_ROW]);
  let firstScore = getRowScore(firstRow);
  let secondScore = getRowScore(secondRow);
  let firstSingular = getRowSingular(firstRow);
  let secondSingular = getRowSingular(secondRow);
  let isFirstValid = isFiniteBits(firstScore);
  let isSecondValid = isFiniteBits(secondScore);
  // mgwr: score_b <= score_d keeps the lower interval. Fewest singular rows first; an invalid
  // score loses to a valid one; with two invalid scores the comparison is false, as for NaN.
  var firstWins = isFirstValid;
  if (isFirstValid && isSecondValid) {
    firstWins = firstSingular < secondSingular || (firstSingular == secondSingular && firstScore <= secondScore);
  }
  let isAdaptive = ladders[laddersOffset + 2u] > 0.5;
  var lower = state[stateOffset + STATE_LOWER];
  var first = state[stateOffset + STATE_LOWER_INTERIOR];
  var upper = state[stateOffset + STATE_UPPER];
  var second = state[stateOffset + STATE_UPPER_INTERIOR];
  let newRow = iteration + ${props.probeUpperBound ? '2u' : '1u'};
  var optimumRow = secondRow;
  var nextFirstRow = firstRow;
  var nextSecondRow = secondRow;
  var newValue = 0.0;
  if (firstWins) {
    // c = d; d = b; b = a + delta |c - a|
    optimumRow = firstRow;
    upper = second;
    second = first;
    nextSecondRow = firstRow;
    first = lower + GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      first = round(first);
    }
    newValue = first;
    nextFirstRow = newRow;
  } else {
    // a = b; b = d; d = c - delta |c - a|
    lower = first;
    first = second;
    nextFirstRow = secondRow;
    second = upper - GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      second = round(second);
    }
    newValue = second;
    nextSecondRow = newRow;
  }
  ${
    props.probeUpperBound
      ? `// The initial upper-bound evaluation is table row 2. A finite endpoint keeps the
  // bracket searchable even when both current interior points have invalid AICc.
  let endpointScore = getRowScore(2u);
  let endpointValid = isFiniteBits(endpointScore);
  let interiorScore = getRowScore(optimumRow);
  if (endpointValid && (!isFiniteBits(interiorScore) || endpointScore < interiorScore)) {
    optimumRow = 2u;
  }`
      : ''
  }
  // mgwr loops while |score_b - score_d| > tol; the difference is of the pair just compared.
  var keepSearching = true;
  if (isFirstValid && isSecondValid && firstSingular == secondSingular) {
    let tolerance = max(TOLERANCE_ABSOLUTE, TOLERANCE_RELATIVE * max(abs(firstScore), abs(secondScore)));
    keepSearching = abs(firstScore - secondScore) > tolerance;
  } else if (!isFirstValid && !isSecondValid) {
    keepSearching = ${props.probeUpperBound ? 'endpointValid' : 'false'};
  }
  state[stateOffset + STATE_LOWER] = lower;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = upper;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = f32(nextFirstRow);
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = f32(nextSecondRow);
  state[stateOffset + STATE_ITERATION_COUNT] = f32(iteration);
  state[stateOffset + STATE_OPTIMUM_ROW] = f32(optimumRow);
  if (!keepSearching || iteration >= MAXIMUM_ITERATIONS) {
    // Converged or out of iterations: no new candidate, and the later passes launch nothing.
    state[stateOffset + STATE_ACTIVE] = 0.0;
    state[stateOffset + STATE_CONVERGED] = select(0.0, 1.0, !keepSearching);
    ${gate.disarmWGSL}
    return;
  }
  // Schedule the one new candidate as block iteration of the ladders and advance the GPU cursor.
  ladders[laddersOffset] = f32(iteration);
  let blockBase = laddersOffset + 1u + iteration * PARAMETER_LENGTH;
  ladders[blockBase] = ladders[laddersOffset + 1u];
  ladders[blockBase + 1u] = ladders[laddersOffset + 2u];
  ladders[blockBase + 2u] = 1.0;
  ladders[blockBase + 3u] = ladders[laddersOffset + 4u];
  ladders[blockBase + HEADER] = newValue;
  state[stateOffset + STATE_EVALUATION_COUNT] = f32(newRow + 1u);`
    });

  return {
    setupNodes: props.callerBoundsOnly ? setupNodes.slice(2) : setupNodes,
    gate,
    createIterationNode
  };
}

/** Float64 MGWR twin of the shared transitions, including the upper-endpoint feasibility probe. @internal */
export function selectGoldenSectionBandwidth(
  lower: number,
  upper: number,
  evaluate: (bandwidth: number) => number,
  maximumIterations: number
): number {
  return selectGoldenSectionBandwidthWithStatus(lower, upper, evaluate, maximumIterations)
    .bandwidth;
}

/** Selected finite bandwidth and whether its score search exhausted the budget. @internal */
export function selectGoldenSectionBandwidthWithStatus(
  lower: number,
  upper: number,
  evaluate: (bandwidth: number) => number,
  maximumIterations: number
): {bandwidth: number; iterationLimitReached: boolean} {
  if (!(lower > 0) || !(upper >= lower) || !Number.isFinite(lower) || !Number.isFinite(upper))
    return {bandwidth: NaN, iterationLimitReached: false};
  const delta = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA;
  let first = lower + delta * (upper - lower);
  let second = upper - delta * (upper - lower);
  let firstScore = evaluate(first);
  let secondScore = evaluate(second);
  const endpoint = upper;
  const endpointScore = evaluate(endpoint);
  const endpointValid = Number.isFinite(endpointScore);
  let best = NaN;
  let iterationLimitReached = false;
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const firstValid = Number.isFinite(firstScore);
    const secondValid = Number.isFinite(secondScore);
    const firstWins = firstValid && (!secondValid || firstScore <= secondScore);
    best = firstWins ? first : secondValid ? second : NaN;
    const interiorScore = firstWins ? firstScore : secondScore;
    if (endpointValid && (!Number.isFinite(interiorScore) || endpointScore < interiorScore))
      best = endpoint;
    const tolerance = Math.max(
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE *
        Math.max(Math.abs(firstScore), Math.abs(secondScore))
    );
    const finished =
      (!firstValid && !secondValid && !endpointValid) ||
      (firstValid && secondValid && Math.abs(firstScore - secondScore) <= tolerance);
    if (finished || iteration + 1 === maximumIterations) {
      iterationLimitReached = !finished;
      break;
    }
    if (firstWins) {
      upper = second;
      second = first;
      secondScore = firstScore;
      first = lower + delta * Math.abs(upper - lower);
      firstScore = evaluate(first);
    } else {
      lower = first;
      first = second;
      firstScore = secondScore;
      second = upper - delta * Math.abs(upper - lower);
      secondScore = evaluate(second);
    }
  }
  return {bandwidth: best, iterationLimitReached};
}
