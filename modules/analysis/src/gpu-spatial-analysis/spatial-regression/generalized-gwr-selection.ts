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
  getWGSLFloatLiteral,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeneralizedGeographicallyWeightedRegression,
  createLocalRegressionGrid,
  getRegressionBindings,
  getRegressionWGSL
} from './gpu-generalized-geographically-weighted-regression';

/** Bounded row sums parallelize the score; the final reference pass visits at most 4,096 tiles. */
const SCORE_TILE_ROWS = 256;
/** Finish narrow adaptive brackets by evaluating all at most five integer candidates. */
const INTEGER_FINISH_MAXIMUM_SPAN = 4;
const COMMON = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`;
const OPERATION = 'GPUGeneralizedGeographicallyWeightedRegression';

/** GPU-only training-mask bandwidth selection. @internal */
export function getGeneralizedSelectionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUGeneralizedGeographicallyWeightedRegression
): readonly GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const selection = props.bandwidthSelection!;
  const isGolden = selection.method === 'golden-section';
  const isAICc = isGolden || selection.criterion === 'aicc';
  const maximumIterations = isGolden ? (selection.maximumIterations ?? 24) : 1;
  const candidateCount = isGolden ? 2 * maximumIterations : selection.bandwidths.length;
  const rows = props.response.length;
  const columns = props.predictorCount + 1;
  const tileCount = Math.ceil(rows / SCORE_TILE_ROWS);
  const read = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' = 'f32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read'});
  const write = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' = 'f32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
  validateGraphViewsBelongToGraph(id, graph, [
    ...Object.values(selection.output).filter(view => view !== undefined),
    ...Object.values(selection.status ?? {}).filter(view => view !== undefined),
    ...(isGolden ? [selection.parameters] : [])
  ]);
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const parameters = transient('selected-parameters', 2);
  const coefficients = transient('selection-coefficients', rows * columns);
  const fitted = transient('selection-fitted', rows);
  const codes = createTransientView(graph, `${id}-selection-codes`, 'uint32', rows);
  const iterationCount = createTransientView(graph, `${id}-selection-iterations`, 'uint32', rows);
  const eligible = createTransientView(graph, `${id}-selection-eligible`, 'uint32', rows);
  const tileScores = transient('selection-tile-scores', tileCount * 4);
  const standardErrors = isAICc ? transient('selection-errors', rows * columns) : undefined;
  const influence = isAICc ? transient('selection-influence', rows) : undefined;
  // Share the training index across all candidates and the final fit, including masked responses.
  const gridSetup = createLocalRegressionGrid(
    graph,
    `${id}-selection`,
    props,
    contributor.sphereRadius,
    props.family === 'poisson'
      ? 'responseValue >= 0.0 && floor(responseValue) == responseValue'
      : 'responseValue == 0.0 || responseValue == 1.0'
  );
  const effective = {...props, ...gridSetup.props};
  const nodes: GPUCommandNode<Parameters>[] = [...gridSetup.nodes];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-selection-eligible`,
      operation: OPERATION,
      invocationCount: rows,
      bindings: [...getRegressionBindings(effective), write('eligible', eligible, 'u32')],
      declarations: getRegressionWGSL(effective, {sphereRadius: contributor.sphereRadius}),
      body: `let observed = response[responseOffset + index];
      eligible[eligibleOffset + index] = select(0u, 1u, included(index) &&
        (${props.family === 'poisson' ? 'observed >= 0.0 && floor(observed) == observed' : 'observed == 0.0 || observed == 1.0'}));`
    })
  );

  // Search state: [lower, upper, first, second, chosen, bestScore, iterations, converged,
  // active, mode (0 golden / 1 integer finish / 2 feasibility probes), dyadic ordinal].
  const state = isGolden ? transient('search-state', 11) : undefined;
  const gate = isGolden
    ? createWGSLDispatchGate<Parameters>(graph, {
        id: `${id}-search-dispatch`,
        operation: OPERATION,
        shapes: [[rows], [1], [tileCount]]
      })
    : undefined;
  const historyBindings = selection.output.evaluatedBandwidths
    ? [write('history', selection.output.evaluatedBandwidths)]
    : [];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-selection-init`,
      operation: OPERATION,
      invocationCount: candidateCount,
      bindings: [
        write('scores', selection.output.scores),
        ...historyBindings,
        ...(isGolden
          ? [read('bounds', selection.parameters), write('state', state!), gate!.binding]
          : [])
      ],
      declarations: COMMON,
      body: `scores[scoresOffset + index] = getNaN();
      ${historyBindings.length ? 'history[historyOffset + index] = getNaN();' : ''}
      if (index != 0u) { return; }
    ${
      isGolden
        ? `
    for (var slot = 0u; slot < 11u; slot++) { state[stateOffset + slot] = 0.0; }
    let lower = bounds[boundsOffset]; let upper = bounds[boundsOffset + 1u];
    let tolerance = bounds[boundsOffset + 2u];
    var valid = isFiniteBits(lower) && isFiniteBits(upper) && lower > 0.0 && lower < upper &&
      isFiniteBits(tolerance) && tolerance > 0.0;
    ${props.bandwidthMode === 'adaptive' ? `valid = valid && floor(lower) == lower && floor(upper) == upper && upper <= ${props.maximumNeighborCount ?? 128}.0;` : ''}
    state[stateOffset] = lower; state[stateOffset + 1u] = upper;
    state[stateOffset + 2u] = ${props.bandwidthMode === 'adaptive' ? 'round(' : ''}lower + 0.38197 * (upper - lower)${props.bandwidthMode === 'adaptive' ? ')' : ''};
    state[stateOffset + 3u] = ${props.bandwidthMode === 'adaptive' ? 'round(' : ''}upper - 0.38197 * (upper - lower)${props.bandwidthMode === 'adaptive' ? ')' : ''};
    ${
      props.bandwidthMode === 'adaptive'
        ? `if (upper - lower <= ${INTEGER_FINISH_MAXIMUM_SPAN}.0) {
      state[stateOffset + 9u] = 1.0;
      state[stateOffset + 2u] = lower; state[stateOffset + 3u] = min(lower + 1.0, upper);
    }`
        : ''
    }
    state[stateOffset + 4u] = getNaN(); state[stateOffset + 5u] = 3.402823e38;
    state[stateOffset + 8u] = select(0.0, 1.0, valid);
    if (valid) { ${gate!.armWGSL} } else { ${gate!.disarmWGSL} }
    `
        : ''
    }`
    })
  );

  const evaluate = (
    candidate: number,
    bandwidth: string,
    candidateGate?: WGSLDispatchGate<Parameters>
  ) => {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-${candidate}-parameters`,
        operation: OPERATION,
        invocationCount: 1,
        ...(candidateGate?.getGate(1) ?? {}),
        bindings: [
          read('source', props.parameters),
          write('parameters', parameters),
          ...(state ? [read('state', state)] : []),
          ...historyBindings
        ],
        body: `parameters[parametersOffset] = ${bandwidth};
        parameters[parametersOffset + 1u] = source[sourceOffset + 1u];
        ${historyBindings.length ? `history[historyOffset + ${candidate}u] = parameters[parametersOffset];` : ''}`
      })
    );
    nodes.push(
      ...new GPUGeneralizedGeographicallyWeightedRegression({
        ...effective,
        id: `${id}-candidate-${candidate}`,
        bandwidthSelection: undefined,
        inference: isAICc ? {standardErrors: standardErrors!, influence: influence!} : undefined,
        leaveOneOut: !isAICc,
        predictExcluded: false,
        fittedScale: 'link',
        parameters,
        output: {coefficients, fitted, status: {codes, iterationCount}}
      }).getCommandNodes(graph, {gridSetup, gate: candidateGate})
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-${candidate}-deviance`,
        operation: OPERATION,
        invocationCount: tileCount,
        ...(candidateGate?.getGate(2) ?? {}),
        bindings: [
          read('response', props.response),
          read('fitted', fitted),
          read('codes', codes, 'u32'),
          read('eligible', eligible, 'u32'),
          ...(influence ? [read('influence', influence)] : []),
          write('tiles', tileScores)
        ],
        declarations: COMMON,
        body: `var deviance = 0.0; var trace = 0.0; var count = 0.0; var failed = 0.0;
        let endRow = min((index + 1u) * ${SCORE_TILE_ROWS}u, ${rows}u);
        for (var row = index * ${SCORE_TILE_ROWS}u; row < endRow; row++) {
          if (eligible[eligibleOffset + row] == 0u) { continue; }
          count += 1.0;
          if (codes[codesOffset + row] != 0u) { failed += 1.0; continue; }
          let observed = response[responseOffset + row]; let linear = fitted[fittedOffset + row];
          ${
            props.family === 'poisson'
              ? `let score = 2.0 * (select(0.0, observed * (log(max(observed, 1.0e-30)) - linear), observed > 0.0) - observed + exp(linear));`
              : `let opposed = select(linear, -linear, observed == 1.0);
          let score = 2.0 * (max(opposed, 0.0) + log(1.0 + exp(-abs(opposed))));`
          }
          if (!isFiniteBits(score)) { failed += 1.0; continue; }
          deviance += max(0.0, score);
          ${
            influence
              ? `let leverage = influence[influenceOffset + row];
          if (!isFiniteBits(leverage)) { failed += 1.0; } else { trace += leverage; }`
              : ''
          }
        }
        let base = tilesOffset + index * 4u;
        tiles[base] = deviance; tiles[base + 1u] = trace;
        tiles[base + 2u] = count; tiles[base + 3u] = failed;`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-${candidate}-score`,
        operation: OPERATION,
        // contract-allow: single-invocation; reference tier, at most 16 AICc or 4,096 legacy LOO tile totals.
        invocationCount: 1,
        ...(candidateGate?.getGate(1) ?? {}),
        bindings: [read('tiles', tileScores), write('scores', selection.output.scores)],
        declarations: COMMON,
        body: `var deviance = 0.0; var trace = 0.0; var count = 0.0; var failed = 0.0;
        for (var tile = 0u; tile < ${tileCount}u; tile++) {
          let base = tilesOffset + tile * 4u;
          deviance += tiles[base]; trace += tiles[base + 1u];
          count += tiles[base + 2u]; failed += tiles[base + 3u];
        }
        var score = getNaN();
        if (failed == 0.0 && count > 0.0 ${isAICc ? '&& count > trace + 1.0' : ''}) {
          score = ${isAICc ? 'deviance + 2.0 * trace + 2.0 * trace * (trace + 1.0) / (count - trace - 1.0)' : 'deviance / count'};
        }
        scores[scoresOffset + ${candidate}u] = score;`
      })
    );
  };

  if (isGolden) {
    for (let iteration = 0; iteration < maximumIterations; iteration++) {
      evaluate(2 * iteration, 'state[stateOffset + 2u]', gate);
      evaluate(2 * iteration + 1, 'state[stateOffset + 3u]', gate);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-search-${iteration}-iterate`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            read('scores', selection.output.scores),
            read('bounds', selection.parameters),
            write('state', state!),
            gate!.binding
          ],
          declarations: `${COMMON}
fn getFeasibilityFraction(ordinal: u32) -> f32 {
  let levelSize = 1u << firstLeadingBit(ordinal);
  return f32(2u * (ordinal - levelSize) + 1u) / f32(2u * levelSize);
}`,
          body: `if (state[stateOffset + 8u] == 0.0) { return; }
          let firstScore = scores[scoresOffset + ${2 * iteration}u];
          let secondScore = scores[scoresOffset + ${2 * iteration + 1}u];
          let firstValid = isFiniteBits(firstScore); let secondValid = isFiniteBits(secondScore);
          let chooseFirst = firstValid && (!secondValid || firstScore <= secondScore);
          let winner = select(state[stateOffset + 3u], state[stateOffset + 2u], chooseFirst);
          let winnerScore = select(secondScore, firstScore, chooseFirst);
          if ((firstValid || secondValid) && winnerScore < state[stateOffset + 5u]) {
            state[stateOffset + 4u] = winner; state[stateOffset + 5u] = winnerScore;
          }
          state[stateOffset + 6u] += 1.0;
          var lower = state[stateOffset]; var upper = state[stateOffset + 1u];
          var first = state[stateOffset + 2u]; var second = state[stateOffset + 3u];
          var mode = state[stateOffset + 9u];
          var converged = false; var finished = false;
          if (mode == 1.0) {
            // Every integer in this bracket must be scored before claiming convergence.
            finished = second >= upper;
            converged = finished && isFiniteBits(state[stateOffset + 4u]);
            if (!finished) { first = second + 1.0; second = min(first + 1.0, upper); }
          } else if (mode == 2.0) {
            // Invalid interiors imply no ordering. Keep the bracket, probe both endpoints first,
            // then cover it by a breadth-first dyadic sequence. Report budget exhaustion, not
            // convergence, since feasibility probes do not establish a minimizing bracket.
            let ordinal = u32(state[stateOffset + 10u]) + 1u;
            first = lower + (upper - lower) * getFeasibilityFraction(ordinal);
            second = lower + (upper - lower) * getFeasibilityFraction(ordinal + 1u);
            state[stateOffset + 10u] += 2.0;
          } else if (!firstValid && !secondValid) {
            // Never discard either side of an unordered pair of invalid scores.
            ${
              props.bandwidthMode === 'adaptive'
                ? 'mode = 1.0; first = lower; second = min(first + 1.0, upper);'
                : 'mode = 2.0; first = lower; second = upper;'
            }
          } else {
            let previousFirst = first; let previousSecond = second;
            let closeScores = firstValid && secondValid && first != second &&
              abs(firstScore - secondScore) <= bounds[boundsOffset + 2u];
            if (chooseFirst) {
              upper = previousSecond;
              first = ${props.bandwidthMode === 'adaptive' ? 'round(' : ''}lower + 0.38197 * (upper - lower)${props.bandwidthMode === 'adaptive' ? ')' : ''};
              second = previousFirst;
            } else {
              lower = previousFirst;
              first = previousSecond;
              second = ${props.bandwidthMode === 'adaptive' ? 'round(' : ''}upper - 0.38197 * (upper - lower)${props.bandwidthMode === 'adaptive' ? ')' : ''};
            }
            ${
              props.bandwidthMode === 'adaptive'
                ? `if (closeScores || upper - lower <= ${INTEGER_FINISH_MAXIMUM_SPAN}.0 ||
                first >= second || first <= lower || second >= upper) {
              mode = 1.0; first = lower; second = min(first + 1.0, upper);
            }`
                : 'converged = closeScores || first == second; finished = converged;'
            }
          }
          state[stateOffset + 9u] = mode;
          state[stateOffset] = lower; state[stateOffset + 1u] = upper;
          state[stateOffset + 2u] = first; state[stateOffset + 3u] = second;
          if (converged) { state[stateOffset + 7u] = 1.0; }
          if (finished || state[stateOffset + 6u] >= ${maximumIterations}.0) {
            state[stateOffset + 8u] = 0.0; ${gate!.disarmWGSL}
          }`
        })
      );
    }
  } else {
    selection.bandwidths.forEach((bandwidth, candidate) =>
      evaluate(candidate, getWGSLFloatLiteral(bandwidth))
    );
  }
  const statusBindings = Object.entries(selection.status ?? {})
    .filter((entry): entry is [string, GraphDataView<'uint32'>] => entry[1] !== undefined)
    .map(([name, view]) => write(name, view, 'u32'));
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-select-bandwidth`,
      operation: OPERATION,
      invocationCount: 1,
      bindings: [
        read('scores', selection.output.scores),
        write('bandwidth', selection.output.bandwidth),
        write('status', selection.output.status, 'u32'),
        write('parameters', parameters),
        ...(state ? [read('state', state)] : []),
        ...statusBindings
      ],
      declarations: COMMON,
      body: `var chosen = getNaN(); var best = 3.402823e38;
      ${
        isGolden
          ? `chosen = state[stateOffset + 4u];`
          : selection.bandwidths
              .map(
                (value, candidate) =>
                  `if (isFiniteBits(scores[scoresOffset + ${candidate}u]) && scores[scoresOffset + ${candidate}u] < best) {
          best = scores[scoresOffset + ${candidate}u]; chosen = ${getWGSLFloatLiteral(value)};
        }`
              )
              .join('\n')
      }
      bandwidth[bandwidthOffset] = chosen; parameters[parametersOffset] = chosen;
      status[statusOffset] = select(1u, 0u, isFiniteBits(chosen));
      ${selection.status?.iterationCount ? `iterationCount[iterationCountOffset] = ${isGolden ? 'u32(state[stateOffset + 6u])' : '1u'};` : ''}
      ${selection.status?.converged ? `converged[convergedOffset] = ${isGolden ? 'u32(state[stateOffset + 7u])' : 'select(0u, 1u, isFiniteBits(chosen))'};` : ''}
      ${selection.status?.iterationLimitReached ? `iterationLimitReached[iterationLimitReachedOffset] = ${isGolden ? `select(0u, 1u, state[stateOffset + 6u] >= ${maximumIterations}.0 && state[stateOffset + 7u] == 0.0)` : '0u'};` : ''}`
    })
  );
  // The final tolerance comes from the caller even when no candidate ran (invalid bounds).
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-selected-tolerance`,
      operation: OPERATION,
      invocationCount: 1,
      bindings: [read('source', props.parameters), write('parameters', parameters)],
      body: 'parameters[parametersOffset + 1u] = source[sourceOffset + 1u];'
    })
  );
  nodes.push(
    ...new GPUGeneralizedGeographicallyWeightedRegression({
      ...effective,
      id: `${id}-selected-fit`,
      bandwidthSelection: undefined,
      parameters
    }).getCommandNodes(graph, {gridSetup})
  );
  return nodes;
}
