// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../analysis-core/kernels/incomplete-gamma-wgsl';
import {POISSON_OBSERVED_TERM_WGSL} from '../../analysis-core/kernels/wgsl-poisson-observed-term';
import {TRANSITION_INTEGER_PRODUCTS_WGSL} from './transition-integer-products-wgsl';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUTransitionDiagnosticsOutput} from './gpu-transition-diagnostics';

/** Maximum condition terms visited by one tile or merge invocation. @internal */
export const TRANSITION_CONDITION_TILE_SIZE = 256;

/** Parallel condition-axis statistics; only the constructor-bounded class axis is visited serially. @internal */
export function createTransitionDiagnosticsSummaryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    counts: GraphDataView<'uint32'>;
    classCount: number;
    conditionCount: number;
    minimumTransitions: number;
    output: GPUTransitionDiagnosticsOutput;
  }
): {nodes: GPUCommandNode<Parameters>[]; pooledCounts?: GraphDataView<'float32'>} {
  const {id, counts, classCount, conditionCount, minimumTransitions, output} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const rowCount = conditionCount * classCount;
  const cellCount = rowCount * classCount;
  const needsPooledCounts = Boolean(
    output.homogeneity || (output.steadyStates && conditionCount > 1)
  );
  // These scratch fields never exceed the already checked count tensor's byte size.
  // Check before creating any transient, including the fixed class-offset arrays.
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'summary counts',
    cellCount * 4,
    'reduce conditionCount or classCount'
  );
  const floating = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const unsigned = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const bind = (name: string, view: GraphDataView, writable = false): WGSLKernelBinding => ({
    name,
    view,
    type: view.format === 'float32' ? 'f32' : 'u32',
    access: writable ? 'read_write' : 'read'
  });
  const constants = `const K: u32 = ${classCount}u;
const C: u32 = ${conditionCount}u;
const MINIMUM_TRANSITIONS: u32 = ${minimumTransitions}u;`;
  const kernel = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string
  ) => {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${name}`,
        operation: 'GPUTransitionDiagnostics',
        variant: name,
        invocationCount,
        bindings,
        declarations: `${constants}\n${POISSON_OBSERVED_TERM_WGSL}\n${TRANSITION_INTEGER_PRODUCTS_WGSL}`,
        body
      })
    );
  };
  const sum = <Format extends 'float32' | 'uint32'>(name: string, input: GraphDataView<Format>) => {
    const result = createTransientView(graph, `${id}-${name}-result`, input.format, 1);
    nodes.push(
      ...new GPUReduction({
        id: `${id}-${name}`,
        input,
        output: result,
        operation: 'sum'
      }).getCommandNodes<Parameters>(graph)
    );
    return result;
  };
  // Exact two-limb integer partials retain small counts beside large regimes. No invocation
  // visits more than TRANSITION_CONDITION_TILE_SIZE = 256 terms; each level shrinks by 256.
  const poolConditions = (
    name: string,
    input: GraphDataView<'uint32'>,
    segments: number,
    transposed: boolean,
    result: GraphDataView<'uint32'> | GraphDataView<'float32'>,
    integerResult?: GraphDataView<'uint32'>
  ) => {
    let terms = conditionCount;
    let low = input;
    let high: GraphDataView<'uint32'> | undefined;
    let level = 0;
    do {
      const tiles = Math.ceil(terms / TRANSITION_CONDITION_TILE_SIZE);
      const length = segments * tiles;
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        `${name} integer partials`,
        length * 4,
        'reduce conditionCount or classCount'
      );
      const nextLow = unsigned(`${name}-${level}-low`, length);
      const nextHigh = unsigned(`${name}-${level}-high`, length);
      kernel(
        `${name}-${level}`,
        length,
        [
          bind('low', low),
          ...(high ? [bind('high', high)] : []),
          bind('nextLow', nextLow, true),
          bind('nextHigh', nextHigh, true)
        ],
        `let segment = index / ${tiles}u;
  let tile = index % ${tiles}u;
  var lowTotal = 0u;
  var highTotal = 0u;
  for (var term = 0u; term < ${TRANSITION_CONDITION_TILE_SIZE}u; term++) {
    let condition = tile * ${TRANSITION_CONDITION_TILE_SIZE}u + term;
    if (condition < ${terms}u) {
      let address = ${level === 0 && !transposed ? `condition * ${segments}u + segment` : `segment * ${terms}u + condition`};
      let value = low[lowOffset + address];
      let next: u32 = lowTotal + value;
      highTotal += select(0u, 1u, next < lowTotal)${high ? ' + high[highOffset + address]' : ''};
      lowTotal = next;
    }
  }
  nextLow[nextLowOffset + index] = lowTotal;
  nextHigh[nextHighOffset + index] = highTotal;`
      );
      low = nextLow;
      high = nextHigh;
      terms = tiles;
      level++;
    } while (terms > 1);
    kernel(
      `${name}-publish`,
      segments,
      [
        bind('low', low),
        ...(result.format === 'float32' ? [bind('high', high)] : []),
        bind('result', result, true),
        ...(integerResult ? [bind('integerResult', integerResult, true)] : [])
      ],
      (result.format === 'float32'
        ? 'result[resultOffset + index] = f32(high[highOffset + index]) * 4294967296.0 + f32(low[lowOffset + index]);'
        : 'result[resultOffset + index] = low[lowOffset + index];') +
        (integerResult
          ? '\n  integerResult[integerResultOffset + index * 2u] = low[lowOffset + index];\n  integerResult[integerResultOffset + index * 2u + 1u] = high[highOffset + index];'
          : '')
    );
  };
  if (output.status?.converged) {
    kernel(
      'reset-converged',
      1,
      [bind('converged', output.status.converged, true)],
      'converged[convergedOffset] = 1u;'
    );
  }
  let pooled: GraphDataView<'float32'> | undefined;
  const pooledIntegers = output.homogeneity
    ? unsigned('pooled-integers', 2 * classCount * classCount)
    : undefined;
  if (needsPooledCounts) {
    pooled = floating('pooled-counts', classCount * classCount);
    poolConditions(
      'pool-conditions',
      counts,
      classCount * classCount,
      false,
      pooled,
      pooledIntegers
    );
  }
  if (!output.homogeneity && !output.stateSparsity) return {nodes, pooledCounts: pooled};

  const rowTotals = output.homogeneity ? floating('regime-totals', rowCount) : undefined;
  const activeRows = output.homogeneity ? unsigned('active-regimes', rowCount) : undefined;
  const sparseRows = output.stateSparsity ? unsigned('sparse-rows', rowCount) : undefined;
  const emptyRows = output.stateSparsity ? unsigned('empty-rows', rowCount) : undefined;
  kernel(
    'regime-rows',
    rowCount,
    [
      bind('counts', counts),
      ...(rowTotals ? [bind('totals', rowTotals, true), bind('hasRegime', activeRows!, true)] : []),
      ...(sparseRows ? [bind('sparse', sparseRows, true), bind('empty', emptyRows!, true)] : [])
    ],
    `let origin = index / C;
  let condition = index % C;
  let base = (condition * K + origin) * K;
  var total = 0.0;
  var integerTotal = 0u;
  var overflowed = false;
  for (var destination = 0u; destination < K; destination++) {
    let count = counts[countsOffset + base + destination];
    total += f32(count);
    overflowed = overflowed || count > 0xffffffffu - integerTotal;
    integerTotal += count;
  }
  ${rowTotals ? 'totals[totalsOffset + index] = total;\n  hasRegime[hasRegimeOffset + index] = select(0u, 1u, total > 0.0);' : ''}
  ${
    sparseRows
      ? `sparse[sparseOffset + index] = select(0u, 1u, !overflowed && integerTotal > 0u && integerTotal < MINIMUM_TRANSITIONS);
  empty[emptyOffset + index] = select(0u, 1u, !overflowed && integerTotal == 0u);`
      : ''
  }`
  );
  if (output.stateSparsity) {
    const sparse = sum('sum-sparse', sparseRows!);
    const empty = sum('sum-empty', emptyRows!);
    kernel(
      'publish-sparsity',
      1,
      [
        bind('sparse', sparse),
        bind('empty', empty),
        bind('stateSparsity', output.stateSparsity, true)
      ],
      'stateSparsity[stateSparsityOffset] = sparse[sparseOffset];\n  stateSparsity[stateSparsityOffset + 1u] = empty[emptyOffset];'
    );
  }
  if (!output.homogeneity) return {nodes, pooledCounts: pooled};
  const regimes = unsigned('regime-counts', classCount);
  poolConditions('count-regimes', activeRows!, classCount, true, regimes);
  const pooledTotals = floating('pooled-totals', classCount);
  const nonzero = unsigned('nonzero-destinations', classCount);
  kernel(
    'pooled-rows',
    classCount,
    [bind('pooled', pooled!), bind('totals', pooledTotals, true), bind('nonzero', nonzero, true)],
    `var total = 0.0;
  var nonzeroCount = 0u;
  for (var destination = 0u; destination < K; destination++) {
    let count = pooled[pooledOffset + index * K + destination];
    total += count;
    nonzeroCount += select(0u, 1u, count > 0.0);
  }
  totals[totalsOffset + index] = total;
  nonzero[nonzeroOffset + index] = nonzeroCount;`
  );
  const pearsonTerms = floating('pearson-terms', rowCount);
  const likelihoodTerms = floating('likelihood-terms', rowCount);
  kernel(
    'homogeneity-terms',
    rowCount,
    [
      bind('counts', counts),
      bind('pooledIntegers', pooledIntegers!),
      bind('pooledTotals', pooledTotals),
      bind('totals', rowTotals!),
      bind('pearson', pearsonTerms, true),
      bind('likelihood', likelihoodTerms, true)
    ],
    `let origin = index / C;
  let condition = index % C;
  let regimeTotal = totals[totalsOffset + index];
  let pooledTotal = pooledTotals[pooledTotalsOffset + origin];
  var regimeTotalInteger = TransitionCount(0u, 0u);
  var pooledTotalInteger = TransitionCount(0u, 0u);
  for (var destination = 0u; destination < K; destination++) {
    regimeTotalInteger = addTransitionCounts(regimeTotalInteger, TransitionCount(0u, counts[countsOffset + (condition * K + origin) * K + destination]));
    let cell = (origin * K + destination) * 2u;
    pooledTotalInteger = addTransitionCounts(pooledTotalInteger, TransitionCount(pooledIntegers[pooledIntegersOffset + cell + 1u], pooledIntegers[pooledIntegersOffset + cell]));
  }
  var statistic = 0.0;
  var ratio = 0.0;
  if (regimeTotal > 0.0 && pooledTotal > 0.0) {
    for (var destination = 0u; destination < K; destination++) {
      let cell = (origin * K + destination) * 2u;
      let pooledCountInteger = TransitionCount(pooledIntegers[pooledIntegersOffset + cell + 1u], pooledIntegers[pooledIntegersOffset + cell]);
      let pooledCount = transitionCountToFloat(pooledCountInteger);
      let expected = regimeTotal * pooledCount / pooledTotal;
      if (expected > 0.0) {
        // Integer products retain a one-count residual even above binary64 precision.
        let observedInteger = counts[countsOffset + (condition * K + origin) * K + destination];
        let difference = getTransitionProductDifference(observedInteger, pooledTotalInteger, regimeTotalInteger, pooledCountInteger) / pooledTotal;
        statistic += difference * difference / expected;
        ratio += expected * getEntropyExcess(difference / expected);
      }
    }
  }
  pearson[pearsonOffset + index] = statistic;
  likelihood[likelihoodOffset + index] = ratio;`
  );
  const pearson = sum('sum-pearson', pearsonTerms);
  const likelihood = sum('sum-likelihood', likelihoodTerms);
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-publish-homogeneity`,
      operation: 'GPUTransitionDiagnostics',
      variant: 'publish-homogeneity',
      bindings: [
        bind('pearson', pearson),
        bind('likelihood', likelihood),
        bind('regimes', regimes),
        bind('nonzero', nonzero),
        bind('homogeneity', output.homogeneity, true)
      ],
      // contract-allow: single-invocation -- GPU_TRANSITION_DIAGNOSTICS_MAXIMUM_CLASS_COUNT = 64 bounds classCount; every condition-axis sum uses published reductions.
      invocationCount: 1,
      declarations: `${constants}\n${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}`,
      body: `var degreesOfFreedom = 0.0;
  for (var origin = 0u; origin < K; origin++) {
    let regimeCount = regimes[regimesOffset + origin];
    let destinations = nonzero[nonzeroOffset + origin];
    if (regimeCount > 0u && destinations > 0u) {
      degreesOfFreedom += f32(regimeCount - 1u) * f32(destinations - 1u);
    }
  }
  let statistic = pearson[pearsonOffset];
  let ratio = max(2.0 * likelihood[likelihoodOffset], 0.0);
  let nan = getNaN();
  homogeneity[homogeneityOffset] = statistic;
  homogeneity[homogeneityOffset + 1u] = select(nan, getChiSquareSurvivalValue(statistic, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + 2u] = ratio;
  homogeneity[homogeneityOffset + 3u] = select(nan, getChiSquareSurvivalValue(ratio, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + 4u] = degreesOfFreedom;`
    })
  );
  return {nodes, pooledCounts: pooled};
}
