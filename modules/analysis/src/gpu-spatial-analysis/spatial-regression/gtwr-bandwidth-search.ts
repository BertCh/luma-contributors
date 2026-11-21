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
  type WGSLDispatchGate
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUStatusPort,
  type GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getLocalKernelWeightWGSL} from './local-weighted-least-squares-wgsl';
import {LOCAL_REGRESSION_PAIRS_WGSL} from './local-regression-pairs-wgsl';
import {
  createGoldenSectionIntervalNodes,
  getGoldenSectionIntervalNodeCount
} from './gwr-golden-section-nodes';

/** Bounded private neighbour list, including the focal observation and zero-distance ties. */
export const GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT = 128;
/** Nested searches unroll quadratically into command nodes; the shared node budget also applies. */
export const GPU_GTWR_MAXIMUM_SEARCH_ITERATIONS = 32;
/** Search bounds and stopping tolerances occupy six float32 words. */
export const GPU_GTWR_SEARCH_PARAMETER_LENGTH = 6;

/** Compile-time search budget with per-frame bounds and tolerances. */
export type GPUGTWRBandwidthSearch = {
  parameters: GraphDataView<'float32'>;
  /** Maximum comparisons per scalar search, default 8, at most 32. */
  maximumIterations?: number;
};

/** @internal */
export function validateGTWRBandwidthSearch(
  id: string,
  search?: GPUGTWRBandwidthSearch,
  status?: GPUIterationStatusPort
): number {
  validateRequirement(id, !status || Boolean(search), 'output.status requires bandwidthSearch');
  if (!search) {
    return 0;
  }
  validateSinglePackedView(search.parameters, ['float32'], `${id} bandwidthSearch.parameters`);
  validateRequirement(
    id,
    search.parameters.length >= GPU_GTWR_SEARCH_PARAMETER_LENGTH,
    `bandwidthSearch.parameters must have at least ${GPU_GTWR_SEARCH_PARAMETER_LENGTH} values`
  );
  const iterations = search.maximumIterations ?? 8;
  validateCapacity(
    id,
    'bandwidthSearch.maximumIterations',
    iterations,
    1,
    GPU_GTWR_MAXIMUM_SEARCH_ITERATIONS
  );
  if (status) {
    validateGPUStatusPort(id, status);
  }
  return iterations;
}

/** Node count of a profile search; scalar objective conversion and publication included. @internal */
export function getGTWRBandwidthSearchNodeCount(iterations: number, fitNodes: number): number {
  return (
    getGoldenSectionIntervalNodeCount(
      iterations,
      getGoldenSectionIntervalNodeCount(iterations, fitNodes + 2) + 1
    ) + 1
  );
}

/** Adaptive bandwidth selection on usable space-time points, including the focal row. @internal */
export function getGTWRAdaptiveBandwidthWGSL(rows: number): string {
  return `
fn getAdaptiveBandwidth(focal: u32, requested: f32, timeScale: f32) -> f32 {
  if (!isFiniteBits(requested) || requested < 1.0 || requested != round(requested)
      || requested > ${GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT}.0) { return getNaN(); }
  let count = u32(requested);
  var distances: array<f32, ${GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT}>;
  for (var neighbor = 0u; neighbor < count; neighbor++) { distances[neighbor] = 3.402823466e+38; }
  let origin = pointsOffset + focal * 3u;
  var included = 0u;
  for (var row = 0u; row < ${rows}u; row++) {
    let base = pointsOffset + row * 3u;
    if (!isFiniteBits(points[base])) { continue; }
    included++;
    let deltaX = points[base] - points[origin];
    let deltaY = points[base + 1u] - points[origin + 1u];
    let deltaTime = points[base + 2u] - points[origin + 2u];
    let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
    var position = count;
    for (var neighbor = 0u; neighbor < count; neighbor++) {
      if (distance < distances[neighbor]) { position = neighbor; break; }
    }
    if (position < count) {
      for (var neighbor = count - 1u; neighbor > position; neighbor--) { distances[neighbor] = distances[neighbor - 1u]; }
      distances[position] = distance;
    }
  }
  if (included < count) { return getNaN(); }
  // The mgtwr/mgwr inflation keeps the kth distance inside the bisquare support. All-coincident
  // neighbourhoods have zero bandwidth and are singular, consistently with fixed bandwidth zero.
  return distances[count - 1u] * 1.0000001;
}`;
}

/**
 * Nested scalar golden-section searches, outer time scale and inner bandwidth. The evaluator
 * returns the usual GTWR summary; a candidate with singular rows cannot win by dropping data.
 * Invalid per-frame bounds publish NaN settings and do not run fits. @internal
 */
export function createGTWRBandwidthSearchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    adaptive: boolean;
    search: GPUGTWRBandwidthSearch;
    parameters: GraphDataView<'float32'>;
    parameterOffset?: number;
    status?: GPUIterationStatusPort;
    accumulateStatus?: boolean;
    evaluate: (
      parameters: GraphDataView<'float32'>,
      gate: WGSLDispatchGate<Parameters>,
      id: string
    ) => {
      nodes: readonly GPUCommandNode<Parameters>[];
      summary: GraphDataView<'float32'>;
    };
  }
): GPUCommandNode<Parameters>[] {
  const iterations = props.search.maximumIterations ?? 8;
  const interval = (
    axis: 0 | 1,
    id: string,
    parent: GraphDataView<'float32'> | undefined,
    evaluate: (
      value: GraphDataView<'float32'>,
      score: GraphDataView<'float32'>,
      enabled: GraphDataView<'float32'>,
      id: string
    ) => readonly GPUCommandNode<Parameters>[]
  ) =>
    createGoldenSectionIntervalNodes(graph, {
      id,
      bounds: props.search.parameters,
      axis,
      integer: axis === 0 && props.adaptive,
      maximumValue: axis === 0 && props.adaptive ? GPU_GTWR_MAXIMUM_NEIGHBOR_COUNT : undefined,
      maximumIterations: iterations,
      parent,
      evaluate
    });
  const outer = interval(
    1,
    `${props.id}-time-scale`,
    undefined,
    (timeScale, profileScore, outerState, outerId) => {
      const inner = interval(
        0,
        `${outerId}-bandwidth`,
        outerState,
        (bandwidth, score, enabled, innerId) => {
          const parameters = createTransientView(graph, `${innerId}-parameters`, 'float32', 4);
          const gate = createWGSLDispatchGate<Parameters>(graph, {
            id: `${innerId}-gate`,
            operation: 'GTWRBandwidthSearch',
            shapes: [[props.rows], [1]]
          });
          const evaluation = props.evaluate(parameters, gate, innerId);
          return [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${innerId}-parameters`,
              operation: 'GTWRBandwidthSearch',
              invocationCount: 1,
              bindings: [
                {name: 'bandwidth', view: bandwidth, type: 'f32', access: 'read'},
                {name: 'timeScale', view: timeScale, type: 'f32', access: 'read'},
                {name: 'enabled', view: enabled, type: 'f32', access: 'read'},
                {name: 'parameters', view: parameters, type: 'f32', access: 'read_write'},
                gate.binding
              ],
              body: `parameters[parametersOffset] = bandwidth[bandwidthOffset];
  parameters[parametersOffset + 1u] = timeScale[timeScaleOffset];
  parameters[parametersOffset + 2u] = 0.0;
  parameters[parametersOffset + 3u] = 0.0;
  if (enabled[enabledOffset + 5u] > 0.5) { ${gate.armWGSL} }
  else { ${gate.disarmWGSL} }`
            }),
            ...evaluation.nodes,
            createWGSLKernelNode<Parameters>(graph, {
              id: `${innerId}-objective`,
              operation: 'GTWRBandwidthSearch',
              invocationCount: 1,
              ...gate.getGate(1),
              bindings: [
                {name: 'summary', view: evaluation.summary, type: 'f32', access: 'read'},
                {name: 'score', view: score, type: 'f32', access: 'read_write'}
              ],
              declarations: GET_NAN_WGSL,
              body: `score[scoreOffset] = select(getNaN(), summary[summaryOffset + 2u],
    summary[summaryOffset + 6u] == 0.0 && summary[summaryOffset + 4u] > 0.0);
  score[scoreOffset + 1u] = 0.0;
  score[scoreOffset + 2u] = 1.0;
  score[scoreOffset + 3u] = 0.0;`
            })
          ];
        }
      );
      return [
        ...inner.nodes,
        createWGSLKernelNode<Parameters>(graph, {
          id: `${outerId}-profile`,
          operation: 'GTWRBandwidthSearch',
          invocationCount: 1,
          bindings: [
            {name: 'result', view: inner.result, type: 'f32', access: 'read'},
            {name: 'score', view: profileScore, type: 'f32', access: 'read_write'}
          ],
          body: `score[scoreOffset] = result[resultOffset + 1u];
  score[scoreOffset + 1u] = result[resultOffset];
  score[scoreOffset + 2u] = result[resultOffset + 3u];
  score[scoreOffset + 3u] = result[resultOffset + 6u];`
        })
      ];
    }
  );
  const statusEntries = Object.entries(props.status ?? {}).filter(
    (entry): entry is [string, GraphDataView<'uint32'>] => Boolean(entry[1])
  );
  return [
    ...outer.nodes,
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-selected-parameters`,
      operation: 'GTWRBandwidthSearch',
      invocationCount: 1,
      bindings: [
        {name: 'result', view: outer.result, type: 'f32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read_write'},
        ...statusEntries.map(([name, view]) => ({
          name,
          view,
          type: 'u32' as const,
          access: 'read_write' as const
        }))
      ],
      body: `parameters[parametersOffset + ${props.parameterOffset ?? 0}u] = result[resultOffset + 2u];
  parameters[parametersOffset + ${(props.parameterOffset ?? 0) + 1}u] = result[resultOffset];
  ${statusEntries
    .map(([name]) => {
      const value =
        name === 'iterationCount'
          ? 'u32(result[resultOffset + 4u])'
          : name === 'converged'
            ? 'u32(result[resultOffset + 3u])'
            : 'u32(result[resultOffset + 6u])';
      return `${name}[${name}Offset] = ${props.accumulateStatus ? (name === 'iterationCount' ? `${name}[${name}Offset] + ${value}` : `${name === 'converged' ? 'min' : 'max'}(${name}[${name}Offset], ${value})`) : value};`;
    })
    .join('\n  ')}`
    })
  ];
}

/** Single-covariate, no-intercept local regression used to score an MGTWR partial residual. @internal */
export function createMGTWRComponentEvaluationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    predictorCount: number;
    component: number;
    points: GraphDataView<'float32'>;
    predictors: GraphDataView<'float32'>;
    partial: GraphDataView<'float32'>;
    partialCorrections?: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    adaptive: boolean;
    bisquare: boolean;
    gate: WGSLDispatchGate<Parameters>;
  }
): {nodes: GPUCommandNode<Parameters>[]; summary: GraphDataView<'float32'>} {
  const {id, rows, gate} = props;
  const scratch = createTransientView(graph, `${id}-fit`, 'float32', rows * 4);
  const scoreScratch = createTransientView(graph, `${id}-score-fit`, 'float32', rows * 3);
  const residuals = createTransientView(graph, `${id}-residuals`, 'float32', rows);
  const summary = createTransientView(graph, `${id}-summary`, 'float32', 12);
  const design = (row: string) =>
    props.component === 0
      ? '1.0'
      : `predictors[predictorsOffset + ${row} * ${props.predictorCount}u + ${props.component - 1}u]`;
  return {
    summary,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: 'MGTWRComponentBandwidthSearch',
        invocationCount: rows,
        ...gate.getGate(0),
        bindings: [
          {name: 'points', view: props.points, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'partial', view: props.partial, type: 'f32', access: 'read'},
          ...(props.partialCorrections
            ? [
                {
                  name: 'partialCorrections',
                  view: props.partialCorrections,
                  type: 'f32',
                  access: 'read'
                } as const
              ]
            : []),
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'scratch', view: scratch, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${getLocalKernelWeightWGSL()}\n${LOCAL_REGRESSION_PAIRS_WGSL}\n${props.adaptive ? getGTWRAdaptiveBandwidthWGSL(rows) : ''}
fn readPartialValue(row: u32) -> vec2f {
  return vec2f(partial[partialOffset + row], ${props.partialCorrections ? 'partialCorrections[partialCorrectionsOffset + row]' : '0.0'});
}`,
        body: `let base = scratchOffset + index * 4u;
  scratch[base] = getNaN();
  scratch[base + 1u] = getNaN();
  scratch[base + 2u] = select(0.0, 1.0, isFiniteBits(points[pointsOffset + index * 3u]));
  scratch[base + 3u] = getNaN();
  if (scratch[base + 2u] == 0.0) { return; }
  let timeScale = parameters[parametersOffset + 1u];
  let bandwidth = ${props.adaptive ? 'getAdaptiveBandwidth(index, parameters[parametersOffset], timeScale)' : 'parameters[parametersOffset]'};
  if (!(bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale))) { return; }
  var numeratorPair = vec2f(0.0);
  var denominatorPair = vec2f(0.0);
  ${
    props.component === 0
      ? `var minimum = 3.402823466e+38;
  var maximum = -3.402823466e+38;
  for (var row = 0u; row < ${rows}u; row++) {
    if (!isFiniteBits(points[pointsOffset + row * 3u])) { continue; }
    let origin = pointsOffset + index * 3u;
    let deltaX = points[pointsOffset + row * 3u] - points[origin];
    let deltaY = points[pointsOffset + row * 3u + 1u] - points[origin + 1u];
    let deltaTime = points[pointsOffset + row * 3u + 2u] - points[origin + 2u];
    let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
    if (!(getLocalKernelWeight(distance, bandwidth, ${props.bisquare ? 'true' : 'false'}) > 0.0)) { continue; }
    minimum = min(minimum, partial[partialOffset + row]);
    maximum = max(maximum, partial[partialOffset + row]);
  }
  let partialAnchor = getRegressionResponseOrigin(minimum, maximum);`
      : 'let partialAnchor = 0.0;'
  }
  for (var row = 0u; row < ${rows}u; row++) {
    let point = pointsOffset + row * 3u;
    let origin = pointsOffset + index * 3u;
    if (!isFiniteBits(points[point])) { continue; }
    let deltaX = points[point] - points[origin];
    let deltaY = points[point + 1u] - points[origin + 1u];
    let deltaTime = points[point + 2u] - points[origin + 2u];
    let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
    let weight = getLocalKernelWeight(distance, bandwidth, ${props.bisquare ? 'true' : 'false'});
    let design = ${design('row')};
    if (!(weight > 0.0)) { continue; }
    let centered = addRegressionPairs(readPartialValue(row), vec2f(-partialAnchor, 0.0));
    numeratorPair = addRegressionPairs(numeratorPair, multiplyRegressionPair(multiplyRegressionPair(centered, design), weight));
    denominatorPair = addRegressionPairs(denominatorPair, multiplyRegressionPair(multiplyRegressionPair(vec2f(design, 0.0), design), weight));
  }
  let ownDesign = ${design('index')};
  let numerator = numeratorPair.x + numeratorPair.y;
  let denominator = denominatorPair.x + denominatorPair.y;
  if (denominator > 0.0 && isFiniteBits(numerator) && isFiniteBits(denominator)) {
    // Divide before multiplying by the focal design, as in the final component smoother.
    // ownDesign * numerator can overflow even when the coefficient and fitted value are finite.
    let coefficient = addRegressionPairs(vec2f(partialAnchor, 0.0), divideRegressionPair(numeratorPair, denominatorPair));
    let fitted = multiplyRegressionPair(coefficient, ownDesign);
    let residual = addRegressionPairs(readPartialValue(index), -fitted);
    scratch[base] = fitted.x + fitted.y;
    scratch[base + 3u] = residual.x + residual.y;
    scratch[base + 1u] = ownDesign * ownDesign / denominator;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-retain-residuals`,
        operation: 'MGTWRComponentBandwidthSearch',
        invocationCount: rows,
        ...gate.getGate(0),
        bindings: [
          {name: 'fit', view: scratch, type: 'f32', access: 'read'},
          {name: 'scratch', view: scoreScratch, type: 'f32', access: 'read_write'},
          {name: 'residuals', view: residuals, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `let source = fitOffset + index * 4u;
  let destination = scratchOffset + index * 3u;
  scratch[destination] = select(fit[source], 0.0, isFiniteBits(fit[source]));
  scratch[destination + 1u] = fit[source + 1u];
  scratch[destination + 2u] = fit[source + 2u];
  residuals[residualsOffset + index] = fit[source + 3u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: 'MGTWRComponentBandwidthSearch',
        invocationCount: 1,
        ...gate.getGate(1),
        bindings: [
          {name: 'scratch', view: scoreScratch, type: 'f32', access: 'read'},
          {name: 'partial', view: residuals, type: 'f32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `var count = 0.0;
  var singular = 0.0;
  var residualSquares = 0.0;
  var trace = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    let base = scratchOffset + row * 3u;
    if (scratch[base + 2u] == 0.0) { continue; }
    if (!isFiniteBits(scratch[base]) || !isFiniteBits(scratch[base + 1u])) { singular += 1.0; continue; }
    count += 1.0;
    let residual = partial[partialOffset + row] - scratch[base];
    residualSquares += residual * residual;
    trace += scratch[base + 1u];
  }
  let denominator = count - trace - 2.0;
  // Work in log space: 2 pi * RSS can overflow even when RSS and 2 pi * RSS / n are finite.
  let logVariance = log(residualSquares) - log(count);
  let logLikelihood = -0.5 * count * (logVariance + 1.8378770664093453 + 1.0);
  for (var field = 0u; field < 12u; field++) { summary[summaryOffset + field] = getNaN(); }
  summary[summaryOffset + 2u] = select(getNaN(), -2.0 * logLikelihood + 2.0 * count * (trace + 1.0) / denominator, count > 0.0 && denominator > 0.0);
  summary[summaryOffset + 4u] = count;
  summary[summaryOffset + 6u] = singular;`
      })
    ]
  };
}
