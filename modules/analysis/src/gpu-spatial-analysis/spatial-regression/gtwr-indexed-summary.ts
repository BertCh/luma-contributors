// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {getBoundsReductionNodes} from '../hilbert-keys/bounds-reduction';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getGroupedSumNodes} from '../geographic-distribution/grouped-sum-reduction';
import type {GPUGeographicallyTemporallyWeightedRegressionProps} from './gpu-geographically-temporally-weighted-regression';

/** Fixed-tree parallel statistics with a second pass for centered total squares. @internal */
export function createGTWRIndexedSummaryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUGeographicallyTemporallyWeightedRegressionProps
): GPUCommandNode<Parameters>[] {
  const rows = props.response.length;
  const operation = 'GPUGeographicallyTemporallyWeightedRegression';
  const {output} = props;
  const statistics = Array.from({length: 4}, (_, index) =>
    createTransientView(graph, `${id}-statistic-${index}`, 'float32', 1)
  );
  const includedCount = createTransientView(graph, `${id}-included-count`, 'uint32', 1);
  const responseAnchor = createTransientView(graph, `${id}-response-anchor`, 'float32', 4);
  const totalSquares = createTransientView(graph, `${id}-total-squares`, 'float32', 1);
  return [
    ...getBoundsReductionNodes(graph, {
      id: `${id}-response-anchor`,
      operation,
      variant: 'response-anchor',
      itemCount: rows,
      output: responseAnchor,
      bindings: [
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read'}
      ],
      declarations: `fn readItem(row: u32) -> vec4f {
  return vec4f(response[responseOffset + row], 0.0,
    select(0.0, 1.0, localStatus[localStatusOffset + row] == 0u), 0.0);
}`
    }),
    ...getGroupedSumNodes(graph, {
      id: `${id}-statistics`,
      operation,
      rows,
      groupCount: 1,
      bindings: [
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'responseAnchor', view: responseAnchor, type: 'f32', access: 'read'},
        {name: 'fitted', view: output.fitted, type: 'f32', access: 'read'},
        {name: 'hatDiagonal', view: output.hatDiagonal, type: 'f32', access: 'read'},
        {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read'}
      ],
      outputs: statistics,
      countOutput: includedCount,
      rowSnippet: `let code = localStatus[localStatusOffset + row];
  if (code != 2u) { group = 0u; }
  if (code == 0u) {
    let observed = response[responseOffset + row];
    let residual = observed - fitted[fittedOffset + row];
    value0 = residual * residual;
    value1 = hatDiagonal[hatDiagonalOffset + row];
    // Center before reducing: every constant response gives exactly zero, even when 1/ROWS
    // is inexact. Half scaling keeps subtraction finite for opposite finite extrema.
    value2 = (observed * 0.5 - responseAnchor[responseAnchorOffset] * 0.5) / f32(ROWS);
    value3 = 1.0;
  }`
    }),
    ...getGroupedSumNodes(graph, {
      id: `${id}-variance`,
      operation,
      rows,
      groupCount: 1,
      bindings: [
        {name: 'response', view: props.response, type: 'f32', access: 'read'},
        {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read'},
        {name: 'scaledDeviation', view: statistics[2], type: 'f32', access: 'read'},
        {name: 'responseAnchor', view: responseAnchor, type: 'f32', access: 'read'},
        {name: 'observationCount', view: statistics[3], type: 'f32', access: 'read'}
      ],
      outputs: [totalSquares],
      rowSnippet: `if (localStatus[localStatusOffset + row] == 0u) {
    group = 0u;
    let meanDeviation = scaledDeviation[scaledDeviationOffset] / (observationCount[observationCountOffset] / f32(ROWS));
    let deviation = 2.0 * ((response[responseOffset + row] * 0.5 - responseAnchor[responseAnchorOffset] * 0.5) - meanDeviation);
    value0 = deviation * deviation;
  }`,
      foldSuffix: 'summary',
      finish: {
        bindings: [
          {name: 'residualSquares', view: statistics[0], type: 'f32', access: 'read'},
          {name: 'hatTrace', view: statistics[1], type: 'f32', access: 'read'},
          {name: 'observationCount', view: statistics[3], type: 'f32', access: 'read'},
          {name: 'includedCount', view: includedCount, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `let observations = observationCount[observationCountOffset];
  let included = f32(includedCount[includedCountOffset]);
  let residualSum = residualSquares[residualSquaresOffset];
  let trace = hatTrace[hatTraceOffset];
  let logVariance = log(residualSum) - log(observations);
  let logLikelihood = -0.5 * observations * (logVariance + 1.8378770664093453 + 1.0);
  let correctedDenominator = observations - trace - 2.0;
  let corrected = -2.0 * logLikelihood + 2.0 * observations * (trace + 1.0) / correctedDenominator;
  for (var slot = 0u; slot < 12u; slot++) { summary[summaryOffset + slot] = getNaN(); }
  summary[summaryOffset] = residualSum;
  summary[summaryOffset + 1u] = trace;
  summary[summaryOffset + 2u] = select(getNaN(), corrected, observations > 0.0 && correctedDenominator > 0.0);
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - residualSum / sum0, sum0 > 0.0);
  summary[summaryOffset + 4u] = observations;
  summary[summaryOffset + 6u] = included - observations;
  summary[summaryOffset + 7u] = select(getNaN(), residualSum / (observations - trace), observations > trace);
  summary[summaryOffset + 10u] = select(getNaN(), logLikelihood, observations > 0.0);
  summary[summaryOffset + 11u] = included;`
      }
    })
  ];
}
