// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeneralizedGeographicallyWeightedRegression,
  getRegressionBindings,
  getRegressionWGSL
} from './gpu-generalized-geographically-weighted-regression';

/** GPU-only bounded exact leave-one-out deviance search. @internal */
export function getGeneralizedSelectionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUGeneralizedGeographicallyWeightedRegression
): readonly GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const selection = props.bandwidthSelection!;
  const rows = props.response.length;
  const columns = props.predictorCount + 1;
  const nodes: GPUCommandNode<Parameters>[] = [];
  validateGraphViewsBelongToGraph(id, graph, Object.values(selection.output));
  const parameters = createTransientView(graph, `${id}-selected-parameters`, 'float32', 2);
  const coefficients = createTransientView(
    graph,
    `${id}-selection-coefficients`,
    'float32',
    rows * columns
  );
  const fitted = createTransientView(graph, `${id}-selection-fitted`, 'float32', rows);
  const codes = createTransientView(graph, `${id}-selection-codes`, 'uint32', rows);
  const iterationCount = createTransientView(graph, `${id}-selection-iterations`, 'uint32', rows);
  const fittedScore = createTransientView(graph, `${id}-selection-row-scores`, 'float32', rows);
  for (const [candidate, bandwidth] of selection.bandwidths.entries()) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-${candidate}-parameters`,
        operation: 'GPUGeneralizedGeographicallyWeightedRegression',
        variant: 'selection-parameters',
        invocationCount: 1,
        bindings: [
          {name: 'source', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}
        ],
        body: `parameters[parametersOffset] = ${getWGSLFloatLiteral(bandwidth)}; parameters[parametersOffset + 1u] = source[sourceOffset + 1u];`
      })
    );
    nodes.push(
      ...new GPUGeneralizedGeographicallyWeightedRegression({
        ...props,
        id: `${id}-candidate-${candidate}`,
        bandwidthSelection: undefined,
        inference: undefined,
        leaveOneOut: true,
        predictExcluded: false,
        // The Bernoulli deviance is taken from the linear predictor (see the deviance node).
        fittedScale: props.family === 'logistic' ? 'link' : 'response',
        parameters,
        output: {coefficients, fitted, status: {codes, iterationCount}}
      }).getCommandNodes(graph)
    );
    // Split scores from eligibility so offset and mask plus all model inputs stay within eight bindings.
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-${candidate}-deviance`,
        operation: 'GPUGeneralizedGeographicallyWeightedRegression',
        variant: 'selection-deviance',
        invocationCount: rows,
        bindings: [
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'fitted', view: fitted, type: 'f32', access: 'read'},
          {name: 'codes', view: codes, type: 'u32', access: 'read'},
          {name: 'scores', view: fittedScore, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        // Poisson: unit deviance from the fitted mean. Logistic: `fitted` holds the linear
        // predictor eta and the Bernoulli deviance is 2 log(1 + exp(-eta)) for a response of 1
        // and 2 log(1 + exp(eta)) for 0, finite for every finite eta. The float32 fitted mean
        // rounds to exactly 1 from eta of about 17, where -2 log(1 - mean) is infinite and
        // -2 log(mean) is 0, so a mean cannot carry a confident prediction.
        body:
          props.family === 'poisson'
            ? `scores[scoresOffset + index] = getNaN(); if (codes[codesOffset + index] != 0u) { return; } let observed = response[responseOffset + index]; let mean = fitted[fittedOffset + index]; if (!isFiniteBits(mean) || !(mean > 0.0)) { return; } scores[scoresOffset + index] = 2.0 * (select(0.0, observed * log(max(observed, 1.0e-30) / mean), observed > 0.0) - observed + mean);`
            : `scores[scoresOffset + index] = getNaN(); if (codes[codesOffset + index] != 0u) { return; } let observed = response[responseOffset + index]; let linear = fitted[fittedOffset + index]; if (!isFiniteBits(linear)) { return; } let opposed = select(linear, -linear, observed == 1.0); scores[scoresOffset + index] = 2.0 * (max(opposed, 0.0) + log(1.0 + exp(-abs(opposed))));`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-${candidate}-score`,
        operation: 'GPUGeneralizedGeographicallyWeightedRegression',
        variant: 'selection-mean-deviance',
        invocationCount: 1,
        bindings: [
          ...getRegressionBindings(props),
          {name: 'rowScores', view: fittedScore, type: 'f32', access: 'read'},
          {name: 'scores', view: selection.output.scores, type: 'f32', access: 'read_write'}
        ],
        declarations: getRegressionWGSL(props),
        body: `var total = 0.0; var count = 0u; var valid = true; for (var row = 0u; row < ${rows}u; row++) { if (!included(row)) { continue; } let observed = response[responseOffset + row]; if (!(${props.family === 'poisson' ? 'observed >= 0.0 && floor(observed) == observed' : 'observed == 0.0 || observed == 1.0'})) { continue; } let score = rowScores[rowScoresOffset + row]; if (!isFiniteBits(score)) { valid = false; } total += score; count++; } scores[scoresOffset + ${candidate}u] = getNaN(); if (valid && count > 0u) { scores[scoresOffset + ${candidate}u] = total / f32(count); }`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-select-bandwidth`,
      operation: 'GPUGeneralizedGeographicallyWeightedRegression',
      variant: 'select-bandwidth',
      invocationCount: 1,
      bindings: [
        {name: 'scores', view: selection.output.scores, type: 'f32', access: 'read'},
        {name: 'bandwidth', view: selection.output.bandwidth, type: 'f32', access: 'read_write'},
        {name: 'status', view: selection.output.status, type: 'u32', access: 'read_write'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `var best = 3.402823e38; var chosen = getNaN(); ${selection.bandwidths.map((value, index) => `if (isFiniteBits(scores[scoresOffset + ${index}u]) && scores[scoresOffset + ${index}u] < best) { best = scores[scoresOffset + ${index}u]; chosen = ${getWGSLFloatLiteral(value)}; }`).join('\n')} bandwidth[bandwidthOffset] = chosen; parameters[parametersOffset] = chosen; status[statusOffset] = select(1u, 0u, isFiniteBits(chosen));`
    })
  );
  nodes.push(
    ...new GPUGeneralizedGeographicallyWeightedRegression({
      ...props,
      id: `${id}-selected-fit`,
      bandwidthSelection: undefined,
      parameters
    }).getCommandNodes(graph)
  );
  return nodes;
}
