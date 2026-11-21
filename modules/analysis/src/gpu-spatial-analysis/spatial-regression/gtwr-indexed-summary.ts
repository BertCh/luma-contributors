// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createWGSLKernelNode,
  createWGSLDispatchGate,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes,
  type WGSLDispatchGate
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GTWR_SUMMARY_PUBLICATION_WGSL} from './gwr-summary-wgsl';

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {getBoundsReductionNodes} from '../hilbert-keys/bounds-reduction';
import {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS} from './geographically-weighted-regression-parameters';
import {
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getGroupedSumNodes} from '../geographic-distribution/grouped-sum-reduction';
import type {GPUGeographicallyTemporallyWeightedRegressionProps} from './gpu-geographically-temporally-weighted-regression';

/** Fixed-tree parallel statistics with a second pass for centered total squares. @internal */
export function createGTWRIndexedSummaryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUGeographicallyTemporallyWeightedRegressionProps,
  residuals?: GraphDataView<'float32'>,
  parentGate?: WGSLDispatchGate<Parameters>
): GPUCommandNode<Parameters>[] {
  // Retarget cooperative kernels using their exact recorded invocation/workgroup shapes.
  if (parentGate) recordWGSLKernelNodes(graph);
  const rows = props.response.length;
  const operation = 'GPUGeographicallyTemporallyWeightedRegression';
  const {output} = props;
  const centeredResiduals = residuals ?? output.residuals;
  validateRequirement(
    id,
    Boolean(centeredResiduals),
    'centered residuals must be supplied to summarize GTWR; use the residuals from the fit, not rounded fitted values'
  );
  validateSinglePackedView(centeredResiduals!, ['float32'], `${id} centeredResiduals`);
  validateViewLength(id, 'centeredResiduals', centeredResiduals!, rows);
  const statistics = Array.from({length: 4}, (_, index) =>
    createTransientView(graph, `${id}-statistic-${index}`, 'float32', 1)
  );
  const includedCount = createTransientView(graph, `${id}-included-count`, 'uint32', 1);
  const responseAnchor = createTransientView(graph, `${id}-response-anchor`, 'float32', 4);
  const totalSquares = createTransientView(graph, `${id}-total-squares`, 'float32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
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
    select(0.0, 1.0, localStatus[localStatusOffset + row] == ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u), 0.0);
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
        {name: 'residuals', view: centeredResiduals!, type: 'f32', access: 'read'},
        {name: 'hatDiagonal', view: output.hatDiagonal, type: 'f32', access: 'read'},
        {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read'}
      ],
      outputs: statistics,
      countOutput: includedCount,
      rowSnippet: `let code = localStatus[localStatusOffset + row];
  if (code != ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED}u) { group = 0u; }
  if (code == ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u) {
    let observed = response[responseOffset + row];
    let residual = residuals[residualsOffset + row];
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
      rowSnippet: `if (localStatus[localStatusOffset + row] == ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u) {
    group = 0u;
    let meanDeviation = scaledDeviation[scaledDeviationOffset] / (observationCount[observationCountOffset] / f32(ROWS));
    let deviation = 2.0 * ((response[responseOffset + row] * 0.5 - responseAnchor[responseAnchorOffset] * 0.5) - meanDeviation);
    value0 = deviation * deviation;
  }`,
      foldSuffix: 'summary',
      finish: {
        bindings: [
          {name: 'residualSums', view: statistics[0], type: 'f32', access: 'read'},
          {name: 'hatTrace', view: statistics[1], type: 'f32', access: 'read'},
          {name: 'observationCount', view: statistics[3], type: 'f32', access: 'read'},
          {name: 'includedCount', view: includedCount, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `let count = observationCount[observationCountOffset];
  let included = f32(includedCount[includedCountOffset]);
  let singular = included - count;
  let residualSquares = residualSums[residualSumsOffset];
  let trace = hatTrace[hatTraceOffset];
  let totalSquares = sum0;
${GTWR_SUMMARY_PUBLICATION_WGSL}`
      }
    })
  ];
  if (!parentGate) return nodes;
  const records = nodes.map(node => getWGSLKernelNodeRecord(graph, node.id)!);
  const gate = createWGSLDispatchGate<Parameters>(graph, {
    id: `${id}-summary-gate`,
    operation,
    shapes: records.map(record => [record.invocationCount, record.workgroupSize[0]] as const)
  });
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-summary-arm`,
      operation,
      invocationCount: 1,
      bindings: [
        {name: 'parent', view: parentGate.view, type: 'u32', access: 'read'},
        gate.binding
      ],
      body: `if (parent[parentOffset] != 0u) { ${gate.armWGSL} } else { ${gate.disarmWGSL} }`
    }),
    ...records.map((record, index) =>
      createWGSLKernelNode(graph, {
        id: record.id,
        operation: record.operation,
        variant: record.variant,
        invocationCount: record.invocationCount,
        workgroupSize: record.workgroupSize[0],
        guardIndex: false,
        bindings: record.bindings,
        declarations: record.declarations,
        body: record.body,
        constants: record.constants,
        ...gate.getGate(index)
      })
    )
  ];
}
