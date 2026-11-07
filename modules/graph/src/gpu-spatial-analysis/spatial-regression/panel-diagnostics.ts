// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {ERFC_VALUE_WGSL, GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import type {GPUSpatialPanelErrorMaximumLikelihood} from './gpu-spatial-panel-error-maximum-likelihood';

function binding(
  name: string,
  view: GraphDataView<'float32'> | GraphDataView<'uint32'>,
  access: 'read' | 'read_write' = 'read',
  type: 'f32' | 'u32' = 'f32'
): WGSLKernelBinding {
  return {name, view, access, type};
}
/** Pooled complete-panel diagnostic definitions in spreg 1.8.3. @internal */
export function getPanelDiagnosticNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUSpatialPanelErrorMaximumLikelihood
): readonly GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const diagnostics = props.diagnostics!;
  const rows = props.response.length;
  const entities = props.weights.offsets.length - 1;
  const columns = props.predictorCount + 1;
  const edges = props.weights.neighbors.length;
  validateGraphViewsBelongToGraph(id, graph, Object.values(diagnostics));
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-diagnostic-${name}`, 'float32', length);
  const residuals = view('residuals', rows),
    fitted = view('fitted', rows),
    lags = view('lags', rows * 4),
    laggedFit = view('lagged-fit', rows),
    projectedResiduals = view('projected-residuals', rows),
    summary = view('summary', 22);
  const status = createTransientView(graph, `${id}-diagnostic-fit-status`, 'uint32', 1),
    projectionStatus = createTransientView(
      graph,
      `${id}-diagnostic-projection-status`,
      'uint32',
      1
    );
  const nodes: GPUCommandNode<Parameters>[] = [
    ...new GPUOrdinaryLeastSquares({
      id: `${id}-diagnostic-pooled`,
      predictors: props.predictors,
      predictorCount: props.predictorCount,
      response: props.response,
      output: {
        coefficients: view('coefficients', columns),
        standardErrors: view('errors', columns),
        tStatistics: view('statistics', columns),
        summary,
        residuals,
        fitted,
        status: {code: status}
      }
    }).getCommandNodes(graph)
  ];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-lags`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'pooled-diagnostic-lags',
      invocationCount: rows,
      bindings: [
        binding('offsets', props.weights.offsets, 'read', 'u32'),
        binding('neighbors', props.weights.neighbors, 'read', 'u32'),
        binding('weights', props.weights.weights),
        binding('response', props.response),
        binding('residuals', residuals),
        binding('fitted', fitted),
        binding('lags', lags, 'read_write'),
        ...(props.weights.status?.overflow
          ? [binding('overflow', props.weights.status.overflow, 'read', 'u32')]
          : [])
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `let entity = index % ${entities}u; let periodStart = index - entity; let base = lagsOffset + index * 4u; for (var component = 0u; component < 4u; component++) { lags[base + component] = getNaN(); } ${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { return; }' : ''} let begin = offsets[offsetsOffset + entity]; let end = offsets[offsetsOffset + entity + 1u]; if (end < begin || end > ${edges}u) { return; } var lagResponse = 0.0; var lagResidual = 0.0; var lagFitted = 0.0; var trace = 0.0; for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ${entities}u || neighbor == entity || !isFiniteBits(weight) || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { return; } var reciprocal = 0.0; for (var other = offsets[offsetsOffset + neighbor]; other < min(offsets[offsetsOffset + neighbor + 1u], ${edges}u); other++) { if (neighbors[neighborsOffset + other] == entity) { reciprocal = weights[weightsOffset + other]; } } trace += weight * (weight + reciprocal); lagResponse += weight * response[responseOffset + periodStart + neighbor]; lagResidual += weight * residuals[residualsOffset + periodStart + neighbor]; lagFitted += weight * fitted[fittedOffset + periodStart + neighbor]; } lags[base] = lagResponse; lags[base + 1u] = lagResidual; lags[base + 2u] = lagFitted; lags[base + 3u] = trace;`
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-project-input`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'diagnostic-project-input',
      invocationCount: rows,
      bindings: [binding('lags', lags), binding('laggedFit', laggedFit, 'read_write')],
      body: 'laggedFit[laggedFitOffset + index] = lags[lagsOffset + index * 4u + 2u];'
    })
  );
  nodes.push(
    ...new GPUOrdinaryLeastSquares({
      id: `${id}-diagnostic-projection`,
      predictors: props.predictors,
      predictorCount: props.predictorCount,
      response: laggedFit,
      output: {
        coefficients: view('projection-coefficients', columns),
        standardErrors: view('projection-errors', columns),
        tStatistics: view('projection-statistics', columns),
        summary: view('projection-summary', 22),
        residuals: projectedResiduals,
        status: {code: projectionStatus}
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-diagnostic-tests`,
      operation: 'GPUSpatialPanelErrorMaximumLikelihood',
      variant: 'pooled-panel-lm-tests',
      invocationCount: 1,
      bindings: [
        binding('residuals', residuals),
        binding('lags', lags),
        binding('projected', projectedResiduals),
        binding('summary', summary),
        binding('fitStatus', status, 'read', 'u32'),
        binding('projectionStatus', projectionStatus, 'read', 'u32'),
        binding('tests', diagnostics.tests, 'read_write'),
        binding('status', diagnostics.status, 'read_write', 'u32')
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${ERFC_VALUE_WGSL}
fn upperChiSquareOne(statistic:f32)->f32 { return erfcValue(sqrt(max(0.0,statistic)*0.5)); }`,
      body: `status[statusOffset] = 1u; for (var test = 0u; test < 6u; test++) { tests[testsOffset + test] = getNaN(); } if (fitStatus[fitStatusOffset] != 0u || projectionStatus[projectionStatusOffset] != 0u || summary[summaryOffset] != ${rows}.0) { return; } var residualLag = 0.0; var responseLag = 0.0; var trace = 0.0; var projectedSquares = 0.0; for (var row = 0u; row < ${rows}u; row++) { let base = lagsOffset + row * 4u; residualLag += residuals[residualsOffset + row] * lags[base + 1u]; responseLag += residuals[residualsOffset + row] * lags[base]; trace += lags[base + 3u]; projectedSquares += projected[projectedOffset + row] * projected[projectedOffset + row]; } let variance = summary[summaryOffset + 3u]; let information = projectedSquares / variance + trace; status[statusOffset] = 2u; if (!(variance > 0.0 && trace > 0.0 && information > trace) || !isFiniteBits(information) || !isFiniteBits(residualLag) || !isFiniteBits(responseLag)) { return; } let error = residualLag * residualLag / (variance * variance * trace); let lagContrast = (responseLag - residualLag) / variance; let robustLag = lagContrast * lagContrast / (information - trace); let errorContrast = residualLag / variance - trace / information * responseLag / variance; let robustError = errorContrast * errorContrast / (trace * (1.0-trace/information)); tests[testsOffset] = error; tests[testsOffset + 1u] = upperChiSquareOne(error); tests[testsOffset + 2u] = robustLag; tests[testsOffset + 3u] = upperChiSquareOne(robustLag); tests[testsOffset + 4u] = robustError; tests[testsOffset + 5u] = upperChiSquareOne(robustError); status[statusOffset] = 0u;`
    })
  );
  return nodes;
}
