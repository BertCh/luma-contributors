// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import {GPUSpatialErrorGM} from './gpu-spatial-error-gm';
import {GPUSeeminglyUnrelatedRegression} from './gpu-seemingly-unrelated-regression';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from './ordinary-least-squares-parameters';

/** Spatial SUR composes the existing GM moment optimizer with a joint filtered-residual GLS. */
export function getSpatialSURNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  model: GPUSeeminglyUnrelatedRegression
): GPUCommandNode<Parameters>[] {
  const {id, props} = model;
  const spatial = props.spatial!;
  const rows = props.rowCount;
  const equations = props.equationCount;
  const columns = props.predictorCount;
  const width = columns + 1;
  const size = equations * width;
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const slice = <Format extends 'float32' | 'uint32'>(
    source: GraphDataView<Format>,
    offset: number,
    length: number
  ) =>
    graph.createDataView(source.buffer, {
      format: source.format,
      byteOffset: source.byteOffset + offset * 4,
      length
    });
  const filteredPredictors = view('filtered-predictors', 'float32', equations * rows * columns);
  const filteredResponses = view('filtered-responses', 'float32', equations * rows);
  const filteredResiduals = view('filtered-residuals', 'float32', equations * rows);
  const coefficients = view('filtered-coefficients', 'float32', size);
  const covariance = view('filtered-covariance', 'float32', size * size);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let equation = 0; equation < equations; equation++) {
    const predictors = slice(props.predictors, equation * rows * columns, rows * columns);
    const response = slice(props.responses, equation * rows, rows);
    const residuals = view(`initial-residuals-${equation}`, 'float32', rows);
    nodes.push(
      ...new GPUOrdinaryLeastSquares({
        id: `${id}-ols-${equation}`,
        predictors,
        response,
        predictorCount: columns,
        output: {
          coefficients: view(`ols-coefficients-${equation}`, 'float32', width),
          standardErrors: view(`ols-errors-${equation}`, 'float32', width),
          tStatistics: view(`ols-tests-${equation}`, 'float32', width),
          summary: view(
            `ols-summary-${equation}`,
            'float32',
            GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
          ),
          status: {code: view(`ols-status-${equation}`, 'uint32', 1)},
          residuals
        }
      }).getCommandNodes(graph)
    );
    const table = view(`gm-table-${equation}`, 'float32', (columns + 2) * 4);
    const status = slice(spatial.output.status, equation, 1);
    nodes.push(
      ...new GPUSpatialErrorGM({
        id: `${id}-gm-${equation}`,
        predictors,
        response,
        predictorCount: columns,
        weights: spatial.weights,
        estimator: 'kelejian-prucha',
        output: {
          table,
          summary: view(`gm-summary-${equation}`, 'float32', 13),
          status: {code: status}
        }
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-parameter-${equation}`,
        operation: 'GPUSeeminglyUnrelatedRegression',
        variant: 'spatial-error-parameter',
        invocationCount: 1,
        bindings: [
          {name: 'offsets', view: spatial.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: spatial.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: spatial.weights.weights, type: 'f32', access: 'read'},
          {name: 'table', view: table, type: 'f32', access: 'read'},
          {name: 'status', view: status, type: 'u32', access: 'read_write'},
          {
            name: 'parameters',
            view: slice(spatial.output.parameters, equation, 1),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `parameters[parametersOffset] = getNaN();
for (var row = 0u; row < ${rows}u; row++) { let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u]; if (end < begin || end > ${spatial.weights.neighbors.length}u) { status[statusOffset] = 4u; return; } var total = 0.0;
for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ${rows}u || neighbor == row || !isFiniteBits(weight) || weight < 0.0 || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { status[statusOffset] = 4u; return; } total += weight; }
if (abs(total - 1.0) > 1e-5) { status[statusOffset] = 4u; return; } }
if (status[statusOffset] == 0u) { parameters[parametersOffset] = table[tableOffset + ${width * 4}u]; }`
      })
    );
    // Pack y, X and OLS errors so filtering stays within eight storage bindings.
    const packed = view(`unfiltered-${equation}`, 'float32', rows * (columns + 2));
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pack-${equation}`,
        operation: 'GPUSeeminglyUnrelatedRegression',
        variant: 'pack-spatial-system',
        invocationCount: rows,
        bindings: [
          {name: 'predictors', view: predictors, type: 'f32', access: 'read'},
          {name: 'response', view: response, type: 'f32', access: 'read'},
          {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
          {name: 'packed', view: packed, type: 'f32', access: 'read_write'}
        ],
        body: `packed[packedOffset + index * ${columns + 2}u] = response[responseOffset + index]; packed[packedOffset + index * ${columns + 2}u + 1u] = residuals[residualsOffset + index]; for (var column = 0u; column < ${columns}u; column++) { packed[packedOffset + index * ${columns + 2}u + column + 2u] = predictors[predictorsOffset + index * ${columns}u + column]; }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-filter-${equation}`,
        operation: 'GPUSeeminglyUnrelatedRegression',
        variant: 'spatial-error-filter',
        invocationCount: rows,
        bindings: [
          {name: 'offsets', view: spatial.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: spatial.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: spatial.weights.weights, type: 'f32', access: 'read'},
          {name: 'packed', view: packed, type: 'f32', access: 'read'},
          {
            name: 'parameter',
            view: slice(spatial.output.parameters, equation, 1),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'predictors',
            view: slice(filteredPredictors, equation * rows * columns, rows * columns),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'response',
            view: slice(filteredResponses, equation * rows, rows),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'residuals',
            view: slice(filteredResiduals, equation * rows, rows),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: `let parameterValue = parameter[parameterOffset]; for (var column = 0u; column < ${columns + 2}u; column++) { var value = packed[packedOffset + index * ${columns + 2}u + column]; for (var slot = offsets[offsetsOffset + index]; slot < min(offsets[offsetsOffset + index + 1u], ${spatial.weights.neighbors.length}u); slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor < ${rows}u) { value -= parameterValue * weights[weightsOffset + slot] * packed[packedOffset + neighbor * ${columns + 2}u + column]; } } if (column == 0u) { response[responseOffset + index] = value; } else if (column == 1u) { residuals[residualsOffset + index] = value; } else { predictors[predictorsOffset + index * ${columns}u + column - 2u] = value; } }`
      })
    );
  }
  nodes.push(
    ...new GPUSeeminglyUnrelatedRegression({
      id: `${id}-filtered-system`,
      predictors: filteredPredictors,
      responses: filteredResponses,
      initialResiduals: filteredResiduals,
      equationCount: equations,
      predictorCount: columns,
      rowCount: rows,
      output: {
        coefficients,
        covariance,
        residualCovariance: props.output.residualCovariance,
        residuals: view('filtered-final-residuals', 'float32', rows * equations),
        status: props.output.status
      }
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unfilter-coefficients`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'restore-spatial-intercepts',
      invocationCount: size * size,
      bindings: [
        {name: 'parameters', view: spatial.output.parameters, type: 'f32', access: 'read'},
        {name: 'sourceCoefficients', view: coefficients, type: 'f32', access: 'read'},
        {name: 'sourceCovariance', view: covariance, type: 'f32', access: 'read'},
        {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read_write'},
        {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read_write'}
      ],
      body: `let first = index / ${size}u; let second = index % ${size}u; let firstScale = select(1.0, 1.0 / (1.0 - parameters[parametersOffset + first / ${width}u]), first % ${width}u == 0u); let secondScale = select(1.0, 1.0 / (1.0 - parameters[parametersOffset + second / ${width}u]), second % ${width}u == 0u); covariance[covarianceOffset + index] = sourceCovariance[sourceCovarianceOffset + index] * firstScale * secondScale; if (index < ${size}u) { coefficients[coefficientsOffset + index] = sourceCoefficients[sourceCoefficientsOffset + index] * secondScale; }`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-residuals`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'structural-spatial-residuals',
      invocationCount: equations * rows,
      bindings: [
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'responses', view: props.responses, type: 'f32', access: 'read'},
        {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read'},
        {name: 'residuals', view: props.output.residuals, type: 'f32', access: 'read_write'}
      ],
      body: `let equation = index / ${rows}u; var residual = responses[responsesOffset + index] - coefficients[coefficientsOffset + equation * ${width}u]; for (var column = 0u; column < ${columns}u; column++) { residual -= predictors[predictorsOffset + index * ${columns}u + column] * coefficients[coefficientsOffset + equation * ${width}u + column + 1u]; } residuals[residualsOffset + index] = residual;`
    })
  );
  return nodes;
}
