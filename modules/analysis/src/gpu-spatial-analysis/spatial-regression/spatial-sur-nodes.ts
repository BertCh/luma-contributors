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
import {
  GPUOrdinaryLeastSquares,
  getOrdinaryLeastSquaresAllocationLengths
} from './gpu-ordinary-least-squares';
import {GPUSpatialErrorGM} from './gpu-spatial-error-gm';
import {GPUSeeminglyUnrelatedRegression} from './gpu-seemingly-unrelated-regression';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from './ordinary-least-squares-parameters';
import {getOrdinarySURScratchLength} from './ordinary-sur-nodes';

/** Complete Kelejian-Prucha SUR composition plan; this route does not request Gram probes. */
export function getSpatialSURAllocationLengths(
  rowCount: number,
  equationCount: number,
  predictorCount: number
): Readonly<Record<string, number>> {
  const {tileCount} = getRegressionTileLayout(rowCount);
  const width = predictorCount + 1;
  const size = equationCount * width;
  const filteredMoments = width * width + width + 1;
  return {
    ...getOrdinaryLeastSquaresAllocationLengths(rowCount, predictorCount),
    'SUR filtered predictors': equationCount * rowCount * predictorCount,
    'SUR filtered responses/residuals': equationCount * rowCount,
    'SUR filtered coefficients': size,
    'SUR filtered covariance': size * size,
    'SUR unfiltered equation': rowCount * (predictorCount + 2),
    'SUR statistics scratch': getOrdinarySURScratchLength(equationCount, predictorCount, rowCount),
    'SUR weight partials': tileCount,
    'SUR GM table': (predictorCount + 2) * 4,
    'SUR GM summary': 13,
    'SUR OLS summary': GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
    'SUR GM workspace': predictorCount + 4 + width + 16,
    'SUR GM mean partials': tileCount * width,
    'SUR GM moment partials': tileCount * 7,
    'SUR GM moments': 7,
    'SUR GM filtered partials': tileCount * filteredMoments,
    'SUR GM filtered moments': filteredMoments,
    'SUR GM residual partials': tileCount * 5,
    'SUR GM residual totals': 5
  };
}

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
  const invalidWeights = view('invalid-weights', 'float32', 1);
  const nodes: GPUCommandNode<Parameters>[] = getSpatialSURWeightValidationNodes(graph, {
    id,
    weights: spatial.weights,
    rowCount: rows,
    output: invalidWeights
  });
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
          {name: 'invalidWeights', view: invalidWeights, type: 'f32', access: 'read'},
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
if (invalidWeights[invalidWeightsOffset] != 0.0) { status[statusOffset] = 4u; return; }
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

/** Shared strict row-standardized CSR check for every spatial SUR equation. @internal */
export function getSpatialSURWeightValidationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    weights: GPUSpatialWeights;
    rowCount: number;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  validateGraphChunkBindingRanges(graph, props.id, {
    offsets: props.weights.offsets,
    neighbors: props.weights.neighbors,
    weights: props.weights.weights,
    overflow: props.weights.status?.overflow,
    output: props.output
  });
  const layout = getRegressionTileLayout(props.rowCount);
  validateGPUStorageBindingByteLength(
    graph.device,
    props.id,
    'weightValidationPartial',
    layout.tileCount * 4,
    'reduce rowCount'
  );
  return getRowTileSumNodes(graph, {
    id: `${props.id}-validate-weights`,
    operation: 'GPUSeeminglyUnrelatedRegression',
    variant: 'validate-spatial-weights',
    rowCount: props.rowCount,
    layout,
    sumCount: 0,
    maximumCount: 1,
    partial: createTransientView(
      graph,
      `${props.id}-weight-validation-partial`,
      'float32',
      layout.tileCount
    ),
    output: props.output,
    bindings: [
      {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: props.weights.weights, type: 'f32', access: 'read'},
      ...(props.weights.status?.overflow
        ? [
            {
              name: 'overflow',
              view: props.weights.status.overflow,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ],
    declarations: `${IS_FINITE_BITS_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  ${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { (*terms)[0] = 1.0; return; }' : ''}
  let begin = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
  if (end < begin || end > ${props.weights.neighbors.length}u) { (*terms)[0] = 1.0; return; }
  var total = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    if (neighbor >= ROW_COUNT || neighbor == row || !isFiniteBits(weight) || weight < 0.0 || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { (*terms)[0] = 1.0; return; }
    total += weight;
  }
  if (abs(total - 1.0) > 1e-5) { (*terms)[0] = 1.0; }
}`
  });
}
