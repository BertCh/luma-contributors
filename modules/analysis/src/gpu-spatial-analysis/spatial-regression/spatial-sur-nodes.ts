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
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

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
    'SUR centered responses': equationCount * rowCount,
    'SUR centered predictors': equationCount * rowCount * predictorCount,
    'SUR filtered coefficients': size,
    'SUR convergence previous coefficients': size,
    'SUR convergence disabled tolerance': 1,
    'SUR filtered covariance': size * size,
    'SUR unfiltered equation': rowCount * (predictorCount + 2),
    'SUR statistics scratch': getOrdinarySURScratchLength(equationCount, predictorCount, rowCount),
    'SUR weight partials': tileCount,
    'SUR GM table': (predictorCount + 2) * 4,
    'SUR GM summary': 13,
    'SUR OLS summary': GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
    'SUR GM workspace': predictorCount + 4 + width + 18,
    'SUR GM mean partials': tileCount * width,
    'SUR GM moment partials': tileCount * 14,
    'SUR GM moments': 14,
    'SUR GM residual lag': 2 * rowCount,
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
  const centeredPredictors = view('centered-predictors', 'float32', equations * rows * columns);
  const centeredResponses = view('centered-responses', 'float32', equations * rows);
  const filteredResiduals = view('filtered-residuals', 'float32', equations * rows);
  const coefficients = view('filtered-coefficients', 'float32', size);
  const covariance = view('filtered-covariance', 'float32', size * size);
  const invalidWeights = view('invalid-weights', 'float32', 1);
  // Its size equals the already-preflighted caller weights binding.
  const normalizedValues = view('normalized-weights', 'float32', spatial.weights.weights.length);
  const normalizedWeights = {...spatial.weights, weights: normalizedValues};
  const nodes: GPUCommandNode<Parameters>[] = getSpatialSURWeightValidationNodes(graph, {
    id,
    weights: spatial.weights,
    rowCount: rows,
    output: invalidWeights
  });
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-normalize-weights`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'normalize-accepted-rows',
      invocationCount: rows,
      bindings: [
        {name: 'offsets', view: spatial.weights.offsets, type: 'u32', access: 'read'},
        {name: 'sourceWeights', view: spatial.weights.weights, type: 'f32', access: 'read'},
        {name: 'normalizedWeights', view: normalizedValues, type: 'f32', access: 'read_write'}
      ],
      body: `let begin = min(offsets[offsetsOffset + index], ${spatial.weights.weights.length}u);
let end = min(offsets[offsetsOffset + index + 1u], ${spatial.weights.weights.length}u);
var total = 0.0;
for (var slot = begin; slot < end; slot++) { total += sourceWeights[sourceWeightsOffset + slot]; }
let divisor = select(1.0, total, total > 0.0);
for (var slot = begin; slot < end; slot++) { normalizedWeights[normalizedWeightsOffset + slot] = sourceWeights[sourceWeightsOffset + slot] / divisor; }`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center-responses`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'shift-spatial-responses',
      invocationCount: equations * rows,
      bindings: [
        {name: 'responses', view: props.responses, type: 'f32', access: 'read'},
        {name: 'centeredResponses', view: centeredResponses, type: 'f32', access: 'read_write'}
      ],
      declarations: IS_FINITE_BITS_WGSL,
      body: `let origin = responses[responsesOffset + (index / ${rows}u) * ${rows}u];
centeredResponses[centeredResponsesOffset + index] = responses[responsesOffset + index] - select(0.0, origin, isFiniteBits(origin));`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center-predictors`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'shift-spatial-predictors',
      invocationCount: equations * rows * columns,
      bindings: [
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'invalidWeights', view: invalidWeights, type: 'f32', access: 'read'},
        {name: 'centered', view: centeredPredictors, type: 'f32', access: 'read_write'}
      ],
      declarations: ROUNDED_DIFFERENCE_WGSL,
      body: `let equation = index / ${rows * columns}u;
let column = index % ${columns}u;
centered[centeredOffset + index] = getRoundedDifference(predictors[predictorsOffset + index], predictors[predictorsOffset + equation * ${rows * columns}u + column], bitcast<u32>(invalidWeights[invalidWeightsOffset]));`
    })
  );
  for (let equation = 0; equation < equations; equation++) {
    const predictors = slice(centeredPredictors, equation * rows * columns, rows * columns);
    const response = slice(centeredResponses, equation * rows, rows);
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
      }).getCommandNodes(graph, true)
    );
    const table = view(`gm-table-${equation}`, 'float32', (columns + 2) * 4);
    const status = slice(spatial.output.status, equation, 1);
    nodes.push(
      ...new GPUSpatialErrorGM({
        id: `${id}-gm-${equation}`,
        predictors,
        response,
        predictorCount: columns,
        weights: normalizedWeights,
        estimator: 'kelejian-prucha',
        output: {
          table,
          summary: view(`gm-summary-${equation}`, 'float32', 13),
          status: {code: status}
        }
      }).getCommandNodes(graph, true)
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
        body: `packed[packedOffset + index * ${columns + 2}u] = response[responseOffset + index] - response[responseOffset]; packed[packedOffset + index * ${columns + 2}u + 1u] = residuals[residualsOffset + index]; for (var column = 0u; column < ${columns}u; column++) { packed[packedOffset + index * ${columns + 2}u + column + 2u] = predictors[predictorsOffset + index * ${columns}u + column]; }`
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
          {name: 'weights', view: normalizedValues, type: 'f32', access: 'read'},
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
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `let parameterValue = parameter[parameterOffset];
let runtimeZero = offsets[offsetsOffset];  // Zero for every admitted spatial SUR graph.
for (var column = 0u; column < ${columns + 2}u; column++) {
  var lag = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < min(offsets[offsetsOffset + index + 1u], ${spatial.weights.neighbors.length}u); slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ${rows}u) { lag += weights[weightsOffset + slot] * packed[packedOffset + neighbor * ${columns + 2}u + column]; }
  }
  let original = packed[packedOffset + index * ${columns + 2}u + column];
  // Keep the lag dot product separate from the update to the original value.
  let roundedLag = bitcast<f32>(bitcast<u32>(lag) ^ runtimeZero);
  let value = select(getNaN(), fma(-parameterValue, roundedLag, original), isFiniteBits(parameterValue));
  if (column == 0u) { response[responseOffset + index] = value; }
  else if (column == 1u) { residuals[residualsOffset + index] = value; }
  else { predictors[predictorsOffset + index * ${columns}u + column - 2u] = value; }
}`
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
      iteration: props.iteration,
      parameters: props.parameters,
      output: {
        coefficients,
        covariance,
        residualCovariance: props.output.residualCovariance,
        residuals: view('filtered-final-residuals', 'float32', rows * equations),
        status: props.output.status,
        iteration: props.output.iteration
      }
    }).getCommandNodes(graph, {predictors: props.predictors, parameters: spatial.output.parameters})
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unfilter-coefficients`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'restore-spatial-intercepts',
      invocationCount: size * size,
      bindings: [
        {name: 'parameters', view: spatial.output.parameters, type: 'f32', access: 'read'},
        {name: 'responses', view: props.responses, type: 'f32', access: 'read'},
        {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
        {name: 'sourceCoefficients', view: coefficients, type: 'f32', access: 'read'},
        {name: 'sourceCovariance', view: covariance, type: 'f32', access: 'read'},
        {name: 'coefficients', view: props.output.coefficients, type: 'f32', access: 'read_write'},
        {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read_write'},
        {name: 'invalidWeights', view: invalidWeights, type: 'f32', access: 'read'}
      ],
      declarations: ROUNDED_DIFFERENCE_WGSL,
      body: `let first = index / ${size}u; let second = index % ${size}u;
let firstEquation = first / ${width}u; let secondEquation = second / ${width}u;
let firstColumn = first % ${width}u; let secondColumn = second % ${width}u;
// Each equation has A[0,:] = [1/(1-lambda), -xOrigin], identity slope rows.
var transformed = 0.0;
for (var left = 0u; left < ${width}u; left++) {
 var leftFactor = select(0.0, 1.0, left == firstColumn);
 if (firstColumn == 0u) { if (left == 0u) { leftFactor = 1.0 / (1.0 - parameters[parametersOffset + firstEquation]); } else { leftFactor = -predictors[predictorsOffset + firstEquation * ${rows * columns}u + left - 1u]; } }
 for (var right = 0u; right < ${width}u; right++) {
  var rightFactor = select(0.0, 1.0, right == secondColumn);
  if (secondColumn == 0u) { if (right == 0u) { rightFactor = 1.0 / (1.0 - parameters[parametersOffset + secondEquation]); } else { rightFactor = -predictors[predictorsOffset + secondEquation * ${rows * columns}u + right - 1u]; } }
  transformed += leftFactor * sourceCovariance[sourceCovarianceOffset + (firstEquation * ${width}u + left) * ${size}u + secondEquation * ${width}u + right] * rightFactor;
 }
}
covariance[covarianceOffset + index] = transformed;
if (index < ${size}u) {
 let equation = index / ${width}u;
 var value = sourceCoefficients[sourceCoefficientsOffset + index];
 if (index % ${width}u == 0u) {
  let centeredIntercept = value / (1.0 - parameters[parametersOffset + equation]);
  var originDifference = responses[responsesOffset + equation * ${rows}u];
  for (var column = 0u; column < ${columns}u; column++) {
    originDifference = getRoundedDifference(originDifference, predictors[predictorsOffset + equation * ${rows * columns}u + column] * sourceCoefficients[sourceCoefficientsOffset + index + column + 1u], bitcast<u32>(invalidWeights[invalidWeightsOffset]));
  }
  value = originDifference + centeredIntercept;
 }
 coefficients[coefficientsOffset + index] = value;
}`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-residuals`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant: 'structural-spatial-residuals',
      invocationCount: equations * rows,
      bindings: [
        {name: 'predictors', view: centeredPredictors, type: 'f32', access: 'read'},
        {name: 'responses', view: props.responses, type: 'f32', access: 'read'},
        {name: 'coefficients', view: coefficients, type: 'f32', access: 'read'},
        {name: 'parameters', view: spatial.output.parameters, type: 'f32', access: 'read'},
        {name: 'invalidWeights', view: invalidWeights, type: 'f32', access: 'read'},
        {name: 'residuals', view: props.output.residuals, type: 'f32', access: 'read_write'}
      ],
      declarations: ROUNDED_DIFFERENCE_WGSL,
      body: `let equation = index / ${rows}u; var residual = getRoundedDifference(responses[responsesOffset + index], responses[responsesOffset + equation * ${rows}u], bitcast<u32>(invalidWeights[invalidWeightsOffset])); residual -= coefficients[coefficientsOffset + equation * ${width}u] / (1.0 - parameters[parametersOffset + equation]); for (var column = 0u; column < ${columns}u; column++) { residual -= predictors[predictorsOffset + index * ${columns}u + column] * coefficients[coefficientsOffset + equation * ${width}u + column + 1u]; } residuals[residualsOffset + index] = residual;`
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
    candidateOverflow: props.weights.status?.candidateOverflow,
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
        : []),
      ...(props.weights.status?.candidateOverflow
        ? [
            {
              name: 'candidateOverflow',
              view: props.weights.status.candidateOverflow,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ],
    declarations: `${IS_FINITE_BITS_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  ${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { (*terms)[0] = 1.0; return; }' : ''}
  ${props.weights.status?.candidateOverflow ? 'if (candidateOverflow[candidateOverflowOffset] != 0u) { (*terms)[0] = 1.0; return; }' : ''}
  let begin = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
  if (row == 0u && begin != 0u) { (*terms)[0] = 1.0; return; }
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
