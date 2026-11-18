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
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {getDenseCholeskyWGSL} from '../../analysis-core/kernels/dense-solve-wgsl';
import {getExactLogDeterminantWGSL} from './spatial-exact-log-determinant';
import {getRegressionTileLayout} from './regression-row-tiles';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {getPanelWithinNodes, getPanelWithinScratchLengths} from './panel-within-nodes';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {PANEL_LAG_PROFILE_WGSL} from './panel-lag-profile-wgsl';
import type {GPUSpatialPanelLagMaximumLikelihoodProps} from './gpu-spatial-panel-lag-maximum-likelihood';

/** Each pivot index is a runtime word in an aligned slot; pivot count never creates new pipelines. */
const PANEL_PIVOT_SLOT_LENGTH = 64;

/**
 * Exact unit-effects panel ML. Row moments and direct residual squares are tiled; a supplied
 * entity spectrum replaces repeated dense determinants. A storage-backed pivoted inverse uses
 * five fixed kernels per pivot and keeps the full conditional information covariance exact.
 */
export function getPanelLagTiledNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialPanelLagMaximumLikelihoodProps
): GPUCommandNode<Parameters>[] {
  const entities = props.weights.offsets.length - 1;
  const rows = props.response.length;
  const columns = props.predictorCount;
  const informationSize = columns + 1;
  const {tileRowCount, tileCount} = getRegressionTileLayout(rows);
  const momentCount = columns * (columns + 2);
  const informationCount = informationSize * informationSize;
  const partialStride = Math.max(momentCount, informationCount, 4);
  const firstBetaStart = 8;
  const lagBetaStart = firstBetaStart + columns;
  const betaStart = lagBetaStart + columns;
  const gramStart = betaStart + columns;
  const covarianceStart = gramStart + columns * columns;
  const modelLength = covarianceStart + informationCount;
  const meanScratch = getPanelWithinScratchLengths(entities, props.periodCount, columns);
  validateGraphChunkBindingRanges(graph, id, {
    predictors: props.predictors,
    response: props.response,
    offsets: props.weights.offsets,
    neighbors: props.weights.neighbors,
    weights: props.weights.weights,
    overflow: props.weights.status?.overflow,
    candidateOverflow: props.weights.status?.candidateOverflow,
    eigenvalues: props.eigenvalues,
    ...props.output
  });
  validateRegressionAllocations(graph.device, id, {
    'within predictors': rows * columns,
    'within response/lag': rows,
    'panel mean partials': meanScratch.partialLength,
    'panel means': meanScratch.meanLength,
    'panel model': modelLength,
    'panel row partials': tileCount * partialStride,
    'entity matrix/inverse': entities * entities,
    'entity pivot factors': entities,
    'entity validation': entities,
    'pivot selection': 1,
    'pivot parameter slot': 1
  });
  validateCapacity(
    id,
    'pivot parameters byteLength',
    entities * PANEL_PIVOT_SLOT_LENGTH * 4,
    1,
    graph.device.limits.maxBufferSize
  );
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const predictors = view('within-predictors', rows * columns);
  const response = view('within-response', rows);
  const lag = view('within-lag', rows);
  const model = view('model', modelLength);
  const partial = view('row-partials', tileCount * partialStride);
  const validation = view('entity-validity', entities);
  const matrices = [view('matrix-a', entities * entities), view('matrix-b', entities * entities)];
  const inverses = [view('inverse-a', entities * entities), view('inverse-b', entities * entities)];
  const factors = view('pivot-factors', entities);
  const chosen = createTransientView(graph, `${id}-pivot-row`, 'uint32', 1);
  const pivotValues = new Uint32Array(entities * PANEL_PIVOT_SLOT_LENGTH);
  for (let pivot = 0; pivot < entities; pivot++)
    pivotValues[pivot * PANEL_PIVOT_SLOT_LENGTH] = pivot;
  const upload = createSpatialStatisticsParameterNodes(
    graph,
    `${id}-pivot-parameters`,
    pivotValues
  );
  const nodes = [
    ...upload.nodes,
    ...getPanelWithinNodes(graph, {
      id: `${id}-within`,
      entityCount: entities,
      periodCount: props.periodCount,
      predictorCount: columns,
      predictors: props.predictors,
      response: props.response,
      overflow: props.weights.status?.overflow,
      output: {predictors, response}
    })
  ];
  const binding = (
    name: string,
    value: GraphDataView,
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({
    name,
    view: value,
    type: value.format === 'uint32' ? 'u32' : 'f32',
    access
  });
  const modelRead = binding('model', model);
  const modelWrite = binding('model', model, 'read_write');
  const partialRead = binding('partial', partial);
  const partialWrite = binding('partial', partial, 'read_write');
  const spatialWeightBindings = [
    binding('offsets', props.weights.offsets),
    binding('neighbors', props.weights.neighbors),
    binding('weights', props.weights.weights)
  ];
  const declarations = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}
const FIRST_BETA: u32 = ${firstBetaStart}u;
const LAG_BETA: u32 = ${lagBetaStart}u;
const BETA: u32 = ${betaStart}u;
const GRAM: u32 = ${gramStart}u;
const COVARIANCE: u32 = ${covarianceStart}u;`;
  const merge = (slot: string, name = 'sum') =>
    getPairwiseSumWGSL({
      count: `${tileCount}u`,
      name,
      term: `partial[partialOffset + ${name}Item * ${partialStride}u + ${slot}]`
    });
  const rowDeclarations = `fn residualPair(row: u32) -> vec2<f32> {
  var first = response[responseOffset + row]; var second = lag[lagOffset + row];
  for (var column = 0u; column < ${columns}u; column++) {
    let value = predictors[predictorsOffset + row * ${columns}u + column];
    first -= value * model[modelOffset + FIRST_BETA + column];
    second -= value * model[modelOffset + LAG_BETA + column];
  }
  return vec2<f32>(first, second);
}`;
  const rowBindings = [
    binding('predictors', predictors),
    binding('response', response),
    binding('lag', lag),
    modelRead
  ];
  const kernel = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    extra = ''
  ) =>
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: name.replace(/-\d+/g, '-pivot'),
        invocationCount: count,
        workgroupSize: 64,
        bindings,
        declarations: `${declarations}\n${extra}`,
        body
      })
    );
  // contract-allow: single-invocation -- <=142 model words from the constructor's <=7 predictors.
  kernel(
    'initialize',
    1,
    [modelWrite, binding('chosen', chosen, 'read_write')],
    `for (var slot = 0u; slot < ${modelLength}u; slot++) { model[modelOffset + slot] = getNaN(); } model[modelOffset] = 0.0; model[modelOffset + 7u] = 0.0; chosen[chosenOffset] = 0u;`
  );
  kernel(
    'validate-entities',
    entities,
    [
      ...spatialWeightBindings,
      binding('eigenvalues', props.eigenvalues!),
      binding('validity', validation, 'read_write'),
      ...(props.weights.status?.overflow
        ? [binding('overflow', props.weights.status.overflow)]
        : []),
      ...(props.weights.status?.candidateOverflow
        ? [binding('candidateOverflow', props.weights.status.candidateOverflow)]
        : [])
    ],
    `var valid = ${props.weights.status?.overflow ? 'overflow[overflowOffset] == 0u' : 'true'} && ${props.weights.status?.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] == 0u' : 'true'};
let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u]; var total = 0.0;
if (end < begin || end > ${props.weights.neighbors.length}u) { valid = false; }
else { for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
if (neighbor >= ${entities}u || neighbor == index || !isFiniteBits(weight) || weight < 0.0 || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { valid = false; } total += weight; } }
let eigenvalue = eigenvalues[eigenvaluesOffset + index];
validity[validityOffset + index] = select(1.0, 0.0, valid && abs(total - 1.0) <= 1e-5 && isFiniteBits(eigenvalue) && abs(eigenvalue) <= 1.00001);`
  );
  kernel(
    'lag-response',
    rows,
    [...spatialWeightBindings, binding('response', response), binding('lag', lag, 'read_write')],
    `let entity = index % ${entities}u; let periodStart = index - entity; var value = 0.0;
let begin = offsets[offsetsOffset + entity]; let end = offsets[offsetsOffset + entity + 1u];
if (end < begin || end > ${props.weights.neighbors.length}u) { lag[lagOffset + index] = getNaN(); return; }
for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor >= ${entities}u) { value = getNaN(); break; } value += weights[weightsOffset + slot] * response[responseOffset + periodStart + neighbor]; }
lag[lagOffset + index] = value;`
  );
  kernel(
    'moment-tiles',
    tileCount,
    [...rowBindings.slice(0, 3), binding('validity', validation), partialWrite],
    `var terms: array<f32, ${momentCount}>; var invalid = 0.0;
for (var row = index * ${tileRowCount}u; row < min((index + 1u) * ${tileRowCount}u, ${rows}u); row++) {
let observed = response[responseOffset + row]; let lagged = lag[lagOffset + row];
if (!isFiniteBits(observed) || !isFiniteBits(lagged) || validity[validityOffset + row % ${entities}u] != 0.0) { invalid = 1.0; }
for (var column = 0u; column < ${columns}u; column++) { let value = predictors[predictorsOffset + row * ${columns}u + column]; if (!isFiniteBits(value)) { invalid = 1.0; }
for (var other = 0u; other < ${columns}u; other++) { terms[column * ${columns + 2}u + other] += value * predictors[predictorsOffset + row * ${columns}u + other]; }
terms[column * ${columns + 2}u + ${columns}u] += value * observed; terms[column * ${columns + 2}u + ${columns + 1}u] += value * lagged; } }
for (var slot = 0u; slot < ${momentCount}u; slot++) { partial[partialOffset + index * ${partialStride}u + slot] = select(terms[slot], getNaN(), invalid != 0.0); }`
  );
  // contract-allow: single-invocation -- reference fit: <=7 columns, <=4096 default-layout tile partials per term.
  kernel(
    'solve-moments',
    1,
    [partialRead, modelWrite],
    `var normal: array<f32, ${columns * columns}>; var first: array<f32, ${columns}>; var second: array<f32, ${columns}>;
for (var column = 0u; column < ${columns}u; column++) {
for (var other = 0u; other < ${columns}u; other++) { ${merge(`column * ${columns + 2}u + other`)} normal[column * ${columns}u + other] = sum; model[modelOffset + GRAM + column * ${columns}u + other] = sum; }
{ ${merge(`column * ${columns + 2}u + ${columns}u`)} first[column] = sum; }
{ ${merge(`column * ${columns + 2}u + ${columns + 1}u`)} second[column] = sum; }
}
var factorization = factorDenseCholesky_${columns}(&normal, ${columns}u, true, 1e-6);
if (factorization.status != 0u) { model[modelOffset] = select(2.0, 1.0, factorization.status == 1u); return; }
if (!solveDenseCholesky_${columns}(&normal, &factorization, &first) || !solveDenseCholesky_${columns}(&normal, &factorization, &second)) { model[modelOffset] = 2.0; return; }
for (var column = 0u; column < ${columns}u; column++) { model[modelOffset + FIRST_BETA + column] = first[column]; model[modelOffset + LAG_BETA + column] = second[column]; }`,
    getDenseCholeskyWGSL(columns)
  );
  kernel(
    'residual-pair-tiles',
    tileCount,
    [...rowBindings, partialWrite],
    `if (model[modelOffset] != 0.0) { return; } var cross = 0.0; var squares = 0.0;
for (var row = index * ${tileRowCount}u; row < min((index + 1u) * ${tileRowCount}u, ${rows}u); row++) { let pair = residualPair(row); cross += pair.x * pair.y; squares += pair.y * pair.y; }
partial[partialOffset + index * ${partialStride}u] = cross; partial[partialOffset + index * ${partialStride}u + 1u] = squares;`,
    rowDeclarations
  );
  // contract-allow: single-invocation -- reference fit: two merges of <=4096 default-layout tile partials.
  kernel(
    'residual-pair-merge',
    1,
    [partialRead, modelWrite],
    `if (model[modelOffset] != 0.0) { return; } { ${merge('0u')} model[modelOffset + 1u] = sum; } { ${merge('1u')} model[modelOffset + 2u] = sum; } model[modelOffset + 1u] = select(0.0, model[modelOffset + 1u] / model[modelOffset + 2u], model[modelOffset + 2u] > 0.0);`
  );
  kernel(
    'direct-tiles',
    tileCount,
    [...rowBindings, partialWrite],
    `if (model[modelOffset] != 0.0) { return; } var squares = 0.0; var cross = 0.0;
for (var row = index * ${tileRowCount}u; row < min((index + 1u) * ${tileRowCount}u, ${rows}u); row++) { let pair = residualPair(row); let remainder = getRoundedDifference(pair.x, model[modelOffset + 1u] * pair.y, bitcast<u32>(model[modelOffset + 7u])); squares += remainder * remainder; cross += remainder * pair.y; }
partial[partialOffset + index * ${partialStride}u] = squares; partial[partialOffset + index * ${partialStride}u + 1u] = cross;`,
    rowDeclarations
  );
  // contract-allow: single-invocation -- reference fit: <=4096 tiles, <=64 search steps over <=1024 eigenvalues.
  kernel(
    'profile-rho',
    1,
    [partialRead, modelWrite, binding('eigenvalues', props.eigenvalues!)],
    `if (model[modelOffset] != 0.0) { return; }
${merge('0u', 'squares')}
${merge('1u', 'cross')}
let lagSquares = model[modelOffset + 2u]; let correction = select(0.0, cross / lagSquares, lagSquares > 0.0);
let direct = vec4<f32>(squares - cross * correction, lagSquares, model[modelOffset + 1u], correction);
var lower = -0.999; var upper = 0.999; var left = lower + 0.38196601125 * (upper - lower); var right = lower + 0.61803398875 * (upper - lower); var leftScore = profileObjective(left, direct); var rightScore = profileObjective(right, direct);
for (var step = 0u; step < ${props.maximumSearchIterations ?? 32}u; step++) { if (leftScore <= rightScore) { upper = right; right = left; rightScore = leftScore; left = lower + 0.38196601125 * (upper - lower); leftScore = profileObjective(left, direct); } else { lower = left; left = right; leftScore = rightScore; right = lower + 0.61803398875 * (upper - lower); rightScore = profileObjective(right, direct); } }
let rho = select(right, left, leftScore <= rightScore); let width = upper - lower; let offset = getRoundedDifference(rho, direct.z, bitcast<u32>(model[modelOffset + 7u])) - direct.w;
let variance = (direct.x + direct.y * offset * offset) / ${rows}.0;
model[modelOffset] = select(0.0, 3.0, width > ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)});
if (!(variance > 0.0) || !isFiniteBits(variance)) { model[modelOffset] = 2.0; }
model[modelOffset + 1u] = rho; model[modelOffset + 2u] = variance;
model[modelOffset + 3u] = -objective(rho, direct) - ${getWGSLFloatLiteral(rows / 2)} * (1.0 + 1.8378770664093453 + log(${rows}.0));
model[modelOffset + 4u] = width; model[modelOffset + 5u] = select(0.0, 1.0, abs(rho) > 0.9989);
model[modelOffset + 6u] = getTraceTwo(rho) * ${props.periodCount}.0;
for (var column = 0u; column < ${columns}u; column++) { model[modelOffset + BETA + column] = model[modelOffset + FIRST_BETA + column] - rho * model[modelOffset + LAG_BETA + column]; }`,
    `${getExactLogDeterminantWGSL('eigenvalues', entities)}\n${PANEL_LAG_PROFILE_WGSL}
fn evaluatePanelObjective(rho: f32, direct: vec4<f32>, centered: bool) -> f32 { let runtimeZero = bitcast<u32>(model[modelOffset + 7u]); let offset = getRoundedDifference(rho, direct.z, runtimeZero) - direct.w; let variance = (direct.x + direct.y * offset * offset) / ${rows}.0; if (!(variance > 0.0)) { return 3.402823e38; } let logarithm = select(log(variance), getPanelLagProfileLogVariance(rho, direct, ${rows}.0, runtimeZero), centered); return ${getWGSLFloatLiteral(rows / 2)} * logarithm - ${props.periodCount}.0 * getSpatialLogDeterminant(rho); }
fn objective(rho: f32, direct: vec4<f32>) -> f32 { return evaluatePanelObjective(rho, direct, false); }
fn profileObjective(rho: f32, direct: vec4<f32>) -> f32 { return evaluatePanelObjective(rho, direct, true); }`
  );
  kernel(
    'matrix-initialize',
    entities * entities,
    [
      ...spatialWeightBindings,
      modelRead,
      binding('matrix', matrices[0], 'read_write'),
      binding('inverse', inverses[0], 'read_write')
    ],
    `let row = index / ${entities}u; let column = index % ${entities}u; var weight = 0.0;
for (var slot = offsets[offsetsOffset + row]; slot < min(offsets[offsetsOffset + row + 1u], ${props.weights.neighbors.length}u); slot++) { if (neighbors[neighborsOffset + slot] == column) { weight = weights[weightsOffset + slot]; } }
matrix[matrixOffset + index] = select(0.0, 1.0, row == column) - model[modelOffset + 1u] * weight; inverse[inverseOffset + index] = select(0.0, 1.0, row == column);`
  );
  for (let pivot = 0; pivot < entities; pivot++) {
    const currentMatrix = matrices[pivot % 2];
    const nextMatrix = matrices[(pivot + 1) % 2];
    const currentInverse = inverses[pivot % 2];
    const nextInverse = inverses[(pivot + 1) % 2];
    const pivotParameters = graph.createDataView(upload.parameters.buffer, {
      format: 'uint32',
      byteOffset: pivot * PANEL_PIVOT_SLOT_LENGTH * 4,
      length: 1
    });
    const pivotBinding = binding('parameters', pivotParameters);
    // This is an entity-axis scan, not coefficient work: tiled construction caps entities at
    // GPU_PANEL_LAG_TILED_MAXIMUM_ENTITY_COUNT (1024). One serial selection per pivot gives
    // O(entities^2) total selection work; swap/normalize/eliminate remain parallel kernels.
    // contract-allow: single-invocation -- reference fit: <=1024 entity candidates per pivot, enforced by the constructor.
    kernel(
      `pivot-select-${pivot}`,
      1,
      [
        pivotBinding,
        binding('matrix', currentMatrix),
        binding('chosen', chosen, 'read_write'),
        modelWrite
      ],
      `if (model[modelOffset] != 0.0) { return; } let pivot = parameters[parametersOffset]; var selected = pivot; for (var row = pivot + 1u; row < ${entities}u; row++) { if (abs(matrix[matrixOffset + row * ${entities}u + pivot]) > abs(matrix[matrixOffset + selected * ${entities}u + pivot])) { selected = row; } } let diagonal = matrix[matrixOffset + selected * ${entities}u + pivot]; if (!(abs(diagonal) > 1e-10) || !isFiniteBits(diagonal)) { model[modelOffset] = 2.0; } chosen[chosenOffset] = selected;`
    );
    kernel(
      `pivot-swap-${pivot}`,
      entities * entities,
      [
        pivotBinding,
        binding('chosen', chosen),
        binding('sourceMatrix', currentMatrix),
        binding('sourceInverse', currentInverse),
        binding('matrix', nextMatrix, 'read_write'),
        binding('inverse', nextInverse, 'read_write')
      ],
      `let pivot = parameters[parametersOffset]; let selected = chosen[chosenOffset]; let row = index / ${entities}u; let column = index % ${entities}u; var sourceRow = row; if (row == pivot) { sourceRow = selected; } else if (row == selected) { sourceRow = pivot; } matrix[matrixOffset + index] = sourceMatrix[sourceMatrixOffset + sourceRow * ${entities}u + column]; inverse[inverseOffset + index] = sourceInverse[sourceInverseOffset + sourceRow * ${entities}u + column];`
    );
    kernel(
      `pivot-factors-${pivot}`,
      entities,
      [pivotBinding, binding('matrix', nextMatrix), binding('factors', factors, 'read_write')],
      `factors[factorsOffset + index] = matrix[matrixOffset + index * ${entities}u + parameters[parametersOffset]];`
    );
    kernel(
      `pivot-normalize-${pivot}`,
      entities,
      [
        pivotBinding,
        binding('factors', factors),
        modelRead,
        binding('matrix', nextMatrix, 'read_write'),
        binding('inverse', nextInverse, 'read_write')
      ],
      `if (model[modelOffset] != 0.0) { return; } let pivot = parameters[parametersOffset]; let diagonal = factors[factorsOffset + pivot]; matrix[matrixOffset + pivot * ${entities}u + index] /= diagonal; inverse[inverseOffset + pivot * ${entities}u + index] /= diagonal;`
    );
    kernel(
      `pivot-eliminate-${pivot}`,
      entities * entities,
      [
        pivotBinding,
        binding('factors', factors),
        modelRead,
        binding('matrix', nextMatrix, 'read_write'),
        binding('inverse', nextInverse, 'read_write')
      ],
      `if (model[modelOffset] != 0.0) { return; } let pivot = parameters[parametersOffset]; let row = index / ${entities}u; let column = index % ${entities}u; if (row == pivot) { return; } let factor = factors[factorsOffset + row]; matrix[matrixOffset + index] -= factor * matrix[matrixOffset + pivot * ${entities}u + column]; inverse[inverseOffset + index] -= factor * inverse[inverseOffset + pivot * ${entities}u + column];`
    );
  }
  const inverse = inverses[entities % 2];
  const product = matrices[0];
  kernel(
    'spatial-product',
    entities * entities,
    [
      ...spatialWeightBindings,
      binding('inverse', inverse),
      binding('product', product, 'read_write')
    ],
    `let row = index / ${entities}u; let column = index % ${entities}u; var total = 0.0; for (var slot = offsets[offsetsOffset + row]; slot < min(offsets[offsetsOffset + row + 1u], ${props.weights.neighbors.length}u); slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor < ${entities}u) { total += weights[weightsOffset + slot] * inverse[inverseOffset + neighbor * ${entities}u + column]; } } product[productOffset + index] = total;`
  );
  kernel(
    'information-tiles',
    tileCount,
    [binding('predictors', predictors), binding('product', product), modelRead, partialWrite],
    `var terms: array<f32, ${informationCount}>;
for (var row = index * ${tileRowCount}u; row < min((index + 1u) * ${tileRowCount}u, ${rows}u); row++) { let entity = row % ${entities}u; let start = row - entity; var lagPrediction = 0.0;
for (var neighbor = 0u; neighbor < ${entities}u; neighbor++) { var prediction = 0.0; for (var column = 0u; column < ${columns}u; column++) { prediction += predictors[predictorsOffset + (start + neighbor) * ${columns}u + column] * model[modelOffset + BETA + column]; } lagPrediction += product[productOffset + entity * ${entities}u + neighbor] * prediction; }
for (var column = 0u; column < ${columns}u; column++) { terms[${columns * informationSize}u + column] += predictors[predictorsOffset + row * ${columns}u + column] * lagPrediction / model[modelOffset + 2u]; }
var frobenius = 0.0; for (var neighbor = 0u; neighbor < ${entities}u; neighbor++) { let value = product[productOffset + entity * ${entities}u + neighbor]; frobenius += value * value; }
terms[${informationCount - 1}u] += lagPrediction * lagPrediction / model[modelOffset + 2u] + frobenius;
}
for (var slot = 0u; slot < ${informationCount}u; slot++) { partial[partialOffset + index * ${partialStride}u + slot] = terms[slot]; }`
  );
  // contract-allow: single-invocation -- reference fit: <=8 information columns, <=4096 default-layout tile partials per term.
  kernel(
    'information-solve',
    1,
    [partialRead, modelWrite],
    `if (model[modelOffset] != 0.0) { return; } var information: array<f32, ${informationCount}>;
for (var row = 0u; row < ${columns}u; row++) { for (var column = 0u; column <= row; column++) { information[row * ${informationSize}u + column] = model[modelOffset + GRAM + row * ${columns}u + column] / model[modelOffset + 2u]; } }
for (var column = 0u; column < ${columns}u; column++) { ${merge(`${columns * informationSize}u + column`)} information[${columns * informationSize}u + column] = sum; }
{ ${merge(`${informationCount - 1}u`)} information[${informationCount - 1}u] = sum + model[modelOffset + 6u]; }
var factorization = factorDenseCholesky_${informationSize}(&information, ${informationSize}u, true, 1e-6); var covariance: array<f32, ${informationCount}>;
if (factorization.status != 0u || !invertDenseCholesky_${informationSize}(&information, &factorization, &covariance)) { model[modelOffset] = 2.0; return; }
for (var slot = 0u; slot < ${informationCount}u; slot++) { model[modelOffset + COVARIANCE + slot] = covariance[slot]; }`,
    getDenseCholeskyWGSL(informationSize)
  );
  // contract-allow: single-invocation -- <=10 coefficient slots, <=81 covariance entries, six summary words.
  kernel(
    'publish',
    1,
    [
      modelRead,
      binding('coefficients', props.output.coefficients, 'read_write'),
      binding('covariance', props.output.covariance, 'read_write'),
      binding('summary', props.output.summary, 'read_write'),
      binding('status', props.output.status, 'read_write')
    ],
    `let state = u32(model[modelOffset]); status[statusOffset] = state;
for (var slot = 0u; slot < ${columns + 3}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${(columns + 2) ** 2}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
for (var slot = 0u; slot < 6u; slot++) { summary[summaryOffset + slot] = getNaN(); }
if (state != 0u && state != 3u) { return; }
for (var column = 0u; column < ${columns}u; column++) { coefficients[coefficientsOffset + column + 1u] = model[modelOffset + BETA + column]; }
coefficients[coefficientsOffset + ${columns + 1}u] = model[modelOffset + 1u]; coefficients[coefficientsOffset + ${columns + 2}u] = 0.0;
summary[summaryOffset] = model[modelOffset + 2u]; summary[summaryOffset + 1u] = model[modelOffset + 3u]; summary[summaryOffset + 2u] = 1.0; summary[summaryOffset + 3u] = 0.0; summary[summaryOffset + 4u] = model[modelOffset + 4u]; summary[summaryOffset + 5u] = model[modelOffset + 5u];
if (state == 0u) { for (var row = 0u; row < ${informationSize}u; row++) { for (var column = 0u; column < ${informationSize}u; column++) { covariance[covarianceOffset + (row + 1u) * ${columns + 2}u + column + 1u] = model[modelOffset + COVARIANCE + row * ${informationSize}u + column]; } } }`
  );
  kernel(
    'predictions',
    rows,
    [
      ...rowBindings,
      binding('fitted', props.output.fitted, 'read_write'),
      binding('residuals', props.output.residuals, 'read_write')
    ],
    `var fittedValue = getNaN(); var residual = getNaN(); if (model[modelOffset] == 0.0 || model[modelOffset] == 3.0) { fittedValue = model[modelOffset + 1u] * lag[lagOffset + index]; for (var column = 0u; column < ${columns}u; column++) { fittedValue += predictors[predictorsOffset + index * ${columns}u + column] * model[modelOffset + BETA + column]; } residual = response[responseOffset + index] - fittedValue; } fitted[fittedOffset + index] = fittedValue; residuals[residualsOffset + index] = residual;`
  );
  return nodes;
}
