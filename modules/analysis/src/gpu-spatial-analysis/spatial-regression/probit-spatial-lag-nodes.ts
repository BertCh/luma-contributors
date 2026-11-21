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
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getDenseCholeskyWGSL} from '../../analysis-core/kernels/dense-solve-wgsl';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE} from './ordinary-least-squares-parameters';

/** Status of the optional one-step latent spatial-lag probit estimate. */
export const GPU_PROBIT_SPATIAL_LAG_STATUS = {
  OK: 0,
  MODEL_NOT_CONVERGED: 1,
  SINGULAR_FIRST_STAGE: 2,
  SINGULAR_SECOND_STAGE: 3,
  RHO_OUT_OF_RANGE: 4,
  INVALID_WEIGHTS: 5
} as const;

/** Fixed optional graph: seed, two sparse products, four tile reductions and four dense/row nodes. @internal */
export const PROBIT_SPATIAL_LAG_COMMAND_NODE_COUNT = 15;

/** Probit accepts a zero-capacity all-island CSR and requires exact scalar statuses. @internal */
export function validateProbitSpatialWeights(
  id: string,
  weights: GPUSpatialWeights,
  name = 'weights'
): number {
  let rows: number;
  if (weights.neighbors.length) rows = validateGPUSpatialWeights(id, weights, name);
  else {
    for (const [field, view, format] of [
      ['offsets', weights.offsets, 'uint32'],
      ['neighbors', weights.neighbors, 'uint32'],
      ['weights', weights.weights, 'float32'],
      ['distances', weights.distances, 'float32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} ${name}.${field}`);
        if (field !== 'offsets')
          validateRequirement(
            id,
            view.length === 0,
            `${name}.${field} length must equal zero neighbor capacity`
          );
      }
    }
    validateCapacity(id, `${name}.offsets length`, weights.offsets.length, 2);
    rows = weights.offsets.length - 1;
    if (weights.information)
      validateRequirement(
        id,
        weights.information.rowCount === rows,
        `${name}.information.rowCount must equal the row count`
      );
  }
  for (const [field, view] of Object.entries(weights.status ?? {})) {
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} ${name}.status.${field}`);
      validateRequirement(
        id,
        view.length === 1,
        `${name}.status.${field} must hold exactly one uint32 value`
      );
    }
  }
  return rows;
}

/** Completeness and convergence assert success with a positive word. @internal */
export function getProbitStatusFailureComparison(name: string): '==' | '!=' {
  return name === 'complete' || name === 'converged' ? '==' : '!=';
}

/** Flags aggregated separately from the written/required count loss comparison. @internal */
export function getProbitWeightsFailureStatusEntries(weights: GPUSpatialWeights) {
  return Object.entries(weights.status ?? {}).filter(
    ([name, view]) => view && !['count', 'requiredCount', 'iterationCount'].includes(name)
  ) as [string, GraphDataView<'uint32'>][];
}

/** Number of bounded status aggregation nodes, including the counter loss proof. @internal */
export function getProbitWeightsStatusNodeCount(weights: GPUSpatialWeights): number {
  return (
    Math.ceil(getProbitWeightsFailureStatusEntries(weights).length / 7) +
    (weights.status?.requiredCount || weights.status?.count ? 1 : 0)
  );
}

/** Packs at most seven loss flags into one binding, preserving the default eight-binding limit. @internal */
export function getProbitWeightsStatusSummary<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GPUSpatialWeights
): {nodes: GPUCommandNode<Parameters>[]; entries: [string, GraphDataView<'uint32'>][]} {
  const entries = getProbitWeightsFailureStatusEntries(weights);
  const nodes: GPUCommandNode<Parameters>[] = [];
  const summaries: [string, GraphDataView<'uint32'>][] = [];
  for (let first = 0; first < entries.length; first += 7) {
    const group = entries.slice(first, first + 7);
    const summary = createTransientView(graph, `${id}-weights-incomplete-${first}`, 'uint32', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weights-status-${first}`,
        operation: 'GPUProbitRegression',
        variant: 'weights-status',
        invocationCount: 1,
        bindings: [
          ...group.map(([, view], index) => ({
            name: `flag${index}`,
            view,
            type: 'u32' as const,
            access: 'read' as const
          })),
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        body: `summary[summaryOffset] = select(0u, 1u, ${group.map(([name], index) => `flag${index}[flag${index}Offset] ${getProbitStatusFailureComparison(name)} 0u`).join(' || ')});`
      })
    );
    summaries.push(['incomplete', summary]);
  }
  if (weights.status?.requiredCount || weights.status?.count) {
    const summary = createTransientView(graph, `${id}-weights-counter-loss`, 'uint32', 1);
    const required = weights.status.requiredCount;
    const count = weights.status.count;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weights-counts`,
        operation: 'GPUProbitRegression',
        variant: 'weights-counts',
        invocationCount: 1,
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          ...(required
            ? [{name: 'required', view: required, type: 'u32' as const, access: 'read' as const}]
            : []),
          ...(count
            ? [{name: 'count', view: count, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        body: `let first = offsets[offsetsOffset];
let last = offsets[offsetsOffset + ${weights.offsets.length - 1}u];
let accessible = min(max(first, last) - first, ${weights.neighbors.length}u);
let written = ${count ? 'count[countOffset]' : 'accessible'};
summary[summaryOffset] = select(0u, 1u, written > ${weights.neighbors.length}u ${required ? '|| required[requiredOffset] > min(written, accessible)' : ''});`
      })
    );
    summaries.push(['incomplete', summary]);
  }
  return {nodes, entries: summaries};
}

/** Preflight optional row tables and tile partials before building any graph nodes. @internal */
export function validateProbitSpatialLagStorage<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rows: number,
  columns: number
): void {
  const instrumentCount = 3 * (columns + 1);
  const coefficientCount = columns + 2;
  const momentCount = instrumentCount * (instrumentCount + coefficientCount) + 1;
  for (const [name, length] of [
    ['spatialLag.seed', rows * (columns + 5)],
    ['spatialLag.sparseProduct', rows * (columns + 4)],
    ['spatialLag.design', rows * (instrumentCount + coefficientCount + 1)],
    ['spatialLag.projectedGradients', rows * coefficientCount],
    ['spatialLag.firstStagePartials', getRegressionTileLayout(rows).tileCount * momentCount]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      length * 4,
      'reduce the row count or use CPUProbitRegressionOperation'
    );
  }
}

/** Internal wiring from the centered ordinary fit; all row work is tiled or sparse. @internal */
export function getProbitSpatialLagNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    columns: number;
    predictors: GraphDataView<'float32'>;
    modelStatus: GraphDataView<'uint32'>;
    weights: GPUSpatialWeights;
    output: {
      coefficients: GraphDataView<'float32'>;
      standardErrors: GraphDataView<'float32'>;
      status: GraphDataView<'uint32'>;
    };
    rowBindings: WGSLKernelBinding[];
    rowDeclarations: string;
    state: GraphDataView<'float32'>;
    meansSlot: number;
    scalesSlot: number;
  }
): GPUCommandNode<Parameters>[] {
  const {rows, columns, weights, output} = props;
  const id = `${props.id}-spatial-lag`;
  const size = columns + 2;
  const ordinarySize = columns + 1;
  const instrumentCount = 3 * ordinarySize;
  const seedStride = ordinarySize + 4;
  const sparseStride = ordinarySize + 3;
  const designStride = instrumentCount + size + 1;
  const momentCount = instrumentCount * (instrumentCount + size);
  const secondMomentCount = size * size + size;
  const tileLayout = getRegressionTileLayout(rows);
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'f32' | 'u32' = 'f32'
  ): WGSLKernelBinding => ({name, view, access, type});
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const seed = transient('seed', rows * seedStride);
  const sparseProduct = transient('sparse-product', rows * sparseStride);
  const design = transient('design', rows * designStride);
  const moments = transient('first-moments', momentCount + 1);
  const instrumentMeans = transient('instrument-means', instrumentCount + 1);
  const projection = transient('projection', instrumentCount * size + 1);
  const projected = transient('projected', rows * size);
  const secondMoments = transient('second-moments', secondMomentCount + 1);
  // Inverse bread, estimate, then status. A separate buffer avoids read/write alias bindings.
  const solution = transient('solution', size * size + size + 1);
  const meat = transient('meat', size * size);
  const declarations = `${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}
const ROWS: u32 = ${rows}u;
const C: u32 = ${columns}u;
const P: u32 = ${ordinarySize}u;
const Q: u32 = ${size}u;
const M: u32 = ${instrumentCount}u;
const SEED_STRIDE: u32 = ${seedStride}u;
const SPARSE_STRIDE: u32 = ${sparseStride}u;
const DESIGN_STRIDE: u32 = ${designStride}u;`;
  const sparseBindings = [
    binding('offsets', weights.offsets, 'read', 'u32'),
    binding('neighbors', weights.neighbors, 'read', 'u32'),
    binding('weights', weights.weights),
    binding('seed', seed)
  ];
  const node = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    extraDeclarations = declarations
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUProbitRegression',
      variant: `spatial-lag-${name}`,
      invocationCount,
      bindings,
      declarations: extraDeclarations,
      body
    });
  const reduce = (
    name: string,
    sumCount: number,
    maximumCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    total: GraphDataView<'float32'>
  ) =>
    getRowTileSumNodes<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUProbitRegression',
      variant: `spatial-lag-${name}`,
      rowCount: rows,
      layout: tileLayout,
      sumCount,
      maximumCount,
      bindings,
      declarations: `${declarations}\nfn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) { ${body} }`,
      partial: transient(`${name}-partial`, tileLayout.tileCount * (sumCount + maximumCount)),
      output: total
    });
  const statusSummary = getProbitWeightsStatusSummary(graph, id, weights);
  const failureStatuses = statusSummary.entries;
  const pivotRatio = GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE ** 2;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...statusSummary.nodes,
    node(
      'seed',
      rows,
      [
        ...props.rowBindings,
        binding('modelStatus', props.modelStatus, 'read', 'u32'),
        binding('seed', seed, 'read_write')
      ],
      `let start = seedOffset + index * ${seedStride}u;
for (var slot = 0u; slot < ${seedStride}u; slot++) { seed[start + slot] = getNaN(); }
if (modelStatus[modelStatusOffset] != 0u || !included(index)) { return; }
let design = getDesign(index); var linear = 0.0;
for (var column = 0u; column < P; column++) { linear += design[column] * stateAt(S_ESTIMATE + column); seed[start + 3u + column] = design[column]; }
let fitted = probability(linear); let derivative = density(linear); let scale = inverseSqrt(fitted * (1.0 - fitted));
seed[start] = linear; seed[start + 1u] = derivative * scale;
seed[start + 2u] = (response[responseOffset + index] - fitted + derivative * linear) * scale;
var errorScale = 1.0; for (var column = 0u; column < P; column++) { errorScale += abs(design[column]) * (1.0 + abs(stateAt(S_ESTIMATE + column))); }
seed[start + 3u + P] = ${(props.columns + 8) * 2 ** -23} * errorScale;`,
      props.rowDeclarations
    ),
    node(
      'sparse-product',
      rows,
      [...sparseBindings, binding('sparseProduct', sparseProduct, 'read_write')],
      `let start = sparseProductOffset + index * SPARSE_STRIDE;
for (var column = 0u; column < SPARSE_STRIDE; column++) { sparseProduct[start + column] = 0.0; }
let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
if (offsets[offsetsOffset] != 0u || end < begin || end > ${weights.neighbors.length}u) { sparseProduct[start] = 1.0; return; }
for (var slot = begin; slot < end; slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
 if (neighbor >= ROWS || neighbor == index || !isFiniteValue(weight) || weight < 0.0 || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { sparseProduct[start] = 1.0; continue; }
 if (!isFiniteValue(seed[seedOffset + index * SEED_STRIDE]) || !isFiniteValue(seed[seedOffset + neighbor * SEED_STRIDE])) { continue; }
 sparseProduct[start + 1u] += weight * seed[seedOffset + neighbor * SEED_STRIDE];
 for (var column = 0u; column < P; column++) { sparseProduct[start + 2u + column] += weight * seed[seedOffset + neighbor * SEED_STRIDE + 3u + column]; }
 sparseProduct[start + 2u + P] += weight * seed[seedOffset + neighbor * SEED_STRIDE + 3u + P];
}`
    ),
    node(
      'design',
      rows,
      [
        ...sparseBindings,
        binding('sparseProduct', sparseProduct),
        binding('design', design, 'read_write')
      ],
      `let start = designOffset + index * DESIGN_STRIDE;
for (var slot = 0u; slot < DESIGN_STRIDE; slot++) { design[start + slot] = 0.0; }
design[start + M + Q] = getNaN();
if (!isFiniteValue(seed[seedOffset + index * SEED_STRIDE])) { return; }
for (var column = 0u; column < P; column++) {
 design[start + column] = seed[seedOffset + index * SEED_STRIDE + 3u + column];
 design[start + P + column] = sparseProduct[sparseProductOffset + index * SPARSE_STRIDE + 2u + column];
}
let begin = min(offsets[offsetsOffset + index], ${weights.neighbors.length}u); let end = min(offsets[offsetsOffset + index + 1u], ${weights.neighbors.length}u);
for (var slot = begin; slot < end; slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; if (neighbor >= ROWS || !isFiniteValue(seed[seedOffset + neighbor * SEED_STRIDE])) { continue; }
 for (var column = 0u; column < P; column++) { design[start + 2u * P + column] += weights[weightsOffset + slot] * sparseProduct[sparseProductOffset + neighbor * SPARSE_STRIDE + 2u + column]; }
}
let derivative = seed[seedOffset + index * SEED_STRIDE + 1u];
for (var column = 0u; column < P; column++) { design[start + M + column] = derivative * seed[seedOffset + index * SEED_STRIDE + 3u + column]; }
design[start + M + P] = derivative * sparseProduct[sparseProductOffset + index * SPARSE_STRIDE + 1u];
design[start + M + Q] = seed[seedOffset + index * SEED_STRIDE + 2u];`
    ),
    ...reduce(
      'instrument-means',
      instrumentCount + 1,
      0,
      [binding('design', design)],
      `let start = designOffset + row * DESIGN_STRIDE; if (!isFiniteValue(design[start + M + Q])) { return; }
for (var column = 1u; column < M; column++) { (*terms)[column] += design[start + column]; } (*terms)[M] += 1.0;`,
      instrumentMeans
    ),
    ...reduce(
      'first-moments',
      momentCount,
      1,
      [
        binding('design', design),
        binding('sparseProduct', sparseProduct),
        binding('instrumentMeans', instrumentMeans)
      ],
      `(*terms)[SUM_COUNT] = max((*terms)[SUM_COUNT], sparseProduct[sparseProductOffset + row * SPARSE_STRIDE]);
let start = designOffset + row * DESIGN_STRIDE; if (!isFiniteValue(design[start + M + Q])) { return; }
for (var first = 0u; first < M; first++) {
 let firstValue = design[start + first] - instrumentMeans[instrumentMeansOffset + first] / max(1.0, instrumentMeans[instrumentMeansOffset + M]);
 for (var second = 0u; second < M; second++) { let secondValue = design[start + second] - instrumentMeans[instrumentMeansOffset + second] / max(1.0, instrumentMeans[instrumentMeansOffset + M]); (*terms)[first * M + second] += firstValue * secondValue; }
 for (var column = 0u; column < Q; column++) { (*terms)[M * M + first * Q + column] += firstValue * design[start + M + column]; }
}`,
      moments
    ),
    // contract-allow: single-invocation -- instrument-basis selection and dense solves over at most 36 instruments, enforced by the contributor's 11-predictor cap.
    node(
      'project-solve',
      1,
      [
        binding('moments', moments),
        binding('modelStatus', props.modelStatus, 'read', 'u32'),
        ...failureStatuses.map(([, view], index) =>
          binding(`weightStatus${index}`, view, 'read', 'u32')
        ),
        binding('projection', projection, 'read_write')
      ],
      `for (var entry = 0u; entry < M * Q; entry++) { projection[projectionOffset + entry] = 0.0; }
projection[projectionOffset + M * Q] = 1.0;
if (modelStatus[modelStatusOffset] != 0u) { return; }
if (moments[momentsOffset + ${momentCount}u] != 0.0 ${failureStatuses.map(([name], index) => `|| weightStatus${index}[weightStatus${index}Offset] ${getProbitStatusFailureComparison(name)} 0u`).join(' ')}) { projection[projectionOffset + M * Q] = 5.0; return; }
projection[projectionOffset + M * Q] = 2.0;
// Select an independent instrument basis in declared order. Each trial uses the shared dense
// factorization; a redundant column is dropped at the same 1e-3 factor-pivot floor as probit.
var selected: array<u32, ${instrumentCount}>; var rank = 0u;
var factor: array<f32, ${instrumentCount * instrumentCount}>;
var factorization: DenseCholesky_${instrumentCount};
for (var candidate = 0u; candidate < M; candidate++) {
 selected[rank] = candidate;
 for (var row = 0u; row <= rank; row++) { for (var column = 0u; column <= row; column++) { factor[row * M + column] = moments[momentsOffset + selected[row] * M + selected[column]]; } }
 let trial = factorDenseCholesky_${instrumentCount}(&factor, rank + 1u, true, ${pivotRatio});
 if (trial.status == 0u) { rank++; }
}
if (rank < Q) { return; }
for (var row = 0u; row < rank; row++) { for (var column = 0u; column <= row; column++) { factor[row * M + column] = moments[momentsOffset + selected[row] * M + selected[column]]; } }
factorization = factorDenseCholesky_${instrumentCount}(&factor, rank, true, ${pivotRatio});
if (factorization.status != 0u) { return; }
for (var column = 0u; column < Q; column++) {
 var right: array<f32, ${instrumentCount}>;
 for (var row = 0u; row < rank; row++) { right[row] = moments[momentsOffset + M * M + selected[row] * Q + column]; }
 if (!solveDenseCholesky_${instrumentCount}(&factor, &factorization, &right)) { return; }
 for (var row = 0u; row < rank; row++) { projection[projectionOffset + selected[row] * Q + column] = right[row]; }
}
projection[projectionOffset + M * Q] = 0.0;`,
      `${declarations}\n${getDenseCholeskyWGSL(instrumentCount)}`
    ),
    node(
      'project',
      rows,
      [
        binding('design', design),
        binding('projection', projection),
        binding('instrumentMeans', instrumentMeans),
        binding('projected', projected, 'read_write')
      ],
      `for (var column = 0u; column < Q; column++) {
 var value = 0.0;
 if (projection[projectionOffset + M * Q] == 0.0 && isFiniteValue(design[designOffset + index * DESIGN_STRIDE + M + Q])) {
  for (var instrument = 0u; instrument < M; instrument++) { let instrumentValue = design[designOffset + index * DESIGN_STRIDE + instrument] - instrumentMeans[instrumentMeansOffset + instrument] / max(1.0, instrumentMeans[instrumentMeansOffset + M]); value += instrumentValue * projection[projectionOffset + instrument * Q + column]; }
 }
 projected[projectedOffset + index * Q + column] = value;
}`
    ),
    ...reduce(
      'second-moments',
      secondMomentCount,
      1,
      [
        binding('design', design),
        binding('projected', projected),
        binding('sparseProduct', sparseProduct)
      ],
      `let rowTarget = design[designOffset + row * DESIGN_STRIDE + M + Q]; if (!isFiniteValue(rowTarget)) { return; }
if (abs(sparseProduct[sparseProductOffset + row * SPARSE_STRIDE + 1u]) > sparseProduct[sparseProductOffset + row * SPARSE_STRIDE + 2u + P]) { (*terms)[SUM_COUNT] = 1.0; }
for (var first = 0u; first < Q; first++) {
 let value = projected[projectedOffset + row * Q + first]; (*terms)[Q * Q + first] += value * rowTarget;
 for (var second = 0u; second < Q; second++) { (*terms)[first * Q + second] += value * projected[projectedOffset + row * Q + second]; }
}`,
      secondMoments
    ),
    // contract-allow: single-invocation -- shared dense solve of at most 13 coefficients, enforced by the contributor's 11-predictor cap.
    node(
      'second-solve',
      1,
      [
        binding('secondMoments', secondMoments),
        binding('projection', projection),
        binding('solution', solution, 'read_write')
      ],
      `let code = projection[projectionOffset + M * Q]; solution[solutionOffset + Q * Q + Q] = code;
if (code != 0.0) { return; }
solution[solutionOffset + Q * Q + Q] = 3.0;
if (secondMoments[secondMomentsOffset + ${secondMomentCount}u] == 0.0) { return; }
var factor: array<f32, ${size * size}>; var right: array<f32, ${size}>;
for (var row = 0u; row < Q; row++) { right[row] = secondMoments[secondMomentsOffset + Q * Q + row]; for (var column = 0u; column <= row; column++) { factor[row * Q + column] = secondMoments[secondMomentsOffset + row * Q + column]; } }
var factorization = factorDenseCholesky_${size}(&factor, Q, true, ${pivotRatio});
if (factorization.status != 0u || !solveDenseCholesky_${size}(&factor, &factorization, &right)) { return; }
var inverse: array<f32, ${size * size}>; if (!invertDenseCholesky_${size}(&factor, &factorization, &inverse)) { return; }
for (var entry = 0u; entry < Q * Q; entry++) { solution[solutionOffset + entry] = inverse[entry]; }
for (var column = 0u; column < Q; column++) { solution[solutionOffset + Q * Q + column] = right[column]; }
solution[solutionOffset + Q * Q + Q] = select(0.0, 4.0, abs(right[Q - 1u]) >= 1.0);`,
      `${declarations}\n${getDenseCholeskyWGSL(size)}`
    ),
    ...reduce(
      'meat',
      size * size,
      0,
      [binding('design', design), binding('projected', projected), binding('solution', solution)],
      `let code = solution[solutionOffset + Q * Q + Q]; if (code != 0.0 && code != 4.0) { return; }
let start = designOffset + row * DESIGN_STRIDE; var residual = design[start + M + Q]; if (!isFiniteValue(residual)) { return; }
for (var column = 0u; column < Q; column++) { residual -= design[start + M + column] * solution[solutionOffset + Q * Q + column]; }
for (var first = 0u; first < Q; first++) { for (var second = 0u; second < Q; second++) { (*terms)[first * Q + second] += residual * residual * projected[projectedOffset + row * Q + first] * projected[projectedOffset + row * Q + second]; } }`,
      meat
    ),
    // contract-allow: single-invocation -- covariance transform over at most 13 coefficients, enforced by the contributor's 11-predictor cap.
    node(
      'publish',
      1,
      [
        binding('solution', solution),
        binding('meat', meat),
        binding('state', props.state),
        binding('coefficients', output.coefficients, 'read_write'),
        binding('errors', output.standardErrors, 'read_write'),
        binding('status', output.status, 'read_write', 'u32')
      ],
      `let code = u32(solution[solutionOffset + Q * Q + Q]); status[statusOffset] = code;
for (var column = 0u; column < Q; column++) { coefficients[coefficientsOffset + column] = getNaN(); errors[errorsOffset + column] = getNaN(); }
if (code != 0u && code != 4u) { return; }
var estimate: array<f32, ${size}>; var errorsResult: array<f32, ${size}>;
var intercept = solution[solutionOffset + Q * Q];
for (var column = 1u; column < P; column++) {
 let slope = solution[solutionOffset + Q * Q + column] * state[stateOffset + ${props.scalesSlot}u + column - 1u]; estimate[column] = slope;
 intercept -= state[stateOffset + ${props.meansSlot}u + column - 1u] * slope;
}
estimate[0] = intercept; estimate[Q - 1u] = solution[solutionOffset + Q * Q + Q - 1u];
for (var column = 0u; column < Q; column++) {
 var transform: array<f32, ${size}>;
 if (column == 0u) { transform[0] = 1.0; for (var other = 1u; other < P; other++) { transform[other] = -state[stateOffset + ${props.meansSlot}u + other - 1u] * state[stateOffset + ${props.scalesSlot}u + other - 1u]; } }
 else if (column < P) { transform[column] = state[stateOffset + ${props.scalesSlot}u + column - 1u]; } else { transform[column] = 1.0; }
 var influence: array<f32, ${size}>;
 for (var row = 0u; row < Q; row++) { for (var other = 0u; other < Q; other++) { influence[row] += solution[solutionOffset + row * Q + other] * transform[other]; } }
 var variance = 0.0; for (var row = 0u; row < Q; row++) { for (var other = 0u; other < Q; other++) { variance += influence[row] * meat[meatOffset + row * Q + other] * influence[other]; } }
 errorsResult[column] = sqrt(max(variance, 0.0));
 if (!isFiniteValue(estimate[column]) || !isFiniteValue(variance) || !isFiniteValue(errorsResult[column])) { status[statusOffset] = 3u; return; }
}
for (var column = 0u; column < Q; column++) { coefficients[coefficientsOffset + column] = estimate[column]; errors[errorsOffset + column] = errorsResult[column]; }`
    )
  ];
  return nodes;
}
