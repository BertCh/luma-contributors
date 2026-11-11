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
  createWGSLDispatchGate,
  getWGSLFloatLiteral,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import type {GPUConditionalGaussianSimulationProps} from './gpu-conditional-gaussian-simulation';

const BLOCK_SIZE = 16;
const TILE_INVOCATIONS = BLOCK_SIZE * BLOCK_SIZE;

type BlockedSimulation = {
  id: string;
  props: GPUConditionalGaussianSimulationProps;
  capacity: number;
  scratch: GraphDataView<'float32'>;
  moments: GraphDataView<'float32x2'>;
  control?: GraphDataView<'uint32'>;
  preparationState?: GraphDataView<'uint32'>;
  declarations: string;
  conditionBindings: readonly WGSLKernelBinding[];
  observationStart: number;
  errorStart: number;
  whitenedStart: number;
  meanExpression: string;
};

/**
 * Right-looking dense Cholesky with one small diagonal solve, parallel panel solves and tiled
 * trailing updates per block. Dispatch boundaries synchronize blocks; no CPU readback or global
 * spin barrier is needed. Zero query pivots retain the semidefinite conditional-field contract.
 * @internal
 */
export function getBlockedConditionalSimulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  simulation: BlockedSimulation
): GPUCommandNode<Parameters>[] {
  const {id, props, capacity, scratch, moments, control} = simulation;
  const queryCount = props.queries.length;
  const failure = createTransientView(graph, `${id}-blocked-failure`, 'uint32', 1);
  const bindings: readonly WGSLKernelBinding[] = [
    ...simulation.conditionBindings,
    {name: 'failure', view: failure, type: 'atomic<u32>', access: 'read_write'}
  ];
  const factorGate =
    'scratch[scratchOffset] == 0.0' +
    (control ? ' && preparationState[preparationStateOffset + 2u] != 0u' : '');
  const momentsGate =
    'scratch[scratchOffset] == 0.0' +
    (control ? ' && preparationState[preparationStateOffset + 3u] != 0u' : '');
  const tolerance = getWGSLFloatLiteral(
    (props.covariance.sill + (props.covariance.nugget ?? 0)) * 1e-6
  );
  const declarations = `${simulation.declarations}
const BLOCK_SIZE: u32 = ${BLOCK_SIZE}u;
const TOLERANCE: f32 = ${tolerance};
fn observedCount() -> u32 { return u32(scratch[scratchOffset + 1u]); }
fn jointSize() -> u32 { return observedCount() + ${queryCount}u; }
// Query rows do not hold observed values. Their slots retain exact-coordinate aliases instead.
fn queryAlias(row: u32) -> u32 { return u32(scratch[scratchOffset + ${simulation.observationStart}u + row]); }
fn getWhitened(row: u32) -> f32 { return scratch[scratchOffset + ${simulation.whitenedStart}u + row]; }
fn writeWhitened(row: u32, value: f32) { scratch[scratchOffset + ${simulation.whitenedStart}u + row] = value; }`;
  const makeNode = (
    variant: string,
    options: Omit<WGSLKernelProps<Parameters>, 'id' | 'operation' | 'variant' | 'bindings'>,
    suffix = ''
  ): WGSLKernelProps<Parameters> => ({
    id: `${id}-${variant}${suffix}`,
    operation: 'GPUConditionalGaussianSimulation',
    variant,
    bindings,
    ...options,
    declarations: declarations + (options.declarations ?? '')
  });
  const preparationKernels: WGSLKernelProps<Parameters>[] = [
    makeNode('blocked-covariance', {
      invocationCount: capacity * capacity,
      body: `if (!(${factorGate})) { return; }
let row = index / ${capacity}u;
let column = index % ${capacity}u;
if (row >= jointSize() || column > row) { return; }
var value = covariance(point(row), point(column));
if (row == column) { value += scratch[scratchOffset + ${simulation.errorStart}u + row]; }
writeFactor(row, column, value);`
    }),
    makeNode('blocked-query-aliases', {
      invocationCount: queryCount,
      body: `if (!(${factorGate})) { return; }
let row = observedCount() + index;
var canonicalRow = row;
for (var candidate = 0u; candidate < row; candidate++) {
  // Covariance includes the process nugget at identical coordinates. Only observation
  // measurement error distinguishes an observed variable from the latent query variable.
  if (all(point(candidate) == point(row)) && (candidate >= observedCount() || scratch[scratchOffset + ${simulation.errorStart}u + candidate] == 0.0)) {
    canonicalRow = candidate; break;
  }
}
scratch[scratchOffset + ${simulation.observationStart}u + row] = f32(canonicalRow);`
    })
  ];
  for (let blockStart = 0; blockStart < capacity; blockStart += BLOCK_SIZE) {
    const constants = {BLOCK_START: blockStart};
    preparationKernels.push(
      makeNode(
        'blocked-diagonal',
        {
          invocationCount: 1,
          declarations: '\noverride BLOCK_START: u32;',
          constants,
          body: `if (!(${factorGate}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let blockEnd = min(BLOCK_START + BLOCK_SIZE, jointSize());
for (var column = BLOCK_START; column < blockEnd; column++) {
  var diagonal = factor(column, column);
  for (var component = BLOCK_START; component < column; component++) {
    let loading = factor(column, component); diagonal -= loading * loading;
  }
  if (!isFiniteValue(diagonal) || diagonal < -TOLERANCE || (column < observedCount() && diagonal <= TOLERANCE)) {
    atomicStore(&failure[failureOffset], 1u); return;
  }
  if (column >= observedCount() && queryAlias(column) < column) {
    // An exact duplicate has no new random degree of freedom. Do not turn cancellation
    // residue into independent noise, or clamp small positive pivots of distinct sites.
    if (abs(diagonal) > TOLERANCE) { atomicStore(&failure[failureOffset], 1u); return; }
    diagonal = 0.0;
  }
  diagonal = sqrt(max(diagonal, 0.0));
  writeFactor(column, column, diagonal);
  for (var row = column + 1u; row < blockEnd; row++) {
    var value = factor(row, column);
    for (var component = BLOCK_START; component < column; component++) { value -= factor(row, component) * factor(column, component); }
    if (diagonal == 0.0) {
      if (abs(value) > TOLERANCE || !isFiniteValue(value)) { atomicStore(&failure[failureOffset], 1u); return; }
      writeFactor(row, column, 0.0);
    } else {
      let loading = value / diagonal;
      if (!isFiniteValue(loading)) { atomicStore(&failure[failureOffset], 1u); return; }
      writeFactor(row, column, loading);
    }
  }
}`
        },
        `-${blockStart}`
      )
    );
    const remainingRows = capacity - blockStart - BLOCK_SIZE;
    if (remainingRows <= 0) continue;
    preparationKernels.push(
      makeNode(
        'blocked-panel',
        {
          invocationCount: remainingRows,
          declarations: '\noverride BLOCK_START: u32;',
          constants,
          body: `if (!(${factorGate}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let row = BLOCK_START + BLOCK_SIZE + index;
if (row >= jointSize()) { return; }
for (var column = BLOCK_START; column < BLOCK_START + BLOCK_SIZE; column++) {
  var value = factor(row, column);
  for (var component = BLOCK_START; component < column; component++) { value -= factor(row, component) * factor(column, component); }
  let diagonal = factor(column, column);
  if (diagonal == 0.0) {
    if (abs(value) > TOLERANCE || !isFiniteValue(value)) { atomicStore(&failure[failureOffset], 1u); return; }
    writeFactor(row, column, 0.0);
  } else {
    let loading = value / diagonal;
    if (!isFiniteValue(loading)) { atomicStore(&failure[failureOffset], 1u); return; }
    writeFactor(row, column, loading);
  }
}`
        },
        `-${blockStart}`
      )
    );
    const tilesPerAxis = Math.ceil(remainingRows / BLOCK_SIZE);
    preparationKernels.push(
      makeNode(
        'blocked-update',
        {
          invocationCount: tilesPerAxis * tilesPerAxis * TILE_INVOCATIONS,
          workgroupSize: TILE_INVOCATIONS,
          guardIndex: false,
          declarations: `
override BLOCK_START: u32;
override TILES_PER_AXIS: u32;
var<workgroup> workgroupActive: u32;
var<workgroup> rowTile: array<f32, ${TILE_INVOCATIONS}>;
var<workgroup> columnTile: array<f32, ${TILE_INVOCATIONS}>;`,
          constants: {...constants, TILES_PER_AXIS: tilesPerAxis},
          body: `let tileRow = workgroupIndex / TILES_PER_AXIS;
let tileColumn = workgroupIndex % TILES_PER_AXIS;
// Tile coordinates and the workgroup-uniform gate are uniform across every barrier.
if (tileColumn > tileRow) { return; }
if (localInvocationIndex == 0u) { workgroupActive = u32(${factorGate} && atomicLoad(&failure[failureOffset]) == 0u); }
if (workgroupUniformLoad(&workgroupActive) == 0u) { return; }
let localRow = localInvocationIndex / BLOCK_SIZE;
let localColumn = localInvocationIndex % BLOCK_SIZE;
let row = BLOCK_START + BLOCK_SIZE + tileRow * BLOCK_SIZE + localRow;
let column = BLOCK_START + BLOCK_SIZE + tileColumn * BLOCK_SIZE + localColumn;
rowTile[localInvocationIndex] = 0.0;
columnTile[localInvocationIndex] = 0.0;
if (row < jointSize()) { rowTile[localInvocationIndex] = factor(row, BLOCK_START + localColumn); }
if (column < jointSize()) { columnTile[localInvocationIndex] = factor(column, BLOCK_START + localRow); }
workgroupBarrier();
if (row < jointSize() && column <= row) {
  var value = factor(row, column);
  for (var component = 0u; component < BLOCK_SIZE; component++) {
    value -= rowTile[localRow * BLOCK_SIZE + component] * columnTile[component * BLOCK_SIZE + localColumn];
  }
  writeFactor(row, column, value);
}`
        },
        `-${blockStart}`
      )
    );
  }
  preparationKernels.push(
    makeNode('blocked-factor-status', {
      invocationCount: 1,
      body: `if (!(${factorGate})) { return; }
if (atomicLoad(&failure[failureOffset]) != 0u) { scratch[scratchOffset] = 2.0; return; }
${control ? 'preparationState[preparationStateOffset + 6u] = 1u; preparationState[preparationStateOffset] = preparationControl[preparationControlOffset];' : ''}`
    }),
    makeNode('blocked-whiten-initialize', {
      invocationCount: Math.max(props.positions.length, 1),
      body: `if (!(${momentsGate})) { return; }
${control ? 'if (index == 0u) { preparationState[preparationStateOffset + 5u]++; }' : ''}
if (index < observedCount()) { writeWhitened(index, scratch[scratchOffset + ${simulation.observationStart}u + index]); }`
    })
  );
  for (let blockStart = 0; blockStart < props.positions.length; blockStart += BLOCK_SIZE) {
    preparationKernels.push(
      makeNode(
        'blocked-whiten-diagonal',
        {
          invocationCount: 1,
          declarations: '\noverride BLOCK_START: u32;',
          constants: {BLOCK_START: blockStart},
          body: `if (!(${momentsGate}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let blockEnd = min(BLOCK_START + BLOCK_SIZE, observedCount());
for (var row = BLOCK_START; row < blockEnd; row++) {
  var value = getWhitened(row);
  for (var column = BLOCK_START; column < row; column++) { value -= factor(row, column) * getWhitened(column); }
  value /= factor(row, row);
  if (!isFiniteValue(value)) { atomicStore(&failure[failureOffset], 1u); return; }
  writeWhitened(row, value);
}`
        },
        `-${blockStart}`
      )
    );
    const remainingRows = props.positions.length - blockStart - BLOCK_SIZE;
    if (remainingRows <= 0) continue;
    preparationKernels.push(
      makeNode(
        'blocked-whiten-update',
        {
          invocationCount: remainingRows,
          declarations: '\noverride BLOCK_START: u32;',
          constants: {BLOCK_START: blockStart},
          body: `if (!(${momentsGate}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let row = BLOCK_START + BLOCK_SIZE + index;
if (row >= observedCount()) { return; }
var value = getWhitened(row);
for (var column = BLOCK_START; column < BLOCK_START + BLOCK_SIZE; column++) { value -= factor(row, column) * getWhitened(column); }
writeWhitened(row, value);`
        },
        `-${blockStart}`
      )
    );
  }
  preparationKernels.push(
    makeNode('blocked-moments', {
      invocationCount: queryCount,
      body: `if (!(${momentsGate}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let row = observedCount() + index;
var expectation = ${simulation.meanExpression};
var variance = 0.0;
for (var column = 0u; column < observedCount(); column++) { expectation += factor(row, column) * getWhitened(column); }
for (var component = 0u; component <= index; component++) { let loading = factor(row, observedCount() + component); variance += loading * loading; }
if (!isFiniteValue(expectation) || !isFiniteValue(variance)) { atomicStore(&failure[failureOffset], 1u); return; }
if (queryAlias(row) < observedCount()) {
  expectation = ${simulation.meanExpression} + scratch[scratchOffset + ${simulation.observationStart}u + queryAlias(row)];
  variance = 0.0;
}
moments[momentsOffset + index * 2u] = expectation;
moments[momentsOffset + index * 2u + 1u] = variance;`
    }),
    makeNode('blocked-moments-status', {
      invocationCount: 1,
      body: `if (!(${momentsGate})) { return; }
if (atomicLoad(&failure[failureOffset]) != 0u) { scratch[scratchOffset] = 2.0; return; }
${control ? 'preparationState[preparationStateOffset + 1u] = preparationControl[preparationControlOffset + 1u];' : ''}`
    })
  );
  // Retained draws launch no preparation workgroups. One GPU updater arms all static shapes;
  // failures inside preparation are handled by the kernels' status guards.
  const dispatch = createWGSLDispatchGate(graph, {
    id: `${id}-blocked-preparation-dispatch`,
    operation: 'GPUConditionalGaussianSimulation',
    shapes: preparationKernels.map(kernel => [kernel.invocationCount, kernel.workgroupSize])
  });
  const factorKernelCount = preparationKernels.findIndex(
    kernel => kernel.variant === 'blocked-whiten-initialize'
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-blocked-preparation-gates`,
      operation: 'GPUConditionalGaussianSimulation',
      variant: 'blocked-preparation-gates',
      invocationCount: 1,
      bindings: [
        {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
        {name: 'failure', view: failure, type: 'atomic<u32>', access: 'read_write'},
        dispatch.binding,
        ...(control && simulation.preparationState
          ? [
              {
                name: 'preparationControl',
                view: control,
                type: 'u32' as const,
                access: 'read' as const
              },
              {
                name: 'preparationState',
                view: simulation.preparationState,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      body: `atomicStore(&failure[failureOffset], 0u);
${dispatch.armWGSL}
if (!(${factorGate})) {
  for (var slot = 0u; slot < ${factorKernelCount}u; slot++) { dispatch[dispatchOffset + slot * 3u] = 0u; }
}${control ? ' else { preparationState[preparationStateOffset + 4u]++; }' : ''}
if (!(${momentsGate})) {
  for (var slot = ${factorKernelCount}u; slot < ${preparationKernels.length}u; slot++) { dispatch[dispatchOffset + slot * 3u] = 0u; }
}`
    }),
    ...preparationKernels.map((kernel, slot) =>
      createWGSLKernelNode(graph, {...kernel, ...dispatch.getGate(slot)})
    ),
    createWGSLKernelNode(graph, {
      id: `${id}-blocked-publish-moments`,
      operation: 'GPUConditionalGaussianSimulation',
      variant: 'blocked-publish-moments',
      invocationCount: queryCount,
      bindings: [
        {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
        {name: 'moments', view: moments, type: 'f32', access: 'read_write'},
        ...(control
          ? [
              {
                name: 'outputMoments',
                view: props.output.moments,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      declarations: GET_NAN_WGSL,
      body: `if (scratch[scratchOffset] != 0.0) { moments[momentsOffset + index * 2u] = getNaN(); moments[momentsOffset + index * 2u + 1u] = getNaN(); }
else {
  let observedCount = u32(scratch[scratchOffset + 1u]);
  let canonicalRow = u32(scratch[scratchOffset + ${simulation.observationStart}u + observedCount + index]);
  // Aliases always refer to the first matching variable, so canonical query moments
  // are read-only throughout this pass and there are no read/write alias chains.
  if (canonicalRow >= observedCount && canonicalRow < observedCount + index) {
    moments[momentsOffset + index * 2u] = moments[momentsOffset + (canonicalRow - observedCount) * 2u];
    moments[momentsOffset + index * 2u + 1u] = moments[momentsOffset + (canonicalRow - observedCount) * 2u + 1u];
  }
}
${control ? 'outputMoments[outputMomentsOffset + index * 2u] = moments[momentsOffset + index * 2u]; outputMoments[outputMomentsOffset + index * 2u + 1u] = moments[momentsOffset + index * 2u + 1u];' : ''}`
    }),
    ...getTiledRealizationNodes(graph, simulation)
  ];
  return nodes;
}

/** Multiplies one shared conditional square root by explicitly batched Philox normal columns. */
function getTiledRealizationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  simulation: BlockedSimulation
): GPUCommandNode<Parameters>[] {
  const {id, props, capacity, scratch, moments, control} = simulation;
  const queryCount = props.queries.length;
  const normalCount = queryCount * props.replicateCount;
  const normals = createTransientView(graph, `${id}-normal-draws`, 'float32', normalCount);
  const controlBinding: WGSLKernelBinding[] = control
    ? [{name: 'preparationControl', view: control, type: 'u32', access: 'read'}]
    : [];
  const replicateCount = control
    ? 'preparationControl[preparationControlOffset + 5u]'
    : `${props.replicateCount}u`;
  const replicateOffset = control
    ? 'preparationControl[preparationControlOffset + 4u]'
    : `${props.replicateOffset ?? 0}u`;
  const seed = control ? 'preparationControl[preparationControlOffset + 3u]' : `${props.seed}u`;
  const queryTiles = Math.ceil(queryCount / BLOCK_SIZE);
  const replicateTiles = Math.ceil(props.replicateCount / BLOCK_SIZE);
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-blocked-normals`,
      operation: 'GPUConditionalGaussianSimulation',
      variant: 'blocked-normals',
      invocationCount: normalCount,
      bindings: [
        {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
        {name: 'normals', view: normals, type: 'f32', access: 'read_write'},
        {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'},
        ...controlBinding
      ],
      declarations: PHILOX_WGSL,
      body: `let replicate = index / ${queryCount}u;
let component = index % ${queryCount}u;
let status = select(u32(scratch[scratchOffset]), 1u, replicate >= ${replicateCount});
if (component == 0u) { statuses[statusesOffset + replicate] = status; }
normals[normalsOffset + index] = 0.0;
if (status != 0u) { return; }
let random = philox4x32(vec4<u32>(replicate + ${replicateOffset}, component, 0u, 0u), vec2<u32>(${seed}, ${PHILOX_PURPOSES.conditionalGaussianSimulation.id}u));
// Box-Muller with an open radius uniform, as in the cooperative path (same words, same draws).
normals[normalsOffset + index] = sqrt(-2.0 * log(philoxOpenUnitFloat(random.x))) * cos(6.28318530718 * (f32(random.y >> 8u) + 0.5) / 16777216.0);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-blocked-realizations`,
      operation: 'GPUConditionalGaussianSimulation',
      variant: 'blocked-realizations',
      invocationCount: queryTiles * replicateTiles * TILE_INVOCATIONS,
      workgroupSize: TILE_INVOCATIONS,
      guardIndex: false,
      bindings: [
        {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
        {name: 'moments', view: moments, type: 'f32', access: 'read'},
        {name: 'normals', view: normals, type: 'f32', access: 'read'},
        {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'},
        {name: 'statuses', view: props.output.status, type: 'atomic<u32>', access: 'read_write'},
        ...controlBinding
      ],
      declarations: `${IS_FINITE_VALUE_WGSL}
var<workgroup> workgroupActive: u32;
var<workgroup> factorTile: array<f32, ${TILE_INVOCATIONS}>;
var<workgroup> normalTile: array<f32, ${TILE_INVOCATIONS}>;`,
      body: `let localQuery = localInvocationIndex / ${BLOCK_SIZE}u;
let localReplicate = localInvocationIndex % ${BLOCK_SIZE}u;
let query = (workgroupIndex / ${replicateTiles}u) * ${BLOCK_SIZE}u + localQuery;
let replicate = (workgroupIndex % ${replicateTiles}u) * ${BLOCK_SIZE}u + localReplicate;
if (localInvocationIndex == 0u) { workgroupActive = u32(scratch[scratchOffset] == 0.0); }
if (workgroupUniformLoad(&workgroupActive) == 0u) { return; }
let observedCount = u32(scratch[scratchOffset + 1u]);
var value = 0.0;
if (query < ${queryCount}u) { value = moments[momentsOffset + query * 2u]; }
for (var componentStart = 0u; componentStart < ${queryCount}u; componentStart += ${BLOCK_SIZE}u) {
  factorTile[localInvocationIndex] = 0.0;
  normalTile[localInvocationIndex] = 0.0;
  let factorColumn = componentStart + localReplicate;
  if (query < ${queryCount}u && factorColumn <= query) { factorTile[localInvocationIndex] = scratch[scratchOffset + 2u + (observedCount + query) * ${capacity}u + observedCount + factorColumn]; }
  let normalComponent = componentStart + localQuery;
  if (replicate < ${replicateCount} && normalComponent < ${queryCount}u) { normalTile[localInvocationIndex] = normals[normalsOffset + replicate * ${queryCount}u + normalComponent]; }
  workgroupBarrier();
  for (var component = 0u; component < ${BLOCK_SIZE}u; component++) {
    if (componentStart + component <= query) { value += factorTile[localQuery * ${BLOCK_SIZE}u + component] * normalTile[component * ${BLOCK_SIZE}u + localReplicate]; }
  }
  workgroupBarrier();
}
if (query < ${queryCount}u && replicate < ${replicateCount}) {
  values[valuesOffset + replicate * ${queryCount}u + query] = value;
  if (!isFiniteValue(value)) { atomicMax(&statuses[statusesOffset + replicate], 2u); }
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-blocked-aliased-realizations`,
      operation: 'GPUConditionalGaussianSimulation',
      variant: 'blocked-aliased-realizations',
      invocationCount: normalCount,
      bindings: [
        {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
        {name: 'moments', view: moments, type: 'f32', access: 'read'},
        {name: 'statuses', view: props.output.status, type: 'u32', access: 'read'},
        {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'}
      ],
      body: `let replicate = index / ${queryCount}u;
if (statuses[statusesOffset + replicate] != 0u) { return; }
let query = index % ${queryCount}u;
let observedCount = u32(scratch[scratchOffset + 1u]);
let canonicalRow = u32(scratch[scratchOffset + ${simulation.observationStart}u + observedCount + query]);
if (canonicalRow < observedCount) { values[valuesOffset + index] = moments[momentsOffset + query * 2u]; }
else if (canonicalRow < observedCount + query) {
  // The earliest matching query never writes in this pass; aliases cannot form chains.
  values[valuesOffset + index] = values[valuesOffset + replicate * ${queryCount}u + canonicalRow - observedCount];
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-blocked-realization-status`,
      operation: 'GPUConditionalGaussianSimulation',
      variant: 'blocked-realization-status',
      invocationCount: normalCount,
      bindings: [
        {name: 'statuses', view: props.output.status, type: 'u32', access: 'read'},
        {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'}
      ],
      declarations: GET_NAN_WGSL,
      body: `if (statuses[statusesOffset + index / ${queryCount}u] != 0u) { values[valuesOffset + index] = getNaN(); }`
    })
  ];
}
