// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSymmetricEigenWGSL} from '../../analysis-core/kernels/symmetric-eigen-wgsl';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {VARIOGRAM_MODEL_SHAPE_WGSL} from './variogram-model';
import type {GPUCoregionalizationFit} from './gpu-coregionalization-fit';

/** @internal Reference-tier sufficient statistics, gated block updates, and publication. */
export function getCoregionalizationFitNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUCoregionalizationFit
): GPUCommandNode<Parameters>[] {
  const {id, props, maximumIterations} = contributor;
  const variableCount = props.variableCount;
  const structureCount = props.structureCount;
  const squareCount = variableCount ** 2;
  const coefficientCount = structureCount * squareCount;
  const targetOffset = structureCount ** 2;
  const matrixOffset = targetOffset + coefficientCount;
  const loadingOffset = matrixOffset + coefficientCount;
  const validityOffset = loadingOffset + coefficientCount;
  const iterationOffset = validityOffset + 1;
  const convergedOffset = iterationOffset + 1;
  const invalidOffset = convergedOffset + 1;
  const usableOffset = invalidOffset + 1;
  const weightOffset = usableOffset + 1;
  const changeOffset = weightOffset + 1;
  const stateLength = changeOffset + 1;
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'fit state',
    stateLength * 4,
    'Reduce variableCount or structureCount'
  );
  const state = createTransientView(graph, `${id}-state`, 'float32', stateLength);
  const stateBinding: WGSLKernelBinding = {
    name: 'state',
    view: state,
    type: 'f32',
    access: 'read_write'
  };
  const parameterBinding: WGSLKernelBinding = {
    name: 'parameters',
    view: props.parameters,
    type: 'f32',
    access: 'read'
  };
  const inputBindings: WGSLKernelBinding[] = [
    {
      name: 'distances',
      view: props.distances.length
        ? props.distances
        : createTransientView(graph, `${id}-empty-distances`, 'float32', 1),
      type: 'f32',
      access: 'read'
    },
    {
      name: 'semivariances',
      view: props.semivariances.length
        ? props.semivariances
        : createTransientView(graph, `${id}-empty-semivariances`, 'float32', 1),
      type: 'f32',
      access: 'read'
    },
    {
      name: 'pairCounts',
      view: props.pairCounts.length
        ? props.pairCounts
        : createTransientView(graph, `${id}-empty-pair-counts`, 'uint32', 1),
      type: 'u32',
      access: 'read'
    }
  ];
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-dispatch`,
    operation: 'GPUCoregionalizationFit',
    shapes: [[1, 32]]
  });
  const constants = `
const VARIABLE_COUNT: u32 = ${variableCount}u;
const STRUCTURE_COUNT: u32 = ${structureCount}u;
const SQUARE_COUNT: u32 = ${squareCount}u;
const COEFFICIENT_COUNT: u32 = ${coefficientCount}u;
const BIN_COUNT: u32 = ${props.distances.length}u;
const TRIANGLE_COUNT: u32 = ${(variableCount * (variableCount + 1)) / 2}u;
const TARGET: u32 = ${targetOffset}u;
const MATRIX: u32 = ${matrixOffset}u;
const LOADING: u32 = ${loadingOffset}u;
const VALIDITY: u32 = ${validityOffset}u;
const ITERATION: u32 = ${iterationOffset}u;
const CONVERGED: u32 = ${convergedOffset}u;
const INVALID: u32 = ${invalidOffset}u;
const USABLE: u32 = ${usableOffset}u;
const WEIGHT: u32 = ${weightOffset}u;
const CHANGE: u32 = ${changeOffset}u;
${IS_FINITE_VALUE_WGSL}
${GET_NAN_WGSL}`;
  const binDeclarations = `${constants}
${VARIOGRAM_MODEL_SHAPE_WGSL}
fn usableBin(bin: u32) -> bool {
  if (!isFiniteValue(distances[distancesOffset + bin]) || distances[distancesOffset + bin] <= 0.0 || pairCounts[pairCountsOffset + bin] == 0u) { return false; }
  for (var entry = 0u; entry < TRIANGLE_COUNT; entry++) {
    if (!isFiniteValue(semivariances[semivariancesOffset + bin * TRIANGLE_COUNT + entry])) { return false; }
  }
  return true;
}
fn structureShape(structure: u32, lag: f32) -> f32 {
  let start = parametersOffset + 1u + 4u * structure;
  return parameters[start + 1u] + parameters[start + 2u] * getVariogramModelShape(u32(parameters[start]), lag / parameters[start + 3u]);
}`;
  const createNode = (
    suffix: string,
    bindings: WGSLKernelBinding[],
    declarations: string,
    body: string
  ) =>
    createWGSLKernelNode(graph, {
      id: `${id}-${suffix}`,
      operation: 'GPUCoregionalizationFit',
      invocationCount: 1,
      workgroupSize: 32,
      bindings,
      declarations,
      body
    });
  const nodes: GPUCommandNode<Parameters>[] = [
    createNode(
      'initialize',
      [...inputBindings, parameterBinding, stateBinding, gate.binding],
      binDeclarations,
      `
for (var entry = 0u; entry < ${stateLength}u; entry++) { state[stateOffset + entry] = 0.0; }
${gate.disarmWGSL}
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  if (usableBin(bin)) {
    state[stateOffset + USABLE] += 1.0;
    state[stateOffset + WEIGHT] += f32(pairCounts[pairCountsOffset + bin]);
  } else { state[stateOffset + INVALID] += 1.0; }
}
if (!isFiniteValue(parameters[parametersOffset]) || parameters[parametersOffset] <= 0.0 || parameters[parametersOffset] > 1.0) { return; }
for (var structure = 0u; structure < STRUCTURE_COUNT; structure++) {
  let start = parametersOffset + 1u + 4u * structure;
  let model = parameters[start];
  let nugget = parameters[start + 1u];
  let sill = parameters[start + 2u];
  let rangeValue = parameters[start + 3u];
  if (!isFiniteValue(model) || model < 0.0 || model > 4.0 || floor(model) != model ||
      !isFiniteValue(nugget) || nugget < 0.0 || !isFiniteValue(sill) || sill < 0.0 ||
      !isFiniteValue(nugget + sill) || nugget + sill <= 0.0 || !isFiniteValue(rangeValue) || rangeValue <= 0.0) { return; }
}
if (state[stateOffset + USABLE] == 0.0) { return; }
for (var bin = 0u; bin < BIN_COUNT; bin++) {
  if (!usableBin(bin)) { continue; }
  let weight = f32(pairCounts[pairCountsOffset + bin]) / state[stateOffset + WEIGHT];
  for (var structure = 0u; structure < STRUCTURE_COUNT; structure++) {
    let shape = structureShape(structure, distances[distancesOffset + bin]);
    for (var other = 0u; other < STRUCTURE_COUNT; other++) {
      state[stateOffset + structure * STRUCTURE_COUNT + other] += weight * shape * structureShape(other, distances[distancesOffset + bin]);
    }
    var triangle = 0u;
    for (var row = 0u; row < VARIABLE_COUNT; row++) {
      for (var column = row; column < VARIABLE_COUNT; column++) {
        let value = weight * shape * semivariances[semivariancesOffset + bin * TRIANGLE_COUNT + triangle];
        state[stateOffset + TARGET + structure * SQUARE_COUNT + row * VARIABLE_COUNT + column] += value;
        if (row != column) { state[stateOffset + TARGET + structure * SQUARE_COUNT + column * VARIABLE_COUNT + row] += value; }
        triangle++;
      }
    }
  }
}
for (var entry = 0u; entry < MATRIX; entry++) {
  if (!isFiniteValue(state[stateOffset + entry])) { return; }
}
for (var structure = 0u; structure < STRUCTURE_COUNT; structure++) {
  if (state[stateOffset + structure * STRUCTURE_COUNT + structure] <= 0.0) { return; }
}
state[stateOffset + VALIDITY] = 1.0;
${gate.armWGSL}`
    )
  ];
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-sweep-${iteration}`,
        operation: 'GPUCoregionalizationFit',
        // contract-allow: single-invocation reference tier; four bounded order-four PSD projections.
        invocationCount: 1,
        workgroupSize: 32,
        ...gate.getGate(0),
        bindings: [parameterBinding, stateBinding],
        declarations: `${constants}\n${getSymmetricEigenWGSL(variableCount)}`,
        body: `
var previous: array<f32, ${coefficientCount}>;
for (var entry = 0u; entry < COEFFICIENT_COUNT; entry++) { previous[entry] = state[stateOffset + MATRIX + entry]; }
for (var structure = 0u; structure < STRUCTURE_COUNT; structure++) {
  var matrix: array<f32, ${squareCount}>;
  var vectors: array<f32, ${squareCount}>;
  let denominator = state[stateOffset + structure * STRUCTURE_COUNT + structure];
  for (var entry = 0u; entry < SQUARE_COUNT; entry++) {
    var numerator = state[stateOffset + TARGET + structure * SQUARE_COUNT + entry];
    for (var other = 0u; other < STRUCTURE_COUNT; other++) {
      if (other != structure) { numerator -= state[stateOffset + structure * STRUCTURE_COUNT + other] * state[stateOffset + MATRIX + other * SQUARE_COUNT + entry]; }
    }
    matrix[entry] = numerator / denominator;
  }
  let eigen = getSymmetricEigen_${variableCount}(&matrix, &vectors, VARIABLE_COUNT, 24u, 1e-7);
  if (eigen.status != 0u) { state[stateOffset + VALIDITY] = 0.0; return; }
  for (var row = 0u; row < VARIABLE_COUNT; row++) {
    for (var column = 0u; column < VARIABLE_COUNT; column++) {
      let loading = vectors[row * VARIABLE_COUNT + column] * sqrt(max(eigen.values[column], 0.0));
      if (!isFiniteValue(loading)) { state[stateOffset + VALIDITY] = 0.0; return; }
      state[stateOffset + LOADING + structure * SQUARE_COUNT + row * VARIABLE_COUNT + column] = loading;
    }
  }
  for (var row = 0u; row < VARIABLE_COUNT; row++) {
    for (var column = 0u; column < VARIABLE_COUNT; column++) {
      var value = 0.0;
      for (var component = 0u; component < VARIABLE_COUNT; component++) {
        value += state[stateOffset + LOADING + structure * SQUARE_COUNT + row * VARIABLE_COUNT + component] *
          state[stateOffset + LOADING + structure * SQUARE_COUNT + column * VARIABLE_COUNT + component];
      }
      if (!isFiniteValue(value)) { state[stateOffset + VALIDITY] = 0.0; return; }
      state[stateOffset + MATRIX + structure * SQUARE_COUNT + row * VARIABLE_COUNT + column] = value;
    }
  }
}
var largest = 0.0;
var change = 0.0;
for (var entry = 0u; entry < COEFFICIENT_COUNT; entry++) {
  let value = state[stateOffset + MATRIX + entry];
  largest = max(largest, max(abs(value), abs(previous[entry])));
  change = max(change, abs(value - previous[entry]));
}
state[stateOffset + ITERATION] += 1.0;
var relativeChange = 0.0;
if (largest > 0.0) { relativeChange = change / largest; }
state[stateOffset + CHANGE] = relativeChange;
if (change <= parameters[parametersOffset] * largest) { state[stateOffset + CONVERGED] = 1.0; }`
      })
    );
    nodes.push(
      createNode(
        `gate-${iteration}`,
        [stateBinding, gate.binding],
        constants,
        `
if (state[stateOffset + VALIDITY] == 0.0 || state[stateOffset + CONVERGED] != 0.0) { ${gate.disarmWGSL} }`
      )
    );
  }
  nodes.push(
    createNode(
      'publish-fit',
      [
        ...inputBindings,
        parameterBinding,
        stateBinding,
        {name: 'matrices', view: props.output.matrices, type: 'f32', access: 'read_write'},
        {name: 'loadings', view: props.output.loadings, type: 'f32', access: 'read_write'},
        {name: 'diagnostics', view: props.output.diagnostics, type: 'f32', access: 'read_write'}
      ],
      binDeclarations,
      `
var residual = 0.0;
if (state[stateOffset + VALIDITY] != 0.0) {
  for (var bin = 0u; bin < BIN_COUNT; bin++) {
    if (!usableBin(bin)) { continue; }
    let weight = f32(pairCounts[pairCountsOffset + bin]) / state[stateOffset + WEIGHT];
    var triangle = 0u;
    for (var row = 0u; row < VARIABLE_COUNT; row++) {
      for (var column = row; column < VARIABLE_COUNT; column++) {
        var predicted = 0.0;
        for (var structure = 0u; structure < STRUCTURE_COUNT; structure++) {
          predicted += structureShape(structure, distances[distancesOffset + bin]) * state[stateOffset + MATRIX + structure * SQUARE_COUNT + row * VARIABLE_COUNT + column];
        }
        let difference = predicted - semivariances[semivariancesOffset + bin * TRIANGLE_COUNT + triangle];
        residual += weight * difference * difference * select(2.0, 1.0, row == column);
        triangle++;
      }
    }
  }
  if (!isFiniteValue(residual)) { state[stateOffset + VALIDITY] = 0.0; }
}
let valid = state[stateOffset + VALIDITY] != 0.0;
for (var entry = 0u; entry < COEFFICIENT_COUNT; entry++) {
  matrices[matricesOffset + entry] = select(getNaN(), state[stateOffset + MATRIX + entry], valid);
  loadings[loadingsOffset + entry] = select(getNaN(), state[stateOffset + LOADING + entry], valid);
}
diagnostics[diagnosticsOffset] = select(getNaN(), residual, valid);
diagnostics[diagnosticsOffset + 1u] = select(getNaN(), state[stateOffset + CHANGE], valid);
diagnostics[diagnosticsOffset + 2u] = state[stateOffset + USABLE];
diagnostics[diagnosticsOffset + 3u] = state[stateOffset + WEIGHT];`
    )
  );
  nodes.push(
    createNode(
      'publish-status',
      [
        stateBinding,
        ...Object.entries(props.output.status).map(([name, view]) => ({
          name,
          view,
          type: 'u32' as const,
          access: 'read_write' as const
        }))
      ],
      constants,
      `
let valid = state[stateOffset + VALIDITY] != 0.0;
validity[validityOffset] = u32(valid);
iterationCount[iterationCountOffset] = u32(state[stateOffset + ITERATION]);
converged[convergedOffset] = u32(valid && state[stateOffset + CONVERGED] != 0.0);
iterationLimitReached[iterationLimitReachedOffset] = u32(valid && state[stateOffset + CONVERGED] == 0.0 && state[stateOffset + ITERATION] >= ${maximumIterations}.0);
invalidCount[invalidCountOffset] = u32(state[stateOffset + INVALID]);`
    )
  );
  return nodes;
}
