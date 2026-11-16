// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createClassificationKernelNode,
  type ClassificationKernelProps
} from './classification-kernel-node';
import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  getBoundedDispatchLayout,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  CLASS_BREAKS_EXTREMES,
  CLASS_BREAKS_STATE,
  getClassBreaksIndirectGate
} from './class-breaks-kernels';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS} from './natural-breaks-limits';
import {GPU_CLASS_BREAKS_METHOD_CODES} from './class-breaks-parameters';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';

/** Divide bounded mantissas, then apply the exponent difference without an extreme reciprocal. */
export const NATURAL_BREAKS_NORMALIZATION_WGSL = /* wgsl */ `fn getNormalizedValue(value: f32, scale: f32) -> f32 {
  let valueParts = frexp(value);
  let scaleParts = frexp(scale);
  return ldexp(valueParts.fract / scaleParts.fract, valueParts.exp - scaleParts.exp);
}`;

/** Full interval-cost/partition DP, with no monotonicity assumption or local refinement. */
export function getExactNaturalBreaksNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    values: GraphDataView<'float32'>;
    mask?: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    extremes: GraphDataView<'uint32'>;
    edges: GraphDataView<'float32'>;
    edgeState: GraphDataView<'uint32'>;
    maximumClassCount: number;
    exactStorage?: 'auto' | 'materialized' | 'recompute';
  }
): GPUCommandNode<Parameters>[] {
  const {values, mask, state, extremes, edges, edgeState, maximumClassCount} = props;
  const id = `${props.id}-exact-natural-breaks`;
  const rowCount = values.length;
  const stride = rowCount + 1;
  const referenceLayout = getBoundedDispatchLayout(
    'GPUClassBreaks',
    rowCount,
    64,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const createWGSLKernelNode = (
    targetGraph: GPUCommandGraph<Parameters>,
    kernelProps: ClassificationKernelProps<Parameters>
  ) =>
    createClassificationKernelNode(targetGraph, {
      ...kernelProps,
      parameterValues: [rowCount, stride, referenceLayout.x, referenceLayout.y, referenceLayout.z]
    });

  const make = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const cursor = make('cursor', 1);
  const dispatch = createTransientView(
    graph,
    `${id}-dispatch`,
    'uint32',
    3,
    Buffer.STORAGE | Buffer.INDIRECT
  );
  const keys = make('keys', rowCount);
  const rows = make('rows', rowCount);
  const sortedKeys = make('sorted-keys', rowCount);
  const sortedRows = make('sorted-rows', rowCount);
  const materialize =
    props.exactStorage === 'materialized' ||
    (props.exactStorage !== 'recompute' && rowCount <= NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS);
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'exact partitions',
    (maximumClassCount + 1) * stride * 4,
    'reduce rows or maximumClassCount'
  );
  if (materialize)
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'exact costs',
      rowCount * rowCount * 4,
      'use naturalBreaksExactStorage: recompute'
    );
  const costs = materialize
    ? createTransientView(graph, `${id}-costs`, 'float32', rowCount * rowCount)
    : undefined;
  const partitions = make('partitions', (maximumClassCount + 1) * stride);
  const scores = createTransientView(
    graph,
    `${id}-scores`,
    'float32',
    (materialize ? maximumClassCount + 1 : 2) * stride
  );
  const binding = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' = 'u32',
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({name, view, type, access});
  const gate = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] != ${GPU_CLASS_BREAKS_METHOD_CODES['natural-breaks']}u`;
  const finiteCount = `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u]`;
  const declarations = `${COLUMN_ORDERED_KEY_WGSL}\nfn getReferenceRowCount() -> u32 { return getChunkParameter(36u); }\nfn getReferenceStride() -> u32 { return getChunkParameter(37u); }`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-keys`,
      operation: 'GPUClassBreaks',
      variant: 'exact-keys',
      invocationCount: rowCount,
      bindings: [
        binding('values', values, 'f32'),
        ...(mask ? [binding('rowMask', mask)] : []),
        binding('keys', keys, 'u32', 'read_write'),
        binding('rows', rows, 'u32', 'read_write')
      ],
      declarations,
      body: `let value = values[valuesOffset + index];
  let include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u ${mask ? '&& rowMask[rowMaskOffset + index] != 0u' : ''};
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`
    })
  );
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort`,
      keys,
      values: rows,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      algorithm: 'radix'
    }).getCommandNodes(graph)
  );
  if (costs)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-costs`,
        operation: 'GPUClassBreaks',
        variant: 'exact-costs',
        invocationCount: rowCount,
        bindings: [
          binding('state', state),
          binding('extremes', extremes),
          binding('sortedKeys', sortedKeys),
          binding('costs', costs, 'f32', 'read_write')
        ],
        declarations: `${declarations}\n${NATURAL_BREAKS_NORMALIZATION_WGSL}`,
        body: `if (${gate}) { return; }
  let count = ${finiteCount};
  if (index >= count) { return; }
  let minimum = decodeOrderedKey(sortedKeys[sortedKeysOffset]);
  let maximum = decodeOrderedKey(sortedKeys[sortedKeysOffset + count - 1u]);
  let magnitude = max(abs(minimum), abs(maximum));
  // Power-of-two scaling preserves normal f32 mantissas and avoids overflow on opposite extremes.
  // Use the magnitude itself for subnormals; zero is the only scale replaced with one.
  let exponentScale = bitcast<f32>(bitcast<u32>(magnitude) & 0x7f800000u);
  let scale = select(select(1.0, magnitude, magnitude > 0.0), exponentScale, exponentScale > 0.0);
  let origin = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + index]), scale);
  var mean = 0.0;
  var variance = 0.0;
  for (var start = index + 1u; start > 0u;) {
    start = start - 1u;
    let value = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + start]), scale) - origin;
    let delta = value - mean;
    mean = mean + delta / f32(index - start + 1u);
    variance = variance + delta * (value - mean);
    costs[costsOffset + index * getReferenceRowCount() + start] = max(variance, 0.0);
  }`
      })
    );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-initialize`,
      operation: 'GPUClassBreaks',
      variant: 'exact-initialize',
      invocationCount: (maximumClassCount + 1) * stride,
      bindings: [
        binding('scores', scores, 'f32', 'read_write'),
        binding('partitions', partitions, 'u32', 'read_write'),
        binding('cursor', cursor, 'u32', 'read_write')
      ],
      body: `if (index == 0u) { cursor[cursorOffset] = 0u; }
  // Normalized finite interval costs are at most 16 * getReferenceRowCount(), far below this sentinel.
  if (index < (arrayLength(&scores) - scoresOffset)) { scores[scoresOffset + index] = select(1e30, 0.0, index == 0u); }
  partitions[partitionsOffset + index] = 0u;`
    })
  );
  for (let layer = 1; layer <= maximumClassCount; layer++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-advance-${layer}`,
        operation: 'GPUClassBreaks',
        variant: 'exact-advance',
        invocationCount: 1,
        bindings: [
          binding('cursor', cursor, 'u32', 'read_write'),
          binding('state', state),
          binding('dispatch', dispatch, 'u32', 'read_write')
        ],
        body: `let layer = cursor[cursorOffset] + 1u;
  cursor[cursorOffset] = layer;
  let isEnabled = !(${gate}) && layer <= state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u];
  dispatch[dispatchOffset] = select(0u, getChunkParameter(38u), isEnabled);
  dispatch[dispatchOffset + 1u] = getChunkParameter(39u);
  dispatch[dispatchOffset + 2u] = getChunkParameter(40u);`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-layer-${layer}`,
        operation: 'GPUClassBreaks',
        variant: 'exact-layer',
        invocationCount: rowCount,
        workgroupSize: 64,
        ...getClassBreaksIndirectGate<Parameters>(`${id}-layer-${layer}`, dispatch, 0),
        bindings: [
          binding('state', state),
          binding('cursor', cursor),
          binding('extremes', extremes),
          binding('sortedKeys', sortedKeys),
          ...(costs ? [binding('costs', costs, 'f32')] : []),
          binding('scores', scores, 'f32', 'read_write'),
          binding('partitions', partitions, 'u32', 'read_write')
        ],
        declarations: `${declarations}\n${NATURAL_BREAKS_NORMALIZATION_WGSL}`,
        body: `if (${gate}) { return; }
  let layer = cursor[cursorOffset];
  if (layer > state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]) { return; }
  let end = index + 1u;
  if (end > ${finiteCount}) { return; }
  var best = 1e30;
  var split = 0u;
  ${
    costs
      ? `for (var start = 0u; start < end; start = start + 1u) {
    if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
    let candidate = scores[scoresOffset + (layer - 1u) * getReferenceStride() + start] + costs[costsOffset + index * getReferenceRowCount() + start];
    if (candidate < best) { best = candidate; split = start; }
  }
  scores[scoresOffset + layer * getReferenceStride() + end] = best;`
      : `
  let minimum = decodeOrderedKey(sortedKeys[sortedKeysOffset]);
  let maximum = decodeOrderedKey(sortedKeys[sortedKeysOffset + ${finiteCount} - 1u]);
  let magnitude = max(abs(minimum), abs(maximum));
  let exponentScale = bitcast<f32>(bitcast<u32>(magnitude) & 0x7f800000u);
  let scale = select(select(1.0, magnitude, magnitude > 0.0), exponentScale, exponentScale > 0.0);
  let origin = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + index]), scale);
  var mean = 0.0;
  var variance = 0.0;
  for (var start = end; start > 0u;) {
    start--;
    // Keep every update, including duplicates: weighting or reordering changes this recurrence.
    let value = getNormalizedValue(decodeOrderedKey(sortedKeys[sortedKeysOffset + start]), scale) - origin;
    let delta = value - mean;
    mean = mean + delta / f32(end - start);
    variance = variance + delta * (value - mean);
    if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
    let candidate = scores[scoresOffset + ((layer - 1u) & 1u) * getReferenceStride() + start] + max(variance, 0.0);
    // Descending traversal still chooses the same lowest split as the ascending materialized path.
    if (candidate < best || (candidate == best && start < split)) { best = candidate; split = start; }
  }
  if (index == 0u) { scores[scoresOffset + (layer & 1u) * getReferenceStride()] = 1e30; }
  scores[scoresOffset + (layer & 1u) * getReferenceStride() + end] = best;`
  }
  partitions[partitionsOffset + layer * getReferenceStride() + end] = split;`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-backtrack`,
      operation: 'GPUClassBreaks',
      variant: 'exact-backtrack',
      invocationCount: 1,
      bindings: [
        binding('state', state),
        binding('extremes', extremes),
        binding('sortedKeys', sortedKeys),
        binding('partitions', partitions),
        binding('edges', edges, 'f32', 'read_write'),
        binding('edgeState', edgeState, 'u32', 'read_write')
      ],
      declarations,
      body: `if (${gate}) { return; }
  var end = ${finiteCount};
  var distinct = select(0u, 1u, end > 0u);
  for (var row = 1u; row < end; row = row + 1u) {
    if (decodeOrderedKey(sortedKeys[sortedKeysOffset + row - 1u]) != decodeOrderedKey(sortedKeys[sortedKeysOffset + row])) { distinct = distinct + 1u; }
  }
  let classes = min(distinct, state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]);
  edgeState[edgeStateOffset] = classes;
  for (var layer = classes; layer > 1u; layer = layer - 1u) {
    let split = partitions[partitionsOffset + layer * getReferenceStride() + end];
    edges[edgesOffset + layer - 1u] = decodeOrderedKey(sortedKeys[sortedKeysOffset + split]);
    end = split;
  }`
    })
  );
  return nodes;
}
