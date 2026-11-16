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
  GPUScan,
  GPUScanUint64,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  CLASS_BREAKS_EXTREMES,
  CLASS_BREAKS_STATE,
  CLASS_BREAKS_WIDE_WGSL,
  getClassBreaksIndirectGate
} from './class-breaks-kernels';
import {GPU_CLASS_BREAKS_METHOD_CODES} from './class-breaks-parameters';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';
import {NATURAL_BREAKS_NORMALIZATION_WGSL} from './exact-natural-breaks-nodes';

/**
 * Full-data monotone DP on 20-bit fixed-point positions. This is an explicit approximation:
 * quantization and f32 score comparisons can change the optimum and invalidate monotone pruning.
 * Unlike the histogram method, candidates are full source-value boundaries, with monotone pruning. The search
 * normally uses O(k n log n) work and O(k n) storage, one dispatch per tree level. An empty
 * pruned interval falls back to a full endpoint search, so worst-case work remains O(k n²).
 * Integer prefix moments avoid cancellation; they do not make the floating DP mathematically exact.
 */
export function getMonotoneNaturalBreaksNodes<Parameters>(
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
  }
): GPUCommandNode<Parameters>[] {
  const {values, mask, state, extremes, edges, edgeState, maximumClassCount} = props;
  const id = `${props.id}-monotone-natural-breaks`;
  const rowCount = values.length;
  const stride = rowCount + 1;
  const createWGSLKernelNode = (
    targetGraph: GPUCommandGraph<Parameters>,
    kernelProps: ClassificationKernelProps<Parameters>
  ) =>
    createClassificationKernelNode(targetGraph, {
      ...kernelProps,
      parameterValues: [rowCount, stride, ...(kernelProps.parameterValues ?? [])]
    });

  const workgroupSize = 64;
  const taskLayout = getBoundedDispatchLayout(
    id,
    rowCount * workgroupSize,
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  for (const [name, words] of [
    ['partitions', (maximumClassCount + 1) * stride],
    ['tasks', rowCount * 8],
    ['prefix moments', rowCount * 5]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      words * 4,
      'reduce rows or maximumClassCount'
    );
  }
  const make = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const keys = make('keys', rowCount);
  const sortedKeys = make('sorted-keys', rowCount);
  const rows = make('rows', rowCount);
  const sortedRows = make('sorted-rows', rowCount);
  // At the 2^18-row cap, 20-bit positions give sum < 2^38 and squares < 2^58.
  // The recentered intermediate terms remain below 2^59, inside the signed wide-cost helpers.
  const moments = make('moments', rowCount * 5);
  const prefix = make('prefix', rowCount * 5);
  const partitions = make('partitions', (maximumClassCount + 1) * stride);
  const scores = createTransientView(graph, `${id}-scores`, 'float32', stride * 2);
  // Two breadth-first task lists, each task [first endpoint, last endpoint, low split, high split].
  const tasks = make('tasks', rowCount * 8);
  // Layer, level, dispatched task count, finite rows, distinct numeric values.
  const cursor = make('cursor', 5);
  const binding = (
    name: string,
    view: GraphDataView,
    type: WGSLKernelBinding['type'] = 'u32',
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({name, view, type, access});
  const active = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] == ${GPU_CLASS_BREAKS_METHOD_CODES['natural-breaks']}u`;
  const finiteCount = `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u]`;
  const declarations = `${COLUMN_ORDERED_KEY_WGSL}\n${CLASS_BREAKS_WIDE_WGSL}\nfn getReferenceRowCount() -> u32 { return getChunkParameter(36u); }\nfn getReferenceStride() -> u32 { return getChunkParameter(37u); }`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-initialize`,
      operation: 'GPUClassBreaks',
      variant: 'monotone-initialize',
      invocationCount: Math.max(scores.length, 5),
      bindings: [
        binding('scores', scores, 'f32', 'read_write'),
        binding('cursor', cursor, 'u32', 'read_write')
      ],
      body: `if (index < (arrayLength(&scores) - scoresOffset)) { scores[scoresOffset + index] = select(1e30, 0.0, index == 0u); }
  if (index < 5u) { cursor[cursorOffset + index] = 0u; }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-keys`,
      operation: 'GPUClassBreaks',
      variant: 'monotone-keys',
      invocationCount: rowCount,
      bindings: [
        binding('values', values, 'f32'),
        ...(mask ? [binding('rowMask', mask)] : []),
        binding('keys', keys, 'u32', 'read_write'),
        binding('rows', rows, 'u32', 'read_write')
      ],
      declarations: COLUMN_ORDERED_KEY_WGSL,
      body: `let value = values[valuesOffset + index];
  let include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u ${mask ? '&& rowMask[rowMaskOffset + index] != 0u' : ''};
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-sort`,
      keys,
      values: rows,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      algorithm: 'radix'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-quantize`,
      operation: 'GPUClassBreaks',
      variant: 'monotone-quantize',
      invocationCount: rowCount,
      bindings: [
        binding('sortedKeys', sortedKeys),
        binding('extremes', extremes),
        binding('moments', moments, 'u32', 'read_write'),
        binding('cursor', cursor, 'atomic<u32>', 'read_write')
      ],
      declarations: `${declarations}\n${NATURAL_BREAKS_NORMALIZATION_WGSL}`,
      body: `let count = ${finiteCount};
  var position = 0u;
  var distinct = false;
  if (index < count) {
    let minimum = decodeOrderedKey(sortedKeys[sortedKeysOffset]);
    let maximum = decodeOrderedKey(sortedKeys[sortedKeysOffset + count - 1u]);
    let magnitude = max(abs(minimum), abs(maximum));
    let exponentScale = bitcast<f32>(bitcast<u32>(magnitude) & 0x7f800000u);
    let scale = select(select(1.0, magnitude, magnitude > 0.0), exponentScale, exponentScale > 0.0);
    let lower = getNormalizedValue(minimum, scale);
    let span = getNormalizedValue(maximum, scale) - lower;
    let value = decodeOrderedKey(sortedKeys[sortedKeysOffset + index]);
    if (span > 0.0) {
      position = u32(round(clamp((getNormalizedValue(value, scale) - lower) / span, 0.0, 1.0) * 1048575.0));
    }
    // Numeric equality makes negative and positive zero one value, even though their keys differ.
    distinct = index == 0u;
    if (index > 0u) { distinct = value != decodeOrderedKey(sortedKeys[sortedKeysOffset + index - 1u]); }
    if (distinct) { atomicAdd(&cursor[cursorOffset + 4u], 1u); }
  }
  let square = multiplyWide(position, position);
  moments[momentsOffset + index] = position;
  moments[momentsOffset + getReferenceRowCount() + index] = 0u;
  moments[momentsOffset + 2u * getReferenceRowCount() + index] = square.x;
  moments[momentsOffset + 3u * getReferenceRowCount() + index] = square.y;
  moments[momentsOffset + 4u * getReferenceRowCount() + index] = u32(distinct);`
    })
  ];
  const momentView = (view: GraphDataView<'uint32'>, field: number) =>
    graph.createDataView(view.buffer, {
      format: 'uint32',
      length: rowCount,
      byteOffset: view.byteOffset + field * rowCount * 4
    });
  const prefixLow = make('prefix-low', rowCount);
  const prefixHigh = make('prefix-high', rowCount);
  const taskDispatch = createTransientView(
    graph,
    `${id}-task-dispatch`,
    'uint32',
    3,
    Buffer.STORAGE | Buffer.INDIRECT
  );
  for (const field of [0, 2]) {
    nodes.push(
      ...new GPUScanUint64({
        id: `${id}-prefix-${field}`,
        inputLow: momentView(moments, field),
        inputHigh: momentView(moments, field + 1),
        outputLow: prefixLow,
        outputHigh: prefixHigh
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-pack-prefix-${field}`,
        operation: 'GPUClassBreaks',
        variant: 'monotone-pack-prefix',
        invocationCount: rowCount,
        bindings: [
          binding('low', prefixLow),
          binding('high', prefixHigh),
          binding(
            'prefix',
            graph.createDataView(prefix.buffer, {
              format: 'uint32',
              byteOffset: prefix.byteOffset + field * rowCount * 4,
              length: 2 * rowCount
            }),
            'u32',
            'read_write'
          )
        ],
        body: `prefix[prefixOffset + index] = low[lowOffset + index];
  prefix[prefixOffset + getChunkParameter(36u) + index] = high[highOffset + index];`
      })
    );
  }
  nodes.push(
    ...new GPUScan({
      id: `${id}-distinct-prefix`,
      input: momentView(moments, 4),
      output: momentView(prefix, 4),
      mode: 'inclusive'
    }).getCommandNodes(graph)
  );
  const maximumLevels = Math.floor(Math.log2(rowCount)) + 1;
  for (let layer = 0; layer < maximumClassCount; layer++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-layer-${layer}`,
        operation: 'GPUClassBreaks',
        variant: 'monotone-layer',
        invocationCount: 1,
        bindings: [
          binding('state', state),
          binding('extremes', extremes),
          binding('cursor', cursor, 'u32', 'read_write'),
          binding('tasks', tasks, 'u32', 'read_write'),
          binding('scores', scores, 'f32', 'read_write')
        ],
        declarations,
        body: `let layer = cursor[cursorOffset] + 1u;
  let count = ${finiteCount};
  cursor[cursorOffset] = layer;
  cursor[cursorOffset + 1u] = 0u;
  cursor[cursorOffset + 2u] = select(0u, 1u, ${active} && count > 0u && layer <= min(cursor[cursorOffset + 4u], state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]));
  cursor[cursorOffset + 3u] = count;
  scores[scoresOffset + (layer & 1u) * getReferenceStride()] = 1e30;
  tasks[tasksOffset] = 1u;
  tasks[tasksOffset + 1u] = count;
  tasks[tasksOffset + 2u] = layer - 1u;
  tasks[tasksOffset + 3u] = select(0u, count - 1u, count > 0u);`
      })
    );
    for (let level = 0; level < maximumLevels; level++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-dispatch-${layer}-${level}`,
          operation: 'GPUClassBreaks',
          invocationCount: 1,
          parameterValues: [taskLayout.x, taskLayout.y],
          bindings: [
            binding('cursor', cursor),
            binding('dispatch', taskDispatch, 'u32', 'read_write')
          ],
          body: `let taskCount = min(cursor[cursorOffset + 2u], getChunkParameter(36u));
  let maximumX = getChunkParameter(38u);
  let maximumY = getChunkParameter(39u);
  var x = min(taskCount, maximumX);
  var y = 1u;
  var z = 1u;
  if (taskCount > maximumX) {
    x = maximumX;
    y = min((taskCount + maximumX - 1u) / maximumX, maximumY);
  }
  if (taskCount > maximumX * maximumY) {
    y = maximumY;
    z = (taskCount + maximumX * maximumY - 1u) / (maximumX * maximumY);
  }
  dispatch[dispatchOffset] = x;
  dispatch[dispatchOffset + 1u] = y;
  dispatch[dispatchOffset + 2u] = z;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-search-${layer}-${level}`,
          operation: 'GPUClassBreaks',
          variant: 'monotone-search',
          workgroupCollective: true,
          invocationCount: rowCount * workgroupSize,
          workgroupSize,
          ...getClassBreaksIndirectGate<Parameters>(
            `${id}-search-${layer}-${level}`,
            taskDispatch,
            0
          ),
          bindings: [
            binding('sortedKeys', sortedKeys),
            binding('prefix', prefix),
            binding('cursor', cursor),
            binding('tasks', tasks, 'u32', 'read_write'),
            binding('scores', scores, 'f32', 'read_write'),
            binding('partitions', partitions, 'u32', 'read_write')
          ],
          declarations: `${declarations}
var<workgroup> candidateCosts: array<f32, ${workgroupSize}>;
var<workgroup> candidateSplits: array<u32, ${workgroupSize}>;
var<workgroup> requiresFallback: u32;
fn readPrefix(field: u32, end: u32) -> vec2<u32> {
  if (end == 0u) { return vec2<u32>(0u); }
  return vec2<u32>(prefix[prefixOffset + field * getReferenceRowCount() + end - 1u], prefix[prefixOffset + (field + 1u) * getReferenceRowCount() + end - 1u]);
}
fn rangeCost(first: u32, end: u32) -> f32 {
  return getClassCost(end - first, subtractWide(readPrefix(0u, end), readPrefix(0u, first)), subtractWide(readPrefix(2u, end), readPrefix(2u, first)));
}`,
          body: `let task = index / ${workgroupSize}u;
  let local = localInvocationIndex;
  let layer = cursor[cursorOffset];
  let level = cursor[cursorOffset + 1u];
  let taskCount = cursor[cursorOffset + 2u];
  let base = tasksOffset + (level & 1u) * getReferenceRowCount() * 4u + min(task, getReferenceRowCount() - 1u) * 4u;
  let first = tasks[base]; let last = tasks[base + 1u];
  let lower = tasks[base + 2u]; let upper = tasks[base + 3u];
  let end = first + (last - min(first, last)) / 2u;
  var best = 1e30; var split = lower;
  let feasible = task < taskCount && first <= last && prefix[prefixOffset + 4u * getReferenceRowCount() + min(end - 1u, getReferenceRowCount() - 1u)] >= layer;
  if (feasible) {
    let high = min(max(upper, lower), end - 1u);
    for (var start = lower + local; start <= high; start += ${workgroupSize}u) {
      if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
      let candidate = scores[scoresOffset + ((layer - 1u) & 1u) * getReferenceStride() + start] + rangeCost(start, end);
      if (candidate < best || (candidate == best && start < split)) { best = candidate; split = start; }
    }
  }
  candidateCosts[local] = best; candidateSplits[local] = split;
  workgroupBarrier();
  for (var width = ${workgroupSize / 2}u; width > 0u; width >>= 1u) {
    if (local < width) {
      let otherCost = candidateCosts[local + width]; let otherSplit = candidateSplits[local + width];
      if (otherCost < candidateCosts[local] || (otherCost == candidateCosts[local] && otherSplit < candidateSplits[local])) {
        candidateCosts[local] = otherCost; candidateSplits[local] = otherSplit;
      }
    }
    workgroupBarrier();
  }
  if (local == 0u) { requiresFallback = u32(feasible && candidateCosts[0] >= 1e30); }
  let fallback = workgroupUniformLoad(&requiresFallback);
  if (fallback != 0u) {
    // Rounding can make the inherited search interval miss all valid boundaries. Recover a
    // valid partition by searching this endpoint exhaustively; this is not an optimality proof.
    best = 1e30; split = layer - 1u;
    for (var start = layer - 1u + local; start < end; start += ${workgroupSize}u) {
      if (start > 0u && decodeOrderedKey(sortedKeys[sortedKeysOffset + start - 1u]) == decodeOrderedKey(sortedKeys[sortedKeysOffset + start])) { continue; }
      let candidate = scores[scoresOffset + ((layer - 1u) & 1u) * getReferenceStride() + start] + rangeCost(start, end);
      if (candidate < best || (candidate == best && start < split)) { best = candidate; split = start; }
    }
    candidateCosts[local] = best; candidateSplits[local] = split;
    workgroupBarrier();
    for (var width = ${workgroupSize / 2}u; width > 0u; width >>= 1u) {
      if (local < width) {
        let otherCost = candidateCosts[local + width]; let otherSplit = candidateSplits[local + width];
        if (otherCost < candidateCosts[local] || (otherCost == candidateCosts[local] && otherSplit < candidateSplits[local])) {
          candidateCosts[local] = otherCost; candidateSplits[local] = otherSplit;
        }
      }
      workgroupBarrier();
    }
  }
  if (local == 0u && task < taskCount) {
    let selected = candidateSplits[0];
    if (first <= last) {
      scores[scoresOffset + (layer & 1u) * getReferenceStride() + end] = candidateCosts[0];
      partitions[partitionsOffset + layer * getReferenceStride() + end] = selected;
    }
    if (taskCount * 2u <= getReferenceRowCount()) {
      let next = tasksOffset + ((level + 1u) & 1u) * getReferenceRowCount() * 4u + task * 8u;
      tasks[next] = first; tasks[next + 1u] = select(0u, end - 1u, first <= last);
      tasks[next + 2u] = lower; tasks[next + 3u] = max(lower, selected);
      tasks[next + 4u] = select(1u, end + 1u, first <= last); tasks[next + 5u] = select(0u, last, first <= last);
      tasks[next + 6u] = max(lower, selected); tasks[next + 7u] = max(upper, selected);
    }
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-advance-${layer}-${level}`,
          operation: 'GPUClassBreaks',
          variant: 'monotone-advance',
          invocationCount: 1,
          bindings: [binding('cursor', cursor, 'u32', 'read_write')],
          body: `cursor[cursorOffset + 1u]++;
  let count = cursor[cursorOffset + 2u] * 2u;
  cursor[cursorOffset + 2u] = select(0u, count, count <= cursor[cursorOffset + 3u]);`
        })
      );
    }
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-backtrack`,
      operation: 'GPUClassBreaks',
      variant: 'monotone-backtrack',
      invocationCount: 1,
      bindings: [
        binding('state', state),
        binding('sortedKeys', sortedKeys),
        binding('cursor', cursor),
        binding('partitions', partitions),
        binding('edges', edges, 'f32', 'read_write'),
        binding('edgeState', edgeState, 'u32', 'read_write')
      ],
      declarations,
      body: `if (!(${active})) { return; }
  var end = cursor[cursorOffset + 3u];
  let classes = min(cursor[cursorOffset + 4u], state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]);
  edgeState[edgeStateOffset] = classes;
  for (var layer = classes; layer > 1u; layer--) {
    let split = partitions[partitionsOffset + layer * getReferenceStride() + end];
    edges[edgesOffset + layer - 1u] = decodeOrderedKey(sortedKeys[sortedKeysOffset + split]);
    end = split;
  }`
    })
  );
  return nodes;
}
