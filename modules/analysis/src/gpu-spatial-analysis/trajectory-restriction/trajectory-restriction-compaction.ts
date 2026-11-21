// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUTrajectoryRestrictionResolvedProps} from './gpu-trajectory-restriction';

/** Stable source-order compaction and interval union, including the legacy track-cursor rule. @internal */
export function createRestrictionCompactionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryRestrictionResolvedProps
): GPUCommandNode<Parameters>[] {
  const output = props.output;
  if (
    !output.compactIntervals ||
    !output.compactPositions ||
    !output.compactSources ||
    !output.compactStatus ||
    !output.compactSummary
  )
    return [];
  const operation = 'GPUTrajectoryRestriction';
  const slotCount = output.intervals.length;
  const length = Math.max(slotCount, 1);
  const capacity = output.compactIntervals.length;
  const slots = props.intervalsPerSegment ?? 1;
  const trackBoundaryCount = Math.max(props.offsets.length - 2, 0);
  const boundaryLength = Math.max(trackBoundaryCount, 1);
  const view = (name: string, size = length) =>
    createTransientView(graph, `${id}-compact-${name}`, 'uint32', size);
  const flags = view('flags');
  const prefixes = view('prefixes');
  const total = view('total', 1);
  const upstreamLoss = view('upstream-loss', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createFillNode(graph, {
      id: `${id}-compact-loss-clear`,
      operation,
      view: upstreamLoss,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-compact-flags`,
      operation,
      invocationCount: length,
      bindings: [
        {name: 'status', view: output.status, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
        {name: 'loss', view: upstreamLoss, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `var present = 0u;
  if (index < ${slotCount}u) {
    let statusWord = status[statusOffset + index];
    present = select(0u, 1u, (statusWord & 3u) != 0u);
    if ((statusWord & 256u) != 0u) { atomicAdd(&loss[lossOffset], 1u); }
  }
  flags[flagsOffset + index] = present;`
    }),
    ...new GPUScan({
      id: `${id}-compact-slot-scan`,
      input: flags,
      output: prefixes,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-compact-total`,
      operation,
      invocationCount: 1,
      bindings: [
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'prefixes', view: prefixes, type: 'u32', access: 'read'},
        {name: 'total', view: total, type: 'u32', access: 'read_write'}
      ],
      body: `total[totalOffset] = prefixes[prefixesOffset + ${length - 1}u] + flags[flagsOffset + ${length - 1}u];`
    })
  ];
  if (!props.mergeIntervals) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-compact-clear`,
        operation,
        invocationCount: capacity,
        bindings: [
          {name: 'intervals', view: output.compactIntervals, type: 'f32', access: 'read_write'},
          {name: 'positions', view: output.compactPositions, type: 'f32', access: 'read_write'},
          {name: 'sources', view: output.compactSources, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.compactStatus, type: 'u32', access: 'read_write'}
        ],
        body: `var bits = 0x7fc00000u; let missing = bitcast<f32>(bits);
  intervals[intervalsOffset + 2u * index] = missing; intervals[intervalsOffset + 2u * index + 1u] = missing;
  for (var component = 0u; component < 4u; component++) { positions[positionsOffset + 4u * index + component] = missing; }
  sources[sourcesOffset + 2u * index] = 0xffffffffu; sources[sourcesOffset + 2u * index + 1u] = 0xffffffffu;
  status[statusOffset + index] = 0u;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-compact-scatter`,
        operation,
        invocationCount: slotCount,
        bindings: [
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'prefixes', view: prefixes, type: 'u32', access: 'read'},
          {name: 'intervals', view: output.intervals, type: 'f32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          {name: 'result', view: output.compactIntervals, type: 'f32', access: 'read_write'},
          {name: 'sources', view: output.compactSources, type: 'u32', access: 'read_write'},
          {name: 'resultStatus', view: output.compactStatus, type: 'u32', access: 'read_write'}
        ],
        body: `let destination = prefixes[prefixesOffset + index];
  if (flags[flagsOffset + index] == 0u || destination >= ${capacity}u) { return; }
  result[resultOffset + 2u * destination] = intervals[intervalsOffset + 2u * index];
  result[resultOffset + 2u * destination + 1u] = intervals[intervalsOffset + 2u * index + 1u];
  sources[sourcesOffset + 2u * destination] = index / ${slots}u; sources[sourcesOffset + 2u * destination + 1u] = index / ${slots}u;
  resultStatus[resultStatusOffset + destination] = status[statusOffset + index];`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-compact-position-scatter`,
        operation,
        invocationCount: slotCount,
        bindings: [
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'prefixes', view: prefixes, type: 'u32', access: 'read'},
          {name: 'positions', view: output.positions, type: 'f32', access: 'read'},
          {name: 'result', view: output.compactPositions, type: 'f32', access: 'read_write'}
        ],
        body: `let destination = prefixes[prefixesOffset + index];
  if (flags[flagsOffset + index] == 0u || destination >= ${capacity}u) { return; }
  for (var component = 0u; component < 4u; component++) { result[resultOffset + 4u * destination + component] = positions[positionsOffset + 4u * index + component]; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-compact-summary`,
        operation,
        invocationCount: 1,
        bindings: [
          {name: 'total', view: total, type: 'u32', access: 'read'},
          {name: 'loss', view: upstreamLoss, type: 'u32', access: 'read'},
          {name: 'summary', view: output.compactSummary, type: 'u32', access: 'read_write'}
        ],
        body: `let required = total[totalOffset]; let stored = min(required, ${capacity}u);
  summary[summaryOffset] = required; summary[summaryOffset + 1u] = stored; summary[summaryOffset + 2u] = required - stored + loss[lossOffset];`
      })
    );
    return nodes;
  }
  const sourceSlots = view('source-slots');
  const tracks = view('tracks');
  const groupStarts = view('group-starts');
  const groupLabels = view('group-labels');
  const boundaryBuffers = [
    view('boundaries-a', boundaryLength),
    view('boundaries-b', boundaryLength)
  ];
  const endBuffers = [view('ends-a'), view('ends-b')];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-compact-track-boundaries`,
      operation,
      invocationCount: boundaryLength,
      bindings: [
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'boundaries', view: boundaryBuffers[0], type: 'u32', access: 'read_write'}
      ],
      body: `var boundary = 0xffffffffu;
  if (index < ${trackBoundaryCount}u) { boundary = offsets[offsetsOffset + index + 1u]; }
  boundaries[boundariesOffset + index] = boundary;`
    })
  );
  // The old monotonically advancing cursor crosses boundary t only after crossing 1..t-1.
  // Prefix maxima preserve that rule even for accepted decreasing/reversed track offsets.
  let boundaries = boundaryBuffers[0];
  for (let distance = 1, pass = 0; distance < boundaryLength; distance *= 2, pass++) {
    const next = boundaryBuffers[(pass + 1) % 2];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-compact-track-max-${pass}`,
        operation,
        invocationCount: boundaryLength,
        bindings: [
          {name: 'previous', view: boundaries, type: 'u32', access: 'read'},
          {name: 'next', view: next, type: 'u32', access: 'read_write'}
        ],
        body: `var boundary = previous[previousOffset + index];
  if (index >= ${distance}u) { boundary = max(boundary, previous[previousOffset + index - ${distance}u]); }
  next[nextOffset + index] = boundary;`
      })
    );
    boundaries = next;
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-compact-source-tracks`,
      operation,
      invocationCount: length,
      bindings: [
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'prefixes', view: prefixes, type: 'u32', access: 'read'},
        {name: 'boundaries', view: boundaries, type: 'u32', access: 'read'},
        {name: 'sourceSlots', view: sourceSlots, type: 'u32', access: 'read_write'},
        {name: 'tracks', view: tracks, type: 'u32', access: 'read_write'},
        {name: 'endSlots', view: endBuffers[0], type: 'u32', access: 'read_write'}
      ],
      body: `if (flags[flagsOffset + index] == 0u) { return; }
  let sourceRow = index / ${slots}u;
  var low = 0u; var high = ${trackBoundaryCount}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (boundaries[boundariesOffset + middle] <= sourceRow) { low = middle + 1u; } else { high = middle; }
  }
  let row = prefixes[prefixesOffset + index];
  sourceSlots[sourceSlotsOffset + row] = index;
  tracks[tracksOffset + row] = low;
  endSlots[endSlotsOffset + row] = index;`
    })
  );
  // Segmented inclusive prefix maximum of interval ends, choosing the latest slot on ties.
  // This also preserves the streaming union for non-monotone times in reversed track ranges.
  let endSlots = endBuffers[0];
  if (props.mergeIntervals) {
    for (let distance = 1, pass = 0; distance < length; distance *= 2, pass++) {
      const next = endBuffers[(pass + 1) % 2];
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-compact-end-max-${pass}`,
          operation,
          invocationCount: length,
          bindings: [
            {name: 'previous', view: endSlots, type: 'u32', access: 'read'},
            {name: 'tracks', view: tracks, type: 'u32', access: 'read'},
            {name: 'intervals', view: output.intervals, type: 'f32', access: 'read'},
            {name: 'total', view: total, type: 'u32', access: 'read'},
            {name: 'next', view: next, type: 'u32', access: 'read_write'}
          ],
          body: `if (index >= total[totalOffset]) { next[nextOffset + index] = 0u; return; }
  var last = previous[previousOffset + index];
  if (index >= ${distance}u && tracks[tracksOffset + index] == tracks[tracksOffset + index - ${distance}u]) {
    let earlier = previous[previousOffset + index - ${distance}u];
    if (intervals[intervalsOffset + 2u * earlier + 1u] > intervals[intervalsOffset + 2u * last + 1u]) { last = earlier; }
  }
  next[nextOffset + index] = last;`
        })
      );
      endSlots = next;
    }
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-compact-group-starts`,
      operation,
      invocationCount: length,
      bindings: [
        ...(props.mergeIntervals
          ? [{name: 'sourceSlots', view: sourceSlots, type: 'u32', access: 'read'} as const]
          : []),
        ...(props.mergeIntervals
          ? [{name: 'endSlots', view: endSlots, type: 'u32', access: 'read'} as const]
          : []),
        ...(props.mergeIntervals
          ? [
              {name: 'tracks', view: tracks, type: 'u32', access: 'read'} as const,
              {name: 'intervals', view: output.intervals, type: 'f32', access: 'read'} as const,
              {name: 'status', view: output.status, type: 'u32', access: 'read'} as const
            ]
          : []),
        {name: 'total', view: total, type: 'u32', access: 'read'},
        {name: 'starts', view: groupStarts, type: 'u32', access: 'read_write'}
      ],
      body: `var beginsGroup = 0u;
  if (index < total[totalOffset]) {
    beginsGroup = 1u;
    ${
      props.mergeIntervals
        ? `if (index > 0u && tracks[tracksOffset + index] == tracks[tracksOffset + index - 1u]) {
      let current = sourceSlots[sourceSlotsOffset + index]; let previous = endSlots[endSlotsOffset + index - 1u];
      let begin = intervals[intervalsOffset + 2u * current]; let finish = intervals[intervalsOffset + 2u * previous + 1u];
      let closedTouch = (status[statusOffset + previous] & 128u) == 0u || (status[statusOffset + current] & 64u) == 0u;
      beginsGroup = select(1u, 0u, begin < finish || (begin == finish && closedTouch));
    }`
        : ''
    }
  }
  starts[startsOffset + index] = beginsGroup;`
    }),
    ...new GPUScan({
      id: `${id}-compact-group-scan`,
      input: groupStarts,
      output: groupLabels,
      mode: 'inclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-compact-publish`,
      operation,
      invocationCount: capacity,
      bindings: [
        {name: 'intervals', view: output.intervals, type: 'f32', access: 'read'},
        {name: 'sourceSlots', view: sourceSlots, type: 'u32', access: 'read'},
        ...(props.mergeIntervals
          ? [{name: 'endSlots', view: endSlots, type: 'u32', access: 'read'} as const]
          : []),
        {name: 'total', view: total, type: 'u32', access: 'read'},
        {name: 'labels', view: groupLabels, type: 'u32', access: 'read'},
        {name: 'result', view: output.compactIntervals, type: 'f32', access: 'read_write'},
        {name: 'sources', view: output.compactSources, type: 'u32', access: 'read_write'}
      ],
      declarations: `fn groupStart(label: u32) -> u32 {
  var low = 0u; var high = ${length}u;
  while (low < high) { let middle = low + (high - low) / 2u; if (labels[labelsOffset + middle] < label) { low = middle + 1u; } else { high = middle; } }
  return low;
}`,
      body: `if (index >= labels[labelsOffset + ${length - 1}u]) {
    var bits = 0x7fc00000u; let missing = bitcast<f32>(bits);
    result[resultOffset + 2u * index] = missing; result[resultOffset + 2u * index + 1u] = missing;
    sources[sourcesOffset + 2u * index] = 0xffffffffu; sources[sourcesOffset + 2u * index + 1u] = 0xffffffffu;
    return;
  }
  let firstRow = groupStart(index + 1u);
  let lastRow = min(groupStart(index + 2u), total[totalOffset]) - 1u;
  let first = sourceSlots[sourceSlotsOffset + firstRow];
  let last = ${props.mergeIntervals ? 'endSlots[endSlotsOffset + lastRow]' : 'sourceSlots[sourceSlotsOffset + firstRow]'};
  let begin = intervals[intervalsOffset + 2u * first]; let finish = intervals[intervalsOffset + 2u * last + 1u];
  result[resultOffset + 2u * index] = begin; result[resultOffset + 2u * index + 1u] = finish;
  sources[sourcesOffset + 2u * index] = first; sources[sourcesOffset + 2u * index + 1u] = last;
`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-compact-summary`,
      operation,
      invocationCount: 1,
      bindings: [
        {name: 'labels', view: groupLabels, type: 'u32', access: 'read'},
        {name: 'loss', view: upstreamLoss, type: 'u32', access: 'read'},
        {name: 'summary', view: output.compactSummary, type: 'u32', access: 'read_write'}
      ],
      body: `let required = labels[labelsOffset + ${length - 1}u]; let stored = min(required, ${capacity}u);
  summary[summaryOffset] = required; summary[summaryOffset + 1u] = stored;
  summary[summaryOffset + 2u] = required - stored + loss[lossOffset];`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-compact-positions`,
      operation,
      invocationCount: capacity,
      bindings: [
        {name: 'positions', view: output.positions, type: 'f32', access: 'read'},
        {name: 'sources', view: output.compactSources, type: 'u32', access: 'read_write'},
        {name: 'result', view: output.compactPositions, type: 'f32', access: 'read_write'},
        {name: 'intervals', view: output.compactIntervals, type: 'f32', access: 'read'},
        {name: 'status', view: output.status, type: 'u32', access: 'read'},
        {name: 'resultStatus', view: output.compactStatus, type: 'u32', access: 'read_write'}
      ],
      body: `let first = sources[sourcesOffset + 2u * index]; let last = sources[sourcesOffset + 2u * index + 1u];
  if (first == 0xffffffffu) { var bits = 0x7fc00000u; for (var component = 0u; component < 4u; component++) { result[resultOffset + 4u * index + component] = bitcast<f32>(bits); } resultStatus[resultStatusOffset + index] = 0u; return; }
  result[resultOffset + 4u * index] = positions[positionsOffset + 4u * first]; result[resultOffset + 4u * index + 1u] = positions[positionsOffset + 4u * first + 1u];
  result[resultOffset + 4u * index + 2u] = positions[positionsOffset + 4u * last + 2u]; result[resultOffset + 4u * index + 3u] = positions[positionsOffset + 4u * last + 3u];
  var flags = status[statusOffset + first];
  if (last != first) { flags = (flags & ~131u) | (status[statusOffset + last] & 128u) | select(1u, 2u, intervals[intervalsOffset + 2u * index] == intervals[intervalsOffset + 2u * index + 1u]); }
  resultStatus[resultStatusOffset + index] = flags;
  sources[sourcesOffset + 2u * index] = first / ${slots}u; sources[sourcesOffset + 2u * index + 1u] = last / ${slots}u;`
    })
  );
  return nodes;
}
