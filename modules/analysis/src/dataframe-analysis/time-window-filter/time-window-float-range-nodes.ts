// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS} from '../temporal-reduction/sorted-range-bounds';
import {FLOAT_TIME_BITS_AT_LEAST_WGSL} from './time-window-classify-node';
import type {SortedTimeWindowRangeProps} from './time-window-sorted-nodes';

/**
 * Two independent integer-lane boundary searches, followed by constant-work publication.
 * Each invocation takes at most SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS = 32 midpoint steps;
 * no full-row scratch or scan is needed. The fingerprinted legacy serial search stays unchanged.
 * @internal
 */
export function createSortedFloatTimeWindowRangeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SortedTimeWindowRangeProps
): GPUCommandNode<Parameters>[] {
  const bindings: WGSLKernelBinding[] = [
    {name: 'parameters', view: props.window, type: 'u32', access: 'read'},
    {name: 'rowHigh', view: props.timestamps, type: 'u32', access: 'read'},
    ...(props.timestampsLow
      ? [{name: 'rowLow', view: props.timestampsLow, type: 'u32' as const, access: 'read' as const}]
      : []),
    {name: 'rangeWords', view: props.range, type: 'u32', access: 'read_write'}
  ];
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-boundaries`,
      operation: 'GPUTimeWindowFilter',
      variant: 'sorted-float-bit-boundaries',
      bindings,
      invocationCount: 2,
      workgroupSize: 2,
      constants: {ROW_COUNT: props.timestamps.length},
      declarations: `${FLOAT_TIME_BITS_AT_LEAST_WGSL}
override ROW_COUNT: u32;
const MAXIMUM_SEARCH_STEPS: u32 = ${SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS}u;`,
      body: `if (index >= 2u) { return; }
  let startHigh = parameters[parametersOffset];
  let startLow = parameters[parametersOffset + 1u];
  let endHigh = parameters[parametersOffset + 2u];
  let endLow = parameters[parametersOffset + 3u];
  if (!isTimeBitsAtLeast(endHigh, endLow, startHigh, startLow)) {
    rangeWords[rangeWordsOffset + index] = 0u;
    return;
  }
  var low = 0u;
  var high = ROW_COUNT;
  for (var searchStep = 0u; searchStep < MAXIMUM_SEARCH_STEPS; searchStep++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u;
    let timeHigh = rowHigh[rowHighOffset + middle];
    let timeLow = ${props.timestampsLow ? 'rowLow[rowLowOffset + middle]' : '0u'};
    let beforeStart = !isTimeBitsAtLeast(timeHigh, timeLow, startHigh, startLow);
    let afterEnd = !isTimeBitsAtLeast(endHigh, endLow, timeHigh, timeLow);
    let moveLow = select(beforeStart, !afterEnd, index == 1u);
    if (moveLow) { low = middle + 1u; } else { high = middle; }
  }
  rangeWords[rangeWordsOffset + index] = low;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: props.id,
      operation: 'GPUTimeWindowFilter',
      variant: 'sorted-float-bit-publish',
      bindings: [
        {name: 'rangeWords', view: props.range, type: 'u32', access: 'read_write'},
        {name: 'total', view: props.total, type: 'u32', access: 'read_write'},
        ...(props.firstInstance
          ? [
              {
                name: 'firstInstance',
                view: props.firstInstance,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: 1,
      body: `let firstRow = rangeWords[rangeWordsOffset];
  let rowCount = rangeWords[rangeWordsOffset + 1u] - firstRow;
  rangeWords[rangeWordsOffset + 1u] = rowCount;
  total[totalOffset] = rowCount;
  ${props.firstInstance ? 'firstInstance[firstInstanceOffset] = firstRow;' : ''}`
    })
  ];
}
