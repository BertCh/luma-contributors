// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Fixed byte-radix histogram size and increasing-digit selection bound. @internal */
export const GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT = 256;

/** Distributed exact top-budget selection of reciprocal pairs in (cost, seeded key, root) order. */
export function getWardMutualSelectionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    targetRegionCount: number;
    rowWorkgroupSize: number;
    gate: WGSLDispatchGate<Parameters>;
    regionCost: GraphDataView<'uint32'>;
    regionKey: GraphDataView<'uint32'>;
    regionOther: GraphDataView<'uint32'>;
    mergeFlag: GraphDataView<'uint32'>;
    summary: GraphDataView<'uint32'>;
    histogram: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, rowCount, gate, rowWorkgroupSize} = props;
  const selectionGate = createWGSLDispatchGate(graph, {
    id: `${id}-selection-dispatch`,
    operation: 'GPUWardRegions',
    shapes: [
      [rowCount, rowWorkgroupSize],
      [1, 2]
    ]
  });
  const summaryBinding = {
    name: 'summary',
    view: props.summary,
    type: 'u32',
    access: 'read'
  } as const;
  const pairBindings = [
    {name: 'regionCost', view: props.regionCost, type: 'u32', access: 'read'},
    {name: 'regionKey', view: props.regionKey, type: 'u32', access: 'read'},
    {name: 'regionOther', view: props.regionOther, type: 'u32', access: 'read'}
  ] as const;
  const pairDeclarations = `const ROW_COUNT: u32 = ${rowCount}u;
fn isMutual(region: u32) -> bool {
  let other = regionOther[regionOtherOffset + region];
  return other > region && other < ROW_COUNT && regionOther[regionOtherOffset + other] == region;
}
fn getWord(region: u32, word: u32) -> u32 {
  if (word == 0u) { return regionCost[regionCostOffset + region]; }
  if (word == 1u) { return regionKey[regionKeyOffset + region]; }
  return region;
}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-selection-reset`,
      operation: 'GPUWardRegions',
      variant: 'mutual-selection-reset',
      invocationCount: GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT,
      workgroupSize: GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT,
      ...gate.getGate(1),
      bindings: [
        {name: 'summary', view: props.summary, type: 'u32', access: 'read_write'},
        {name: 'histogram', view: props.histogram, type: 'u32', access: 'read_write'}
      ],
      body: `histogram[histogramOffset + index] = 0u; if (index < 7u) { summary[summaryOffset + index] = 0u; }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-selection-count`,
      operation: 'GPUWardRegions',
      variant: 'mutual-selection-count',
      invocationCount: rowCount,
      workgroupSize: rowWorkgroupSize,
      ...gate.getGate(0),
      bindings: [
        ...pairBindings,
        {name: 'summary', view: props.summary, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: pairDeclarations,
      body: 'if (isMutual(index)) { atomicAdd(&summary[summaryOffset], 1u); }'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-selection-prepare`,
      operation: 'GPUWardRegions',
      variant: 'mutual-selection-prepare',
      invocationCount: 1,
      bindings: [
        {name: 'mergeDispatch', view: gate.view, type: 'u32', access: 'read'},
        {name: 'summary', view: props.summary, type: 'u32', access: 'read_write'},
        {name: 'state', view: props.state, type: 'u32', access: 'read'},
        {name: 'status', view: props.status, type: 'u32', access: 'read'},
        selectionGate.binding
      ],
      body: `var budget = 0u;
  if (mergeDispatch[mergeDispatchOffset] != 0u && status[statusOffset + 3u] == 0u && state[stateOffset] > ${props.targetRegionCount}u) { budget = state[stateOffset] - ${props.targetRegionCount}u; }
  summary[summaryOffset + 1u] = budget; summary[summaryOffset + 2u] = budget;
  ${selectionGate.disarmWGSL}
  if (budget > 0u && summary[summaryOffset] > budget) { ${selectionGate.armWGSL} }`
    })
  ];
  // Twelve base-256 digits replace 96 full-input scans by a single workgroup. The kernels and
  // scratch are reused for every digit and round; the cursor resides on the GPU.
  for (let digit = 0; digit < 12; digit++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-selection-count-${digit}`,
        operation: 'GPUWardRegions',
        variant: 'mutual-selection-digit-count',
        invocationCount: rowCount,
        workgroupSize: rowWorkgroupSize,
        ...selectionGate.getGate(0),
        bindings: [
          ...pairBindings,
          summaryBinding,
          {name: 'histogram', view: props.histogram, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: pairDeclarations,
        body: `if (!isMutual(index)) { return; }
  let cursor = summary[summaryOffset + 6u]; let word = cursor / 4u; let shift = 24u - (cursor % 4u) * 8u;
  for (var previous = 0u; previous < word; previous++) { if (getWord(index, previous) != summary[summaryOffset + 3u + previous]) { return; } }
  let value = getWord(index, word);
  if (shift < 24u && (value >> (shift + 8u)) != (summary[summaryOffset + 3u + word] >> (shift + 8u))) { return; }
  atomicAdd(&histogram[histogramOffset + ((value >> shift) & 255u)], 1u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-selection-choose-${digit}`,
        operation: 'GPUWardRegions',
        variant: 'mutual-selection-digit-choose',
        invocationCount: 1,
        workgroupSize: 2,
        ...selectionGate.getGate(1),
        bindings: [
          {name: 'summary', view: props.summary, type: 'u32', access: 'read_write'},
          {name: 'histogram', view: props.histogram, type: 'u32', access: 'read_write'}
        ],
        body: `let cursor = summary[summaryOffset + 6u]; let word = cursor / 4u; let shift = 24u - (cursor % 4u) * 8u;
  var remaining = summary[summaryOffset + 2u]; var selected = false;
  for (var digit = 0u; digit < ${GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT}u; digit++) {
    let count = histogram[histogramOffset + digit]; histogram[histogramOffset + digit] = 0u;
    if (!selected) {
      if (remaining > count) { remaining -= count; }
      else { summary[summaryOffset + 3u + word] |= digit << shift; selected = true; }
    }
  }
  summary[summaryOffset + 2u] = remaining; summary[summaryOffset + 6u] = cursor + 1u;`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-select-pairs`,
      operation: 'GPUWardRegions',
      variant: 'mutual-select-distributed-pairs',
      invocationCount: rowCount,
      workgroupSize: rowWorkgroupSize,
      ...gate.getGate(0),
      bindings: [
        ...pairBindings,
        summaryBinding,
        {name: 'mergeFlag', view: props.mergeFlag, type: 'u32', access: 'read_write'}
      ],
      declarations: pairDeclarations,
      body: `let budget = summary[summaryOffset + 1u];
  var accepted = budget > 0u && isMutual(index);
  if (accepted && summary[summaryOffset] > budget) {
    let cost = getWord(index, 0u); let key = getWord(index, 1u);
    let thresholdCost = summary[summaryOffset + 3u]; let thresholdKey = summary[summaryOffset + 4u];
    accepted = cost < thresholdCost || (cost == thresholdCost && (key < thresholdKey || (key == thresholdKey && index <= summary[summaryOffset + 5u])));
  }
  mergeFlag[mergeFlagOffset + index] = u32(accepted);`
    })
  );
  return nodes;
}
