// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Every scan workgroup visits this many facilities or block totals. @internal */
export const FACILITY_COMPACTION_WORKGROUP_SIZE = 256;

/** Length of each level, from facility flags through the final single-workgroup scan. */
export function getFacilityCompactionLevelCounts(facilityCount: number): number[] {
  const counts = [facilityCount];
  while (counts[counts.length - 1] > FACILITY_COMPACTION_WORKGROUP_SIZE) {
    counts.push(Math.ceil(counts[counts.length - 1] / FACILITY_COMPACTION_WORKGROUP_SIZE));
  }
  return counts;
}

/** Allocate once; reuse ranks and dispatch triples across every swap round. @internal */
export function createFacilitySelectedCompaction<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    facilityCount: number;
    slots: number;
    selected: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
    swapState: GraphDataView<'uint32'>;
    swapDispatch: GraphDataView<'uint32'>;
    selectedList: GraphDataView<'uint32'>;
    slotOfFacility: GraphDataView<'uint32'>;
  }
): (iteration: number) => GPUCommandNode<Parameters>[] {
  const operation = 'GPULocationAllocation';
  const counts = getFacilityCompactionLevelCounts(props.facilityCount);
  const prefixes = counts.map((count, level) =>
    createTransientView(graph, `${props.id}-list-prefix-${level}`, 'uint32', count)
  );
  const totals = counts.map((count, level) =>
    createTransientView(
      graph,
      `${props.id}-list-totals-${level}`,
      'uint32',
      Math.ceil(count / FACILITY_COMPACTION_WORKGROUP_SIZE)
    )
  );
  const gate = createWGSLDispatchGate(graph, {
    id: `${props.id}-list-dispatch`,
    operation,
    shapes: counts.map(count => [count, FACILITY_COMPACTION_WORKGROUP_SIZE] as const)
  });
  const bind = (name: string, view: GraphDataView<'uint32'>, write = false): WGSLKernelBinding => ({
    name,
    view,
    type: 'u32',
    access: write ? 'read_write' : 'read'
  });
  const activeBindings = [
    bind('selected', props.selected),
    bind('status', props.status),
    bind('swapState', props.swapState)
  ];
  const active = 'swapState[swapStateOffset] == 0u && status[statusOffset + 3u] == 0u';
  return iteration => {
    const id = `${props.id}-swap-${iteration}-list`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-dispatch`,
        operation,
        variant: 'swap-list-dispatch',
        invocationCount: 1,
        bindings: [bind('swapDispatch', props.swapDispatch), gate.binding],
        // Keep the original swap gate's layout and its baselined init/apply kernels unchanged.
        // This ungated dispatch also disarms all scan levels after the swap phase converges.
        body: `if (swapDispatch[swapDispatchOffset] != 0u) { ${gate.armWGSL} }
else { ${gate.disarmWGSL} }`
      })
    ];
    for (let level = 0; level < counts.length; level++) {
      const count = counts[level];
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-scan-${level}`,
          operation,
          variant: 'swap-list-scan',
          invocationCount: count,
          workgroupSize: FACILITY_COMPACTION_WORKGROUP_SIZE,
          guardIndex: false,
          ...gate.getGate(level),
          bindings: [
            ...(level === 0
              ? [...activeBindings, bind('selectedList', props.selectedList, true)]
              : [bind('input', totals[level - 1])]),
            bind('prefix', prefixes[level], true),
            bind('totals', totals[level], true)
          ],
          declarations: `var<workgroup> scan: array<u32, ${FACILITY_COMPACTION_WORKGROUP_SIZE}>;`,
          // Padded lanes must participate in every barrier, including the final partial block.
          body: `var value = 0u;
if (index < ${count}u) {
  ${
    level === 0
      ? `value = select(0u, 1u, ${active} && selected[selectedOffset + index] != 0u);
  if (index < ${props.slots}u) { selectedList[selectedListOffset + index] = 0xffffffffu; }`
      : 'value = input[inputOffset + index];'
  }
}
scan[localInvocationIndex] = value;
workgroupBarrier();
for (var stride = 1u; stride < ${FACILITY_COMPACTION_WORKGROUP_SIZE}u; stride <<= 1u) {
  var previous = 0u;
  if (localInvocationIndex >= stride) { previous = scan[localInvocationIndex - stride]; }
  workgroupBarrier();
  scan[localInvocationIndex] += previous;
  workgroupBarrier();
}
if (index < ${count}u) { prefix[prefixOffset + index] = scan[localInvocationIndex]; }
if (localInvocationIndex == ${FACILITY_COMPACTION_WORKGROUP_SIZE - 1}u && workgroupIndex < ${Math.ceil(count / FACILITY_COMPACTION_WORKGROUP_SIZE)}u) {
  totals[totalsOffset + workgroupIndex] = scan[localInvocationIndex];
}`
        })
      );
    }
    for (let level = counts.length - 2; level >= 0; level--) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-offset-${level}`,
          operation,
          variant: 'swap-list-offset',
          invocationCount: counts[level],
          workgroupSize: FACILITY_COMPACTION_WORKGROUP_SIZE,
          ...gate.getGate(level),
          bindings: [bind('prefix', prefixes[level], true), bind('parent', prefixes[level + 1])],
          body: `let blockIndex = index / ${FACILITY_COMPACTION_WORKGROUP_SIZE}u;
if (blockIndex > 0u) { prefix[prefixOffset + index] += parent[parentOffset + blockIndex - 1u]; }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-scatter`,
        operation,
        variant: 'swap-list-scatter',
        invocationCount: props.facilityCount,
        workgroupSize: FACILITY_COMPACTION_WORKGROUP_SIZE,
        ...gate.getGate(0),
        bindings: [
          ...activeBindings,
          bind('prefix', prefixes[0]),
          bind('selectedList', props.selectedList, true),
          bind('slotOfFacility', props.slotOfFacility, true)
        ],
        body: `var slot = 0xffffffffu;
if (${active} && selected[selectedOffset + index] != 0u) {
  let rank = prefix[prefixOffset + index] - 1u;
  if (rank < ${props.slots}u) { slot = rank; selectedList[selectedListOffset + slot] = index; }
}
slotOfFacility[slotOfFacilityOffset + index] = slot;`
      })
    );
    return nodes;
  };
}
