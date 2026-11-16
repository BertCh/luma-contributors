// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {CSRStageAttribute} from './spatial-weights-algebra-wgsl';

/** Saturates every prefix before clamping to capacity, so later rows never overwrite earlier rows. @internal */
export function getFuzzyContiguityOffsetNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    counts: GraphDataView<'uint32'>;
    offsets: GraphDataView<'uint32'>;
    slots: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
  }
): readonly GPUCommandNode<Parameters>[] {
  const starts = createTransientView(graph, `${props.id}-starts`, 'uint32', props.counts.length);
  const total = createTransientView(graph, `${props.id}-total`, 'uint32', 1);
  return [
    ...createSaturatedCountScanNodes(graph, {
      id: `${props.id}-scan`,
      operation: 'GPUFuzzyContiguity',
      counts: props.counts,
      offsets: starts,
      rowCount: props.counts.length,
      total
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-offsets`,
      operation: 'GPUFuzzyContiguity',
      variant: 'offsets',
      bindings: [
        {name: 'starts', view: starts, type: 'u32', access: 'read'},
        {name: 'total', view: total, type: 'u32', access: 'read'},
        {name: 'slots', view: props.slots, type: 'u32', access: 'read'},
        {name: 'outOffsets', view: props.offsets, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
        ...(props.requiredCount
          ? [
              {
                name: 'requiredCount',
                view: props.requiredCount,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: props.counts.length + 1,
      body: `let rowCount = arrayLength(&starts) - startsOffset;
let capacity = arrayLength(&slots) - slotsOffset;
if (index < rowCount) {
  outOffsets[outOffsetsOffset + index] = min(starts[startsOffset + index], capacity);
} else {
  let required = total[totalOffset];
  outOffsets[outOffsetsOffset + rowCount] = min(required, capacity);
  overflow[overflowOffset] = select(0u, 1u, required > capacity);
  ${props.requiredCount ? 'requiredCount[requiredCountOffset] = required;' : ''}
}`
    })
  ];
}

/** Fuzzy contiguity's count/saturated-scan/emit stages; other CSR producers are unchanged. @internal */
export function getFuzzyContiguityCSRNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    offsets: GraphDataView<'uint32'>;
    slots: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    bindings: readonly WGSLKernelBinding[];
    declarations: string;
    visit: (onItem: string) => string;
    attributes: readonly Pick<CSRStageAttribute, 'name' | 'output' | 'type' | 'value'>[];
  }
): readonly GPUCommandNode<Parameters>[] {
  const counts = createTransientView(graph, `${props.id}-counts`, 'uint32', props.rows);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-counts`,
      operation: 'GPUFuzzyContiguity',
      variant: 'counts',
      bindings: [
        ...props.bindings,
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: props.rows,
      declarations: props.declarations,
      body: `var count = 0u;
${props.visit('count++;')}
counts[countsOffset + index] = count;`
    }),
    ...getFuzzyContiguityOffsetNodes(graph, {...props, counts}),
    ...props.attributes.map(attribute =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-emit-${attribute.name}`,
        operation: 'GPUFuzzyContiguity',
        variant: `emit-${attribute.name}`,
        bindings: [
          ...props.bindings,
          {name: 'outOffsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'outValues', view: attribute.output, type: attribute.type, access: 'read_write'}
        ],
        invocationCount: props.rows,
        declarations: props.declarations,
        // Exhausted rows skip exact distance work altogether, including after a saturated prefix.
        body: `var next = outOffsets[outOffsetsOffset + index];
let end = outOffsets[outOffsetsOffset + index + 1u];
if (next < end) {
  ${props.visit(`if (next < end) { outValues[outValuesOffset + next] = ${attribute.value}; next++; }`)}
}`
      })
    )
  ];
}
