// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, type GraphDataView} from './gpu-command-graph';
import {getBoundedWorkgroupDispatchLayout, getBoundedDispatchOverrides} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {
  getViewElementOffset,
  validatePackedUint32View,
  validatePackedView,
  type GPUScalarFormat
} from './graph-data-view-utils';

const WORKGROUP_SIZE = 256;
const SCALAR_FORMATS = ['uint32', 'sint32', 'float32'] as const;

export type GPUSegmentedReductionOperation = 'sum' | 'min' | 'max';

export type GPUSegmentedReductionProps<T extends GPUScalarFormat = GPUScalarFormat> = {
  id?: string;
  input: GraphDataView<T>;
  /** CSR-style element offsets. Length must equal output.length + 1. */
  segmentOffsets: GraphDataView<'uint32'>;
  output: GraphDataView<T>;
  operation: GPUSegmentedReductionOperation;
};

/** Reduces packed scalar rows independently for each CSR-style segment. */
export class GPUSegmentedReduction<T extends GPUScalarFormat = GPUScalarFormat> {
  readonly id: string;
  readonly input: GraphDataView<T>;
  readonly segmentOffsets: GraphDataView<'uint32'>;
  readonly output: GraphDataView<T>;
  readonly operation: GPUSegmentedReductionOperation;

  constructor(props: GPUSegmentedReductionProps<T>) {
    this.id = props.id ?? 'gpu-segmented-reduction';
    this.input = props.input;
    this.segmentOffsets = props.segmentOffsets;
    this.output = props.output;
    this.operation = props.operation;

    validatePackedView(this.input, SCALAR_FORMATS, `${this.id} input`);
    validatePackedUint32View(this.segmentOffsets, `${this.id} segmentOffsets`);
    validatePackedView(this.output, SCALAR_FORMATS, `${this.id} output`);
    if (this.input.format !== this.output.format) {
      throw new Error(`${this.id} input and output formats must match`);
    }
    if (this.segmentOffsets.length !== this.output.length + 1) {
      throw new Error(`${this.id} segmentOffsets length must equal output.length + 1`);
    }
    if (!['sum', 'min', 'max'].includes(this.operation)) {
      throw new Error(`${this.id} operation must be sum, min, or max`);
    }
    if (
      this.output.buffer === this.input.buffer ||
      this.output.buffer === this.segmentOffsets.buffer
    ) {
      throw new Error(`${this.id} output must use a separate buffer`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (const view of [this.input, this.segmentOffsets, this.output]) {
      if (view.buffer.graph !== graph)
        throw new Error(`${this.id} views must belong to target graph`);
    }
    if (this.output.length === 0) return nodes;

    // One workgroup reduces each segment; the segment count is spread over a bounded 3D dispatch.
    const dispatchLayout = getBoundedWorkgroupDispatchLayout(
      'GPUSegmentedReduction',
      this.output.length,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const bindings: GPUKernelBinding[] = [
      {name: 'inputValues', view: this.input, access: 'read'},
      {name: 'segmentOffsets', view: this.segmentOffsets, access: 'read'},
      {name: 'outputValues', view: this.output, access: 'write'}
    ];
    const constants = {
      SEGMENT_COUNT: this.output.length,
      INPUT_OFFSET: getViewElementOffset(this.input),
      SEGMENT_OFFSET: getViewElementOffset(this.segmentOffsets),
      OUTPUT_OFFSET: getViewElementOffset(this.output),
      ...getBoundedDispatchOverrides(dispatchLayout)
    };
    nodes.push(
      createGPUKernelNode(graph, {
        id: this.id,
        operation: 'GPUSegmentedReduction',
        moduleId: 'GPUSegmentedReduction/reduce',
        formula: 'gpu-core.segmented-reduction',
        source: makeShaderSource(this, constants, bindings),
        constants,
        bindings,
        workgroupSize: WORKGROUP_SIZE,
        dispatchLayout,
        // One workgroup per segment; its lanes stride over the segment's rows.
        invocationCount: this.output.length * WORKGROUP_SIZE,
        workload: {
          readByteLength: this.input.length * 4 + this.segmentOffsets.length * 4,
          writeByteLength: this.output.length * 4
        }
      })
    );

    return nodes;
  }
}

function makeShaderSource(
  reduction: GPUSegmentedReduction,
  constants: Readonly<Record<string, number>>,
  bindings: readonly GPUKernelBinding[]
): string {
  const type =
    reduction.input.format === 'uint32'
      ? 'u32'
      : reduction.input.format === 'sint32'
        ? 'i32'
        : 'f32';
  const identity =
    reduction.operation === 'sum'
      ? type === 'f32'
        ? '0.0'
        : '0'
      : reduction.operation === 'min'
        ? type === 'f32'
          ? '3.402823466e+38'
          : type === 'u32'
            ? '0xffffffffu'
            : '2147483647'
        : type === 'f32'
          ? '-3.402823466e+38'
          : type === 'u32'
            ? '0u'
            : '-2147483648';
  const combine =
    reduction.operation === 'sum'
      ? 'a + b'
      : reduction.operation === 'min'
        ? 'min(a, b)'
        : 'max(a, b)';
  return `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
var<workgroup> scratch: array<${type}, ${WORKGROUP_SIZE}>;
fn combine(a: ${type}, b: ${type}) -> ${type} { return ${combine}; }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let segment = (wg.z * DISPATCH_Y + wg.y) * DISPATCH_X + wg.x;
  if (segment >= SEGMENT_COUNT) { return; }
  let begin = segmentOffsets[SEGMENT_OFFSET + segment];
  let end = segmentOffsets[SEGMENT_OFFSET + segment + 1u];
  var value: ${type} = ${identity};
  var i = begin + lane;
  loop {
    if (i >= end) { break; }
    value = combine(value, inputValues[INPUT_OFFSET + i]);
    i += ${WORKGROUP_SIZE}u;
  }
  scratch[lane] = value;
  workgroupBarrier();
  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { scratch[lane] = combine(scratch[lane], scratch[lane + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    outputValues[OUTPUT_OFFSET + segment] = select(scratch[0], ${type}(${reduction.operation === 'sum' ? '0' : '0'}), begin == end);
  }
}`;
}
