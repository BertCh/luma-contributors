// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, type GraphDataView, GraphVectorView} from './gpu-command-graph';
import {alignGraphVectorViews, getGraphVectorData} from './graph-vector-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {
  doGraphDataViewsOverlap,
  getViewElementOffset,
  validatePackedView,
  type GPUScalarFormat
} from './graph-data-view-utils';

const WORKGROUP_SIZE = 256;
const SCALAR_FORMATS = ['uint32', 'sint32', 'float32'] as const;

export type GPUElementwiseOperation =
  | 'copy'
  | 'add'
  | 'subtract'
  | 'multiply'
  | 'multiply-add'
  | 'min'
  | 'max';

export type GPUElementwiseProps<T extends GPUScalarFormat = GPUScalarFormat> = {
  id?: string;
  /** First packed scalar input. */
  input: GraphDataView<T> | GraphVectorView<T>;
  /** Second packed scalar input required by binary and ternary operations. */
  inputB?: GraphDataView<T> | GraphVectorView<T>;
  /** Third packed scalar input required by `multiply-add`. */
  inputC?: GraphDataView<T> | GraphVectorView<T>;
  /** Caller-owned packed scalar output. */
  output: GraphDataView<T> | GraphVectorView<T>;
  /** Operation applied independently to every row. */
  operation: GPUElementwiseOperation;
};

/**
 * Applies one scalar operation independently to each packed GPU row.
 *
 * `copy` is unary. `add`, `subtract`, `multiply`, `min`, and `max` are binary.
 * `multiply-add` is ternary and computes `a * b + c`. All inputs and output must have identical
 * format and logical length.
 */
export class GPUElementwise<T extends GPUScalarFormat = GPUScalarFormat> {
  readonly id: string;
  readonly input: GPUElementwiseProps<T>['input'];
  readonly inputB: GPUElementwiseProps<T>['inputB'];
  readonly inputC: GPUElementwiseProps<T>['inputC'];
  readonly output: GPUElementwiseProps<T>['output'];
  readonly operation: GPUElementwiseOperation;

  constructor(props: GPUElementwiseProps<T>) {
    this.id = props.id ?? 'gpu-elementwise';
    this.input = props.input;
    this.inputB = props.inputB;
    this.inputC = props.inputC;
    this.output = props.output;
    this.operation = props.operation;

    for (const view of [this.input, this.inputB, this.inputC, this.output]) {
      if (view)
        for (const chunk of getGraphVectorData(view)) {
          validatePackedView(chunk, SCALAR_FORMATS, this.id);
        }
    }
    if (this.input.format !== this.output.format || this.input.length !== this.output.length) {
      throw new Error(`${this.id} input and output must have matching format and length`);
    }

    const isUnary = this.operation === 'copy';
    const isTernary = this.operation === 'multiply-add';
    if (!isUnary && !this.inputB) {
      throw new Error(`${this.id} ${this.operation} requires inputB`);
    }
    if (isTernary && !this.inputC) {
      throw new Error(`${this.id} multiply-add requires inputC`);
    }
    if (!isTernary && this.inputC) {
      throw new Error(`${this.id} inputC is only valid for multiply-add`);
    }

    for (const [name, input] of [
      ['inputB', this.inputB],
      ['inputC', this.inputC]
    ] as const) {
      if (!input) continue;
      if (input.format !== this.input.format || input.length !== this.input.length) {
        throw new Error(`${this.id} ${name} must match input format and length`);
      }
    }

    const inputBuffers = new Set(
      [this.input, this.inputB, this.inputC].flatMap(view =>
        view ? getGraphVectorData(view).map(chunk => chunk.buffer) : []
      )
    );
    const outputChunks = getGraphVectorData(this.output);
    if (outputChunks.some(chunk => inputBuffers.has(chunk.buffer))) {
      throw new Error(`${this.id} output must use separate buffers from inputs`);
    }
    for (const [index, chunk] of outputChunks.entries()) {
      if (outputChunks.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, chunk))) {
        throw new Error(`${this.id} output chunks must not overlap`);
      }
    }

    if (
      !['copy', 'add', 'subtract', 'multiply', 'multiply-add', 'min', 'max'].includes(
        this.operation
      )
    ) {
      throw new Error(`${this.id} unsupported operation ${this.operation}`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    const views = [this.input, this.inputB, this.inputC, this.output];
    for (const view of views) {
      if (view && getGraphVectorData(view).some(chunk => chunk.buffer.graph !== graph)) {
        throw new Error(`${this.id} views must belong to the target graph`);
      }
    }
    if (
      this.input instanceof GraphVectorView ||
      this.inputB instanceof GraphVectorView ||
      this.inputC instanceof GraphVectorView ||
      this.output instanceof GraphVectorView
    ) {
      const operands = [
        this.input,
        this.inputB ?? this.input,
        this.inputC ?? this.input,
        this.output
      ] as const;
      return alignGraphVectorViews(graph, operands).flatMap(
        ([input, inputB, inputC, output], index) =>
          new GPUElementwise({
            id: `${this.id}-${index}`,
            input,
            inputB: this.inputB ? inputB : undefined,
            inputC: this.inputC ? inputC : undefined,
            output,
            operation: this.operation
          }).getCommandNodes(graph)
      );
    }
    // All vector operands have been lowered to borrowed atomic spans above.
    const input = this.input;
    const inputB = this.inputB;
    const inputC = this.inputC;
    const output = this.output;
    if (this.output.length === 0) return nodes;

    const bindings: GPUKernelBinding[] = [
      {name: 'inputValues', view: input, access: 'read'},
      ...(inputB ? [{name: 'inputBValues', view: inputB, access: 'read' as const}] : []),
      ...(inputC ? [{name: 'inputCValues', view: inputC, access: 'read' as const}] : []),
      {name: 'outputValues', view: output, access: 'write'}
    ];
    const dispatchLayout = getBoundedDispatchLayout(
      'GPUElementwise',
      this.output.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    // Length, view offsets, and dispatch shape are pipeline constants shared by every size.
    const constants: Record<string, number> = {
      LENGTH: this.output.length,
      INPUT_OFFSET: getViewElementOffset(input),
      OUTPUT_OFFSET: getViewElementOffset(output),
      ...(inputB && {INPUT_B_OFFSET: getViewElementOffset(inputB)}),
      ...(inputC && {INPUT_C_OFFSET: getViewElementOffset(inputC)}),
      ...getBoundedDispatchOverrides(dispatchLayout)
    };
    nodes.push(
      createGPUKernelNode(graph, {
        id: this.id,
        operation: 'GPUElementwise',
        moduleId: `GPUElementwise/${this.operation}`,
        formula: 'gpu-core.elementwise',
        source: makeShaderSource({...this, input, inputB, inputC, output}, constants, bindings),
        constants,
        bindings,
        workgroupSize: WORKGROUP_SIZE,
        dispatchLayout,
        invocationCount: this.output.length,
        workload: {
          readByteLength:
            this.input.length *
            4 *
            (1 + Number(Boolean(this.inputB)) + Number(Boolean(this.inputC))),
          writeByteLength: this.output.length * 4
        }
      })
    );

    return nodes;
  }
}

function makeShaderSource(
  elementwise: {
    input: GraphDataView;
    inputB?: GraphDataView;
    inputC?: GraphDataView;
    output: GraphDataView;
    operation: GPUElementwiseOperation;
  },
  constants: Readonly<Record<string, number>>,
  bindings: readonly GPUKernelBinding[]
): string {
  const expression = getExpression(elementwise.operation);
  const optionalLoads: string[] = [];
  if (elementwise.inputB) optionalLoads.push('let b = inputBValues[INPUT_B_OFFSET + index];');
  if (elementwise.inputC) optionalLoads.push('let c = inputCValues[INPUT_C_OFFSET + index];');

  return `${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexOverrideSource(WORKGROUP_SIZE).body}
  if (index >= LENGTH) { return; }
  let a = inputValues[INPUT_OFFSET + index];
  ${optionalLoads.join('\n  ')}
  outputValues[OUTPUT_OFFSET + index] = ${expression};
}`;
}

function getExpression(operation: GPUElementwiseOperation): string {
  switch (operation) {
    case 'copy':
      return 'a';
    case 'add':
      return 'a + b';
    case 'subtract':
      return 'a - b';
    case 'multiply':
      return 'a * b';
    case 'multiply-add':
      return 'a * b + c';
    case 'min':
      return 'min(a, b)';
    case 'max':
      return 'max(a, b)';
  }
}
