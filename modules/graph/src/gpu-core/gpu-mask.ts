// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from './gpu-command-graph';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource,
  type GPUBoundedDispatchLayout
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {getViewElementOffset, validatePackedUint32View} from './graph-data-view-utils';
import {alignGraphVectorViews} from './graph-vector-view-utils';

const MASK_WORKGROUP_SIZE = 256;

/** Logical operation used to compose packed GPU selection masks. */
export type GPUMaskOperation = 'and' | 'or' | 'xor' | 'difference' | 'not';

/** One packed mask or an ordered vector of packed mask chunks. */
export type GPUMaskInput = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/** Properties for one graph-native, source-order-preserving mask composition. */
export type GPUMaskProps = {
  /** Prefix for generated command-graph node identifiers. */
  id?: string;
  /** Input masks with matching logical lengths; chunk boundaries may differ. */
  inputs: readonly GPUMaskInput[];
  /** Caller-owned output mask with the same logical length as every input. */
  output: GPUMaskInput;
  /** Boolean operation. Defaults to intersection. */
  operation?: GPUMaskOperation;
};

/**
 * Composes GPU-resident visibility and selection masks without CPU synchronization.
 *
 * Nonzero input values are true. Output values are always canonical zero or one. Vector inputs
 * retain their original chunk boundaries and never cause implicit concatenation or repacking.
 */
export class GPUMask {
  /** Prefix for generated command-graph node identifiers. */
  readonly id: string;
  /** Packed source masks. */
  readonly inputs: readonly GPUMaskInput[];
  /** Caller-owned, nonaliasing packed result. */
  readonly output: GPUMaskInput;
  /** Boolean operation applied independently to each row. */
  readonly operation: GPUMaskOperation;

  constructor(props: GPUMaskProps) {
    this.id = props.id ?? 'gpu-mask';
    this.inputs = props.inputs;
    this.output = props.output;
    this.operation = props.operation ?? 'and';

    if (this.inputs.length === 0) {
      throw new Error(`${this.id} requires at least one input mask`);
    }
    if (this.operation === 'not' && this.inputs.length !== 1) {
      throw new Error(`${this.id} not requires exactly one input mask`);
    }

    const outputChunks = getMaskChunks(this.output);
    for (const chunk of outputChunks) {
      validatePackedUint32View(chunk, `${this.id} output`);
    }
    for (const [inputIndex, input] of this.inputs.entries()) {
      for (const chunk of getMaskChunks(input)) {
        validatePackedUint32View(chunk, `${this.id} input ${inputIndex}`);
      }
      if (input.length !== this.output.length) {
        throw new Error(`${this.id} input ${inputIndex} length must equal output length`);
      }
      for (const inputChunk of getMaskChunks(input)) {
        if (outputChunks.some(outputChunk => outputChunk.buffer === inputChunk.buffer)) {
          throw new Error(`${this.id} output must use separate buffers from input masks`);
        }
      }
    }
  }

  /**
   * Adds one compute pass per nonempty source chunk.
   *
   * The caller remains responsible for graph compilation, command submission, and readback.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...getGPUMaskCommandNodesWithDispatchLimit(
        this,
        graph,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );

    return nodes;
  }
}

/** Adds source-aligned mask composition with an explicit dispatch limit. @internal */
export function getGPUMaskCommandNodesWithDispatchLimit<Parameters>(
  mask: GPUMask,
  graph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const allViews = [mask.output, ...mask.inputs] as const;
  for (const chunk of allViews.flatMap(getMaskChunks)) {
    if (chunk.buffer.graph !== graph) {
      throw new Error(`${mask.id} masks must belong to the target graph`);
    }
  }

  const spans = alignGraphVectorViews(graph, [mask.output, ...mask.inputs]);
  for (const [chunkIndex, span] of spans.entries()) {
    const [output, ...inputs] = span;
    if (output.length === 0) {
      continue;
    }
    nodes.push(
      ...addMaskPass(graph, {
        id: spans.length > 1 ? `${mask.id}-chunk-${chunkIndex}` : mask.id,
        inputs,
        output,
        operation: mask.operation,
        dispatchLayout: getBoundedDispatchLayout(
          'GPUMask',
          output.length,
          MASK_WORKGROUP_SIZE,
          maxComputeWorkgroupsPerDimension
        )
      })
    );
  }

  return nodes;
}

/** Returns the original, ordered source chunks without materializing a packed vector. */
function getMaskChunks(input: GPUMaskInput): readonly GraphDataView<'uint32'>[] {
  return input instanceof GraphVectorView ? input.data : [input];
}

/** Adds one independently composable packed-mask compute node. */
function addMaskPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    inputs: readonly GraphDataView<'uint32'>[];
    output: GraphDataView<'uint32'>;
    operation: GPUMaskOperation;
    dispatchLayout: GPUBoundedDispatchLayout;
  }
): readonly GPUCommandNode<Parameters>[] {
  const bindings: GPUKernelBinding[] = [
    ...props.inputs.map(
      (input, inputIndex): GPUKernelBinding => ({
        name: `input${inputIndex}`,
        view: input,
        access: 'read'
      })
    ),
    {name: 'outputMask', view: props.output, access: 'write'}
  ];
  // Sizes and offsets are pipeline constants; the operation and input count shape the shader.
  const constants: Record<string, number> = {
    ELEMENT_COUNT: props.output.length,
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    ...Object.fromEntries(
      props.inputs.map((input, inputIndex) => [
        `INPUT_${inputIndex}_OFFSET`,
        getViewElementOffset(input)
      ])
    ),
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const conditions = props.inputs.map(
    (_, inputIndex) => `input${inputIndex}[INPUT_${inputIndex}_OFFSET + index] != 0u`
  );
  const expression = getMaskExpression(props.operation, conditions);
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

@compute @workgroup_size(${MASK_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(MASK_WORKGROUP_SIZE).body}
  if (index >= ELEMENT_COUNT) {
    return;
  }
  outputMask[OUTPUT_OFFSET + index] = select(0u, 1u, ${expression});
}`;
  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUMask',
      moduleId: `GPUMask/${props.operation}`,
      formula: 'gpu-core.mask',
      source,
      constants,
      bindings,
      workgroupSize: MASK_WORKGROUP_SIZE,
      dispatchLayout: props.dispatchLayout,
      invocationCount: props.output.length,
      // Mask nodes have never carried a workload estimate.
      workloadScope: 'never'
    })
  ];
}

/** Constructs a boolean WGSL expression with explicit zero/nonzero mask semantics. */
function getMaskExpression(operation: GPUMaskOperation, conditions: readonly string[]): string {
  switch (operation) {
    case 'and':
      return conditions.map(condition => `(${condition})`).join(' && ');
    case 'or':
      return conditions.map(condition => `(${condition})`).join(' || ');
    case 'xor':
      return `(${conditions.map(condition => `select(0u, 1u, ${condition})`).join(' ^ ')}) != 0u`;
    case 'difference':
      return [
        `(${conditions[0]})`,
        ...conditions.slice(1).map(condition => `!(${condition})`)
      ].join(' && ');
    case 'not':
      return `!(${conditions[0]})`;
  }
}
