// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUGraphPrimitive,
  GPUGraphVectorFormat,
  GPUOperation,
  GPUOperationContract,
  GPUOperationLowering,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity, validateRequirement} from '../kernels/gpu-contributor-utils';

/** Operation type shared by every {@link GPUContributorOperation}; the lowering matches on it. */
export const GPU_CONTRIBUTOR_OPERATION_TYPE = 'analysis-contributor';

/** One logical graph value a contributor operation writes. Contributor outputs are caller-sized. */
export type GPUContributorOperationOutput = {
  /** Name of the logical graph value. */
  value: string;
  format: GPUGraphVectorFormat;
  /** Row count of the value: at least one, or zero for a port listed in `emptyOutputPorts`. */
  length: number;
};

/**
 * Returns the packed view of one port at lowering time. `port` is a key of
 * {@link GPUContributorOperationProps.inputs} or {@link GPUContributorOperationProps.outputs}.
 */
export type GPUContributorOperationViewGetter = <Format extends GPUGraphVectorFormat>(
  port: string
) => GraphDataView<Format>;

/** Properties of {@link GPUContributorOperation}. */
export type GPUContributorOperationProps = {
  /** Operation ID, also the ID in messages. */
  id: string;
  /** Logical graph values the contributor reads, by port name. */
  inputs: Readonly<Record<string, string>>;
  /** Logical graph values the contributor writes, by port name. */
  outputs: Readonly<Record<string, GPUContributorOperationOutput>>;
  /**
   * Output ports whose length may be zero, for contributors that accept an empty output (an
   * empty batch, a zero capacity). Every other output must have at least one row. A name that
   * is not an output port is ignored, so an optional port can be listed unconditionally.
   */
  emptyOutputPorts?: readonly string[];
  /** Builds the contributor from the views the graph resolved for the ports. */
  createContributor(getView: GPUContributorOperationViewGetter): GPUGraphPrimitive;
};

/**
 * A contributor as a luma.gl `GPUGraph` operation: logical value names in place of bound views.
 *
 * A contributor is a `GPUCommandNodeProducer` whose views are bound when it is constructed, so a
 * `GPUGraph` cannot plan its inputs and outputs. This operation declares them as a contract and
 * builds the contributor when the graph lowers it, from the views the graph resolved. With it a
 * contributor can run beside `GPUCPUOperation` stages, and a CPU operation with the same inputs
 * and outputs can take its place (dev-docs/contributor-contract.md, section 14).
 *
 * Compile the graph with {@link GPU_CONTRIBUTOR_OPERATION_LOWERING} in `lowerings`.
 *
 * Limits, all from the published mixed executor: every value is a `float32`, `sint32` or `uint32`
 * vector of one to four components; no textures; no value read and written by the same operation;
 * each value is one packed chunk.
 *
 * The graph lowers its operations again on every run, so the contributor is built once per run.
 * A contributor that owns GPU resources (it has a `destroy()` method) is kept by the operation and
 * destroyed when the next run builds its successor; call {@link GPUContributorOperation.destroy}
 * after the compiled graph is destroyed to release the last one. An operation therefore serves
 * one compiled graph, and one prepared execution of it at a time.
 */
export class GPUContributorOperation implements GPUOperation {
  readonly type = GPU_CONTRIBUTOR_OPERATION_TYPE;
  readonly id: string;
  readonly inputs: Readonly<Record<string, string>>;
  readonly outputs: Readonly<Record<string, GPUContributorOperationOutput>>;
  private readonly contributorFactory: GPUContributorOperationProps['createContributor'];
  private ownedContributor?: {destroy(): void};

  constructor(props: GPUContributorOperationProps) {
    this.id = props.id;
    this.inputs = Object.freeze({...props.inputs});
    this.outputs = Object.freeze(
      Object.fromEntries(
        Object.entries(props.outputs).map(([port, output]) => [port, Object.freeze({...output})])
      )
    );
    this.contributorFactory = props.createContributor;
    const inputValues = new Set(Object.values(this.inputs));
    const emptyOutputPorts = new Set(props.emptyOutputPorts ?? []);
    const outputValues = new Set<string>();
    for (const [port, output] of Object.entries(this.outputs)) {
      validateRequirement(
        this.id,
        !Object.hasOwn(this.inputs, port),
        `port "${port}" must not be both an input and an output`
      );
      // An output is at least one row unless the operation names it as one that may be empty.
      validateCapacity(
        this.id,
        `outputs.${port}.length`,
        output.length,
        emptyOutputPorts.has(port) ? 0 : 1
      );
      validateRequirement(
        this.id,
        !inputValues.has(output.value) && !outputValues.has(output.value),
        `outputs.${port} must name a graph value ("${output.value}") that no other port uses`
      );
      outputValues.add(output.value);
    }
  }

  get contract(): GPUOperationContract {
    return {
      outputs: Object.fromEntries(
        Object.values(this.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      values: [
        ...[...new Set(Object.values(this.inputs))].map(value => ({
          value,
          access: 'read' as const
        })),
        ...Object.values(this.outputs).map(output => ({
          value: output.value,
          access: 'write' as const,
          shape: {length: output.length}
        }))
      ]
    };
  }

  /** Builds the contributor from resolved views. Called by the lowering. @internal */
  createContributor(getView: GPUContributorOperationViewGetter): GPUGraphPrimitive {
    // The previous run's commands were submitted before this run was lowered.
    this.destroy();
    const contributor = this.contributorFactory(getView);
    const owner = contributor as GPUGraphPrimitive & {destroy?: unknown};
    if (typeof owner.destroy === 'function') {
      this.ownedContributor = contributor as GPUGraphPrimitive & {destroy(): void};
    }
    return contributor;
  }

  /**
   * Destroys the contributor built for the last run, if it owns GPU resources. Call it after the
   * compiled graph is destroyed. Safe to call more than once.
   */
  destroy(): void {
    const contributor = this.ownedContributor;
    this.ownedContributor = undefined;
    contributor?.destroy();
  }
}

/**
 * Lowers every {@link GPUContributorOperation} of a graph: resolves its ports to packed views and
 * adds the contributor's command nodes. Pass it in `lowerings` when compiling the `GPUGraph`.
 */
export const GPU_CONTRIBUTOR_OPERATION_LOWERING: GPUOperationLowering = {
  type: GPU_CONTRIBUTOR_OPERATION_TYPE,
  lower(operation, context) {
    if (!(operation instanceof GPUContributorOperation)) {
      throw new Error(`${operation.id} is not a GPUContributorOperation`);
    }
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> => {
      // Own keys only: a port name such as `constructor` must not reach Object.prototype.
      const value = Object.hasOwn(operation.inputs, port)
        ? operation.inputs[port]
        : Object.hasOwn(operation.outputs, port)
          ? operation.outputs[port].value
          : undefined;
      validateRequirement(operation.id, value !== undefined, `has no port "${port}"`);
      const chunks = context.resolveVector<Format>(value).data;
      // Contributors take single packed views; a graph value split by the binding limit cannot bind.
      validateRequirement(
        operation.id,
        chunks.length === 1,
        `port "${port}" (graph value "${value}") must be one packed chunk, not ${chunks.length}`
      );
      return chunks[0];
    };
    context.lower(operation.createContributor(getView));
  }
};
