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

// The mixed executor accepts only portable 32-bit vectors, not every GPUVector memory format.
const CONTRIBUTOR_OPERATION_FORMATS = new Set<GPUGraphVectorFormat>([
  'float32',
  'float32x2',
  'float32x3',
  'float32x4',
  'sint32',
  'sint32x2',
  'sint32x3',
  'sint32x4',
  'uint32',
  'uint32x2',
  'uint32x3',
  'uint32x4'
]);

/** Operation type shared by every {@link GPUContributorOperation}; the lowering matches on it. */
export const GPU_CONTRIBUTOR_OPERATION_TYPE = 'analysis-contributor';

// Symbol.for shares identity across the package's import/require implementations. The version
// names the lowering protocol, not the package release. A type string alone never establishes it.
const CONTRIBUTOR_OPERATION_BRAND = Symbol.for(
  '@luma-contributors/analysis/GPUContributorOperation/v1'
);

/** One logical graph value a contributor operation writes. Contributor outputs are caller-sized. */
export type GPUContributorOperationOutput = {
  /** Non-empty name of the logical graph value. */
  value: string;
  format: GPUGraphVectorFormat;
  /** Row count of the value: at least one, or zero for a port listed in `emptyOutputPorts`. */
  length: number;
};

// Construction captures descriptor fields in a frozen plain object, not a descriptor subclass.
type OperationOutputSnapshot<Descriptor> = Descriptor extends GPUContributorOperationOutput
  ? Readonly<Pick<Descriptor, keyof GPUContributorOperationOutput>>
  : undefined;

// Distribute over both map alternatives and requested keys. An index signature (including a
// template pattern such as `band${number}`) does not guarantee that any matching key is present.
// A broader/overlapping request can also miss a finite known key, so it includes absence.
type OperationOutputLookup<Outputs, Port extends string> = Outputs extends unknown
  ? Port extends unknown
    ? [Port & keyof Outputs] extends [never]
      ? undefined
      : {
          [Key in keyof Outputs]-?: [Port & Key] extends [never]
            ? [never]
            : [
                | OperationOutputSnapshot<Outputs[Key]>
                | (Port extends Key
                    ? {} extends Pick<Outputs, Key>
                      ? undefined
                      : never
                    : undefined)
              ];
        }[keyof Outputs][0]
    : never
  : never;

/**
 * Returns the packed view of one port at lowering time. `port` is a key of
 * {@link GPUContributorOperationProps.inputs} or {@link GPUContributorOperationProps.outputs}.
 * Pass a format or readonly format tuple to check the resolved view and infer its format. Outputs
 * are always checked against their declaration. The legacy one-argument input form is unchecked.
 */
export type GPUContributorOperationViewGetter = {
  <Format extends GPUGraphVectorFormat>(port: string, format: Format): GraphDataView<Format>;
  <Format extends GPUGraphVectorFormat>(
    port: string,
    formats: readonly Format[]
  ): GraphDataView<Format>;
  <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format>;
};

/** Properties of {@link GPUContributorOperation}. */
export type GPUContributorOperationProps<
  Outputs extends Readonly<Record<string, GPUContributorOperationOutput>> = Readonly<
    Record<string, GPUContributorOperationOutput>
  >
> = {
  /** Operation ID, also the ID in messages. */
  id: string;
  /** Non-empty names of logical graph values the contributor reads, by port name. */
  inputs: Readonly<Record<string, string>>;
  /** Logical graph values the contributor writes, by port name. */
  outputs: Outputs;
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
 * Validates original logical-port maps before CPU or GPU forms key outputs by destination.
 * Repeated input reads are allowed; outputs must have distinct destinations and cannot overwrite
 * inputs. This checks names only and imposes no GPU format, allocation or row-count limits.
 */
export function validateOperationGraphNames(
  id: string,
  inputs: Readonly<Record<string, string>>,
  outputs: Readonly<Record<string, string>>
): void {
  const inputValues = new Set(Object.values(inputs));
  const outputValues = new Set<string>();
  for (const [port, value] of Object.entries(inputs)) {
    validateRequirement(id, port.length > 0, 'input port name must be non-empty');
    validateRequirement(
      id,
      typeof value === 'string' && value.length > 0,
      `inputs.${port} must be a non-empty graph value name`
    );
  }
  for (const [port, value] of Object.entries(outputs)) {
    validateRequirement(id, port.length > 0, 'output port name must be non-empty');
    validateRequirement(
      id,
      !Object.hasOwn(inputs, port),
      `port "${port}" must not be both an input and an output`
    );
    validateRequirement(
      id,
      typeof value === 'string' && value.length > 0,
      `outputs.${port}.value must be a non-empty graph value name`
    );
    validateRequirement(
      id,
      !inputValues.has(value) && !outputValues.has(value),
      `outputs.${port} must name a graph value ("${value}") that no other port uses`
    );
    outputValues.add(value);
  }
}

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
 * Limits from the published mixed executor: every value is a `float32`, `sint32` or `uint32`
 * vector of one to four components; no textures; no value read and written by the same operation.
 * The executor accepts multiple chunks, but this adapter's lowering
 * ({@link GPU_CONTRIBUTOR_OPERATION_LOWERING}) requires every resolved port to be one packed chunk
 * and rejects a split value, even when the contributor itself accepts chunked views; use the
 * contributor's `getCommandNodes(graph)` directly for chunked input.
 * Graph value names, output formats and output lengths are validated at construction, before
 * luma.gl plans the contract or the contributor is built. Zero output lengths require the port
 * to be listed in `emptyOutputPorts`. Packed chunks are checked when views are resolved.
 *
 * The graph lowers its operations again on every run, so the contributor is built once per run.
 * A contributor that owns GPU resources (it has a `destroy()` method) is kept by the operation and
 * destroyed when the next run builds its successor; call {@link GPUContributorOperation.destroy}
 * after the compiled graph is destroyed to release the last one. An operation therefore serves
 * one compiled graph, and one prepared execution of it at a time.
 */
export class GPUContributorOperation<
  Outputs extends Readonly<Record<string, GPUContributorOperationOutput>> = Readonly<
    Record<string, GPUContributorOperationOutput>
  >
> implements GPUOperation
{
  readonly [CONTRIBUTOR_OPERATION_BRAND] = true;
  readonly type = GPU_CONTRIBUTOR_OPERATION_TYPE;
  readonly id: string;
  readonly inputs: Readonly<Record<string, string>>;
  readonly outputs: Readonly<Record<string, Readonly<GPUContributorOperationOutput>>>;
  private readonly contributorFactory: GPUContributorOperationProps['createContributor'];
  private ownedContributor?: {destroy(): void};

  constructor(props: GPUContributorOperationProps<Outputs>) {
    this.id = props.id;
    this.inputs = Object.freeze({...props.inputs});
    this.outputs = Object.freeze(
      Object.fromEntries(
        Object.entries(props.outputs).map(([port, output]) => [port, Object.freeze({...output})])
      )
    );
    this.contributorFactory = props.createContributor;
    validateOperationGraphNames(
      this.id,
      this.inputs,
      Object.fromEntries(Object.entries(this.outputs).map(([port, output]) => [port, output.value]))
    );
    const emptyOutputPorts = new Set(props.emptyOutputPorts ?? []);
    for (const [port, output] of Object.entries(this.outputs)) {
      validateRequirement(
        this.id,
        CONTRIBUTOR_OPERATION_FORMATS.has(output.format),
        `outputs.${port}.format must be float32, sint32 or uint32 with one to four components`
      );
      // An output is at least one row unless the operation names it as one that may be empty.
      validateCapacity(
        this.id,
        `outputs.${port}.length`,
        output.length,
        emptyOutputPorts.has(port) ? 0 : 1
      );
    }
  }

  /**
   * Looks up a logical output port, not its graph destination. Returns frozen value/format/length
   * fields with their literal types. Conditional selections, index signatures and dynamic keys
   * include possible absence. Wrappers can pass their selected output map as the class type
   * argument. The legacy `outputs` dictionary remains available.
   */
  getOutput<Port extends string>(port: Port): OperationOutputLookup<Outputs, Port>;
  getOutput(port: string): Readonly<GPUContributorOperationOutput> | undefined {
    return Object.hasOwn(this.outputs, port) ? this.outputs[port] : undefined;
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

/** Checks the versioned identity and every field the lowering consumes across module conditions. */
function isContributorOperation(operation: GPUOperation): operation is GPUContributorOperation {
  const candidate = operation as Partial<GPUContributorOperation>;
  if (
    candidate[CONTRIBUTOR_OPERATION_BRAND] !== true ||
    candidate.type !== GPU_CONTRIBUTOR_OPERATION_TYPE ||
    typeof candidate.id !== 'string' ||
    typeof candidate.createContributor !== 'function' ||
    typeof candidate.destroy !== 'function' ||
    !candidate.inputs ||
    typeof candidate.inputs !== 'object' ||
    Array.isArray(candidate.inputs) ||
    !candidate.outputs ||
    typeof candidate.outputs !== 'object' ||
    Array.isArray(candidate.outputs)
  ) {
    return false;
  }
  if (
    !Object.values(candidate.inputs).every(value => typeof value === 'string' && value.length > 0)
  ) {
    return false;
  }
  return Object.values(candidate.outputs).every(
    output =>
      output !== null &&
      typeof output === 'object' &&
      typeof output.value === 'string' &&
      output.value.length > 0 &&
      CONTRIBUTOR_OPERATION_FORMATS.has(output.format) &&
      Number.isInteger(output.length) &&
      output.length >= 0 &&
      output.length <= 0xffffffff
  );
}

/**
 * Lowers every {@link GPUContributorOperation} of a graph: resolves its ports to packed views and
 * adds the contributor's command nodes. Pass it in `lowerings` when compiling the `GPUGraph`.
 */
export const GPU_CONTRIBUTOR_OPERATION_LOWERING: GPUOperationLowering = {
  type: GPU_CONTRIBUTOR_OPERATION_TYPE,
  lower(operation, context) {
    if (!isContributorOperation(operation)) {
      throw new Error(`${operation.id} is not a GPUContributorOperation`);
    }
    validateOperationGraphNames(
      operation.id,
      operation.inputs,
      Object.fromEntries(
        Object.entries(operation.outputs).map(([port, output]) => [port, output.value])
      )
    );
    const getView: GPUContributorOperationViewGetter = <Format extends GPUGraphVectorFormat>(
      port: string,
      formats?: Format | readonly Format[]
    ): GraphDataView<Format> => {
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
      const view = chunks[0];
      const output = Object.hasOwn(operation.outputs, port) ? operation.outputs[port] : undefined;
      const checkFormats = (expected: readonly GPUGraphVectorFormat[]) => {
        validateRequirement(
          operation.id,
          expected.includes(view.format),
          `port "${port}" (graph value "${value}") must have format ${expected.join(' or ')}, not ${view.format}`
        );
      };
      if (output) {
        checkFormats([output.format]);
      }
      if (formats !== undefined) {
        checkFormats(typeof formats === 'string' ? [formats] : formats);
      }
      return view;
    };
    context.lower(operation.createContributor(getView));
  }
};
