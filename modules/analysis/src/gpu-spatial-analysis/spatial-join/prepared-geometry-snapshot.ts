// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createGPUCopyCommandNode,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  importGraphBuffer,
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** A fixed table supplies every possible byte as an aligned uint32 copy source. */
const PREPARED_GENERATION_BYTE_VALUES = 256;
/** Two uint32 generation words are recorded as eight independently copied byte values. */
const PREPARED_GENERATION_BYTE_COUNT = 8;

/** @internal Two uint32 words avoid wrapping a generation after 2^32 invalidations. */
export function advancePreparedGeometryGeneration(generation: Uint32Array): void {
  generation[0] = (generation[0] + 1) >>> 0;
  if (generation[0] === 0) generation[1] = (generation[1] + 1) >>> 0;
}

/**
 * @internal Retain geometry until a requested generation actually executes.
 * The published builders replay against these views; only geometry refresh is gated. Their
 * scratch is therefore populated on every execution without requiring private dispatch metadata.
 */
export function createPreparedGeometrySnapshot<
  Parameters,
  Views extends Record<string, GraphDataView>
>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    views: Views;
    generation: Uint32Array;
    ownedBuffers: Buffer[];
    recordEncoding(parameters: Parameters): void;
  }
) {
  const {id, generation, ownedBuffers} = props;
  const entries = Object.entries(props.views);
  const copied = entries.filter(([, view]) => view.length > 0);
  for (const [name, byteLength] of [
    ['generation state', 16],
    ['generation byte table', 4 * PREPARED_GENERATION_BYTE_VALUES],
    ['requested generation bytes', 4 * PREPARED_GENERATION_BYTE_COUNT],
    ['snapshot gate', Math.max(4, 12 * copied.length)]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'use a device with larger storage bindings'
    );
    validateCapacity(id, `${name}.byteLength`, byteLength, 4, graph.device.limits.maxBufferSize);
  }
  // This walk is over the fixed geometry port map (at most four views), never over input rows.
  for (const [name, view] of entries) {
    const byteLength = Math.max(4, view.length * view.rowByteLength);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      `snapshot.${name}`,
      byteLength,
      'reduce geometry rows'
    );
    validateCapacity(
      id,
      `snapshot.${name}.byteLength`,
      byteLength,
      4,
      graph.device.limits.maxBufferSize
    );
  }
  const allocate = <Format extends GraphDataView['format']>(
    name: string,
    format: Format,
    length: number,
    byteLength: number,
    values?: Uint32Array
  ): GraphDataView<Format> => {
    const buffer = graph.device.createBuffer({
      id: `${id}-${name}`,
      byteLength: Math.max(byteLength, 4),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
      ...(values ? {data: values} : {})
    });
    ownedBuffers.push(buffer);
    return importGraphBuffer(graph, `${id}-${name}`, buffer, format, length);
  };
  const views = Object.fromEntries(
    entries.map(([name, view]) => [
      name,
      allocate(`snapshot-${name}`, view.format, view.length, view.length * view.rowByteLength)
    ])
  ) as Views;
  // Buffer.write queues data outside command encoders. Instead record source offsets into one
  // immutable table: reverse submission observes the request recorded by each command stream.
  // Storage stays fixed across encodings and invalidations (1024-byte table + 32-byte request).
  const byteValues = allocate(
    'generation-byte-values',
    'uint32',
    PREPARED_GENERATION_BYTE_VALUES,
    4 * PREPARED_GENERATION_BYTE_VALUES,
    Uint32Array.from({length: PREPARED_GENERATION_BYTE_VALUES}, (_, value) => value)
  );
  const control = allocate(
    'requested-generation',
    'uint32',
    PREPARED_GENERATION_BYTE_COUNT,
    4 * PREPARED_GENERATION_BYTE_COUNT
  );
  const request = createGPUCopyCommandNode<Parameters>({
    id: `${id}-snapshot-request`,
    resources: [
      {buffer: byteValues, usage: 'copy-source'},
      {buffer: control, usage: 'copy-destination'}
    ],
    workload: {
      operation: 'PreparedGeometrySnapshot',
      commandCount: PREPARED_GENERATION_BYTE_COUNT,
      maximumInvocationCount: 0,
      readByteLength: 4 * PREPARED_GENERATION_BYTE_COUNT,
      writeByteLength: 4 * PREPARED_GENERATION_BYTE_COUNT
    },
    condition: {
      id: `${id}-rebuild`,
      source: 'cpu',
      evaluate(parameters) {
        props.recordEncoding(parameters);
        return true;
      }
    },
    compile: () => ({
      encode({commandEncoder, getBuffer}) {
        const requestedLow = generation[0];
        const requestedHigh = generation[1];
        // Fixed eight-copy walk, independent of geometry rows and invalidation count.
        for (let byteIndex = 0; byteIndex < PREPARED_GENERATION_BYTE_COUNT; byteIndex++) {
          const word = byteIndex < 4 ? requestedLow : requestedHigh;
          const value = (word >>> (8 * (byteIndex % 4))) & 0xff;
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(byteValues),
            sourceOffset: byteValues.byteOffset + 4 * value,
            destinationBuffer: getBuffer(control),
            destinationOffset: control.byteOffset + 4 * byteIndex,
            size: 4
          });
        }
      }
    })
  });
  // Applied low/high, then the generation latched by executed preparation work.
  const state = allocate('applied-generation', 'uint32', 4, 16, new Uint32Array(4));
  const gate = copied.length
    ? createWGSLDispatchGate(graph, {
        id: `${id}-snapshot-gate`,
        operation: 'PreparedGeometrySnapshot',
        shapes: copied.map(([, view]) => [(view.length * view.rowByteLength) / 4] as const)
      })
    : undefined;
  const prepare = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-snapshot-prepare`,
    operation: 'PreparedGeometrySnapshot',
    variant: 'requested-generation',
    invocationCount: 1,
    bindings: [
      {name: 'control', view: control, type: 'u32', access: 'read'},
      {name: 'state', view: state, type: 'u32', access: 'read_write'},
      ...(gate ? [gate.binding] : [])
    ],
    body: `let requestedLow = control[controlOffset] | (control[controlOffset + 1u] << 8u) |
    (control[controlOffset + 2u] << 16u) | (control[controlOffset + 3u] << 24u);
  let requestedHigh = control[controlOffset + 4u] | (control[controlOffset + 5u] << 8u) |
    (control[controlOffset + 6u] << 16u) | (control[controlOffset + 7u] << 24u);
  let refresh = state[stateOffset] != requestedLow || state[stateOffset + 1u] != requestedHigh;
  state[stateOffset + 2u] = requestedLow;
  state[stateOffset + 3u] = requestedHigh;
  ${gate ? `${gate.armWGSL}\n  if (!refresh) { ${gate.disarmWGSL} }` : 'if (!refresh) { return; }'}`
  });
  const nodes: GPUCommandNode<Parameters>[] = [request, prepare];
  for (const [slot, [name, source]] of copied.entries()) {
    const wordCount = (source.length * source.rowByteLength) / 4;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-snapshot-${name}`,
        operation: 'PreparedGeometrySnapshot',
        variant: 'copy-geometry',
        invocationCount: wordCount,
        ...gate!.getGate(slot),
        bindings: [
          {name: 'source', view: source, type: 'u32', access: 'read'},
          {name: 'destination', view: views[name], type: 'u32', access: 'read_write'}
        ],
        body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
      })
    );
  }
  return {
    views,
    nodes,
    commit(buildNodes: readonly GPUCommandNode<Parameters>[]): GPUCommandNode<Parameters> {
      const commit = createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-snapshot-commit`,
        operation: 'PreparedGeometrySnapshot',
        variant: 'applied-generation',
        invocationCount: 1,
        bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
        body: `state[stateOffset] = state[stateOffset + 2u];
  state[stateOffset + 1u] = state[stateOffset + 3u];`
      });
      return {...commit, dependsOn: [...nodes, ...buildNodes].map(node => node.id)};
    }
  };
}
