// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  importGraphBuffer,
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

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
  const control = allocate('requested-generation', 'uint32', 2, 8, generation);
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
    body: `let requestedLow = control[controlOffset];
  let requestedHigh = control[controlOffset + 1u];
  let refresh = state[stateOffset] != requestedLow || state[stateOffset + 1u] != requestedHigh;
  state[stateOffset + 2u] = requestedLow;
  state[stateOffset + 3u] = requestedHigh;
  ${gate ? `${gate.armWGSL}\n  if (!refresh) { ${gate.disarmWGSL} }` : 'if (!refresh) { return; }'}`
  });
  const nodes: GPUCommandNode<Parameters>[] = [
    {
      ...prepare,
      condition: {
        id: `${id}-rebuild`,
        source: 'cpu',
        evaluate(parameters) {
          props.recordEncoding(parameters);
          control.buffer.defaultBuffer!.write(generation);
          return true;
        }
      }
    }
  ];
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
