// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  getViewBindingRange,
  type GraphBufferHandle,
  type GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createReadableGWRAuditBuffer} from './gwr-audit-buffer-test-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

/** Executes emitted kernels in Node, or submits the same graph on the lead's GPU runner. */
export function createGWRReviewHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const buffers = new Map<GraphBufferHandle, Buffer>();
  const memory = new Map<GraphBufferHandle, ArrayBuffer>();
  let serial = 0;
  const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: ArrayLike<number>
  ) => {
    const data = format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values);
    const buffer = createReadableGWRAuditBuffer(device, data);
    const view = importGraphBuffer(
      graph,
      `review-${serial++}`,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
    buffers.set(view.buffer, buffer);
    memory.set(view.buffer, data.buffer);
    return view;
  };
  const getMemory = (view: GraphDataView) => {
    if (!memory.has(view.buffer)) memory.set(view.buffer, new ArrayBuffer(view.buffer.byteLength));
    return memory.get(view.buffer)!;
  };
  const getRecord = (node: GPUCommandNode) => {
    const recorded = getWGSLKernelNodeRecord(graph, node.id);
    if (recorded) return recorded;
    // Runtime merge kernels use the public command-node API directly. Capture their
    // real compiled source and public resource declarations; no numerical code is replaced.
    const sources = getGPUCommandNodeKernelSources([node]);
    if (sources.length !== 1) throw new Error(`Review interpreter needs one kernel: ${node.id}`);
    const captured = sources[0];
    const locations = Array.from(
      captured.source.matchAll(/@binding\((\d+)\) var<storage,[^>]+> (\w+):/g)
    );
    const bindings = locations.map(match => {
      const resource = node.resources?.[Number(match[1])];
      if (!resource || !('buffer' in resource) || !('format' in resource.buffer))
        throw new Error(`Review needs a view for ${match[2]}`);
      return {name: match[2], view: resource.buffer as GraphDataView};
    });
    const size = Number(captured.source.match(/@workgroup_size\((\d+)\)/)?.[1]);
    const invocations = node.workload?.maximumInvocationCount;
    if (!(size > 0) || invocations === undefined || invocations > size)
      throw new Error(`Review runtime fixture exceeds one workgroup: ${node.id}`);
    return {...captured, bindings, workgroups: [1, 1, 1] as const};
  };
  return {
    graph,
    port,
    set: (view: GraphDataView, values: ArrayLike<number>) => {
      const data = view.format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values);
      new Uint8Array(getMemory(view), view.byteOffset, data.byteLength).set(
        new Uint8Array(data.buffer)
      );
      buffers.get(view.buffer)?.write(data, view.byteOffset);
    },
    get: (view: GraphDataView) =>
      Array.from(
        view.format === 'uint32'
          ? new Uint32Array(getMemory(view), view.byteOffset, view.length)
          : new Float32Array(getMemory(view), view.byteOffset, view.length)
      ),
    read: async (view: GraphDataView) =>
      view.format === 'uint32'
        ? readUint32(buffers.get(view.buffer)!)
        : readFloat32(buffers.get(view.buffer)!),
    execute: (nodes: readonly GPUCommandNode[]) => {
      for (const node of nodes) {
        const record = getRecord(node);
        const bindings = Object.fromEntries(
          record.bindings.map((binding, index) => {
            const range = getViewBindingRange(binding.view);
            return [index, new Uint8Array(getMemory(binding.view), range.offset, range.size)];
          })
        );
        // Preserve WGSL scalar let snapshots before Givens/Jacobi overwrites array entries.
        const source = record.source
          .replace(/^\s*_\s*=.*;$/gm, '')
          .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)')
          .replace(
            /let (held|incoming|x|y|vectorFirst|vectorSecond) = ([^;]+);/g,
            'let $1 = ($2) + 0.0;'
          );
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          'main',
          [...record.workgroups],
          {0: bindings},
          {constants: record.constants}
        );
      }
    },
    submit: (nodes: readonly GPUCommandNode[]) => {
      for (const node of nodes) graph.add(node);
      const compiled = graph.compile();
      submitGraph(device, compiled);
      return compiled;
    },
    record: getRecord,
    destroy: () => {
      for (const buffer of buffers.values()) buffer.destroy();
    }
  };
}
