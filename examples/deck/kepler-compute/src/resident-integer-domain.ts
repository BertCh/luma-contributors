// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUBatchHashIndex,
  createTransientView,
  createTransientVectorView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma-contributors/graph/gpu-core';

// Keep an extra domain beyond the accepted group limit so >512 reliably selects native intervals.
// The table also distinguishes 513-1,000 calendar periods (native shows them) from >1,000.
const INTEGER_DOMAIN_CAPACITY = 2048;
export const INTEGER_DOMAIN_READ_BYTES = INTEGER_DOMAIN_CAPACITY * 4 + 28;

/** Bounded distinct keys over the selected source chunks; no CPU source-row scan. */
export class ResidentIntegerChartDomain {
  private readonly compiled: CompiledGPUCommandGraph;
  private readonly keys: Buffer;
  private readonly statistics: Buffer;
  private readonly reserved: Buffer;
  readonly byteLength: number;

  constructor(props: {
    device: Device;
    ordered: GPUVector<'uint32'>;
    validity?: GPUVector<'uint32'>;
    selected?: GPUVector<'uint32'>;
  }) {
    const {device} = props;
    const buffers: Buffer[] = [];
    const graph = new GPUCommandGraph(device, {id: 'chart-integer-domain'});
    const allocate = (id: string, length: number) => {
      const buffer = device.createBuffer({
        id,
        byteLength: length * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC
      });
      buffers.push(buffer);
      return {
        buffer,
        view: graph.createDataView(
          graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
          {format: 'uint32', length}
        )
      };
    };
    try {
      const tableKeys = allocate('integer-domain-keys', INTEGER_DOMAIN_CAPACITY);
      const statistics = allocate('integer-domain-statistics', 6);
      const reserved = allocate('integer-domain-reserved', 1);
      const ordered = graph.importGPUVector('ordered', props.ordered);
      const validity = props.validity
        ? graph.importGPUVector('validity', props.validity)
        : undefined;
      const selected = props.selected
        ? graph.importGPUVector('selected', props.selected)
        : undefined;
      const present = createTransientVectorView(graph, 'hash-validity', ordered);
      graph.addComputePass({
        id: 'reset-reserved',
        resources: [{buffer: reserved.view, usage: 'storage-write'}],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            source:
              '@group(0) @binding(0) var<storage,read_write> reserved: array<u32>; @compute @workgroup_size(1) fn main() {reserved[0] = 0u;}',
            shaderLayout: {bindings: [{name: 'reserved', group: 0, location: 0, type: 'storage'}]}
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {reserved: getBuffer(reserved.view.buffer)},
                x: 1
              }),
            destroy: () => kernel.destroy()
          };
        }
      });
      for (const [index, input] of ordered.data.entries()) {
        if (!input.length) continue;
        const valid = validity?.data[index];
        const mask = selected?.data[index];
        const output = present.data[index];
        const dispatch = getBoundedDispatchLayout(
          'IntegerChartDomain',
          input.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `domain-validity-${index}`,
          resources: [
            {buffer: input, usage: 'storage-read'},
            ...(valid ? [{buffer: valid, usage: 'storage-read' as const}] : []),
            ...(mask ? [{buffer: mask, usage: 'storage-read' as const}] : []),
            {buffer: output, usage: 'storage-write'},
            {buffer: reserved.view, usage: 'storage-read-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> input: array<u32>;
@group(0) @binding(1) var<storage,read_write> output: array<u32>;
@group(0) @binding(2) var<storage,read_write> reserved: array<atomic<u32>>;
${valid ? '@group(0) @binding(3) var<storage,read> valid: array<u32>;' : ''}
${mask ? '@group(0) @binding(4) var<storage,read> selected: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if(index >= ${input.length}u) {return;}
  let present = ${valid ? `valid[${valid.byteOffset / 4}u + index] != 0u` : 'true'} && ${mask ? `selected[${mask.byteOffset / 4}u + index] != 0u` : 'true'};
  let special = input[${input.byteOffset / 4}u + index] == 0xffffffffu;
  if(present && special) {atomicOr(&reserved[0], 1u);}
  output[index] = select(0u, 1u, present && !special);
}`,
              shaderLayout: {
                bindings: [
                  {name: 'input', location: 0, group: 0, type: 'storage'},
                  {name: 'output', location: 1, group: 0, type: 'storage'},
                  {name: 'reserved', location: 2, group: 0, type: 'storage'},
                  ...(valid
                    ? [{name: 'valid', location: 3, group: 0, type: 'storage' as const}]
                    : []),
                  ...(mask
                    ? [{name: 'selected', location: 4, group: 0, type: 'storage' as const}]
                    : [])
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    input: getBuffer(input.buffer),
                    output: getBuffer(output.buffer),
                    reserved: getBuffer(reserved.view.buffer),
                    ...(valid ? {valid: getBuffer(valid.buffer)} : {}),
                    ...(mask ? {selected: getBuffer(mask.buffer)} : {})
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      graph.add(
        new GPUBatchHashIndex({
          keys: ordered,
          validity: present,
          tableKeys: tableKeys.view,
          tableValues: createTransientView(graph, 'first-rows', 'uint32', INTEGER_DOMAIN_CAPACITY),
          statistics: statistics.view
        })
      );
      this.compiled = graph.compile();
      this.keys = tableKeys.buffer;
      this.statistics = statistics.buffer;
      this.reserved = reserved.buffer;
      this.byteLength =
        buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
        this.compiled.stats.physicalTransientResourceBytes;
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      throw error;
    }
  }
  encode(encoder: CommandEncoder): void {
    this.compiled.encode(encoder, {parameters: undefined});
  }
  async read(): Promise<{keys: number[]; overflow: boolean}> {
    const [keyBytes, statisticBytes, reservedBytes] = await Promise.all([
      this.keys.readAsync(),
      this.statistics.readAsync(),
      this.reserved.readAsync()
    ]);
    const keys = [
      ...new Uint32Array(keyBytes.buffer, keyBytes.byteOffset, INTEGER_DOMAIN_CAPACITY)
    ].filter(key => key !== 0xffffffff);
    if (new DataView(reservedBytes.buffer, reservedBytes.byteOffset, 4).getUint32(0, true))
      keys.push(0xffffffff);
    const statistics = new DataView(statisticBytes.buffer, statisticBytes.byteOffset, 24);
    return {
      keys: keys.sort((first, second) => first - second),
      overflow: statistics.getUint32(8, true) !== 0 || statistics.getUint32(12, true) !== 0
    };
  }
  /** Caller has awaited reads and GPU completion before releasing borrowed source/scope buffers. */
  destroy(): void {
    this.compiled.destroy();
    this.keys.destroy();
    this.statistics.destroy();
    this.reserved.destroy();
  }
}
