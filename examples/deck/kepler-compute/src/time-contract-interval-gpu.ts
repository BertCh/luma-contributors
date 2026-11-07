// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {TIME_WORDS_WGSL} from '@luma-contributors/analysis/dataframe-analysis';
import {
  DEFAULT_INTERVAL_BOUNDARY_CLOSURE,
  makeIntervalOverlapWgsl,
  splitInt64Words,
  type IntervalBoundaryClosure,
  type IntervalWindow
} from './time-contract-interval';

export type ContractIntervalSelectionProps = {
  device: Device;
  id?: string;
  /** Borrowed Int64 start words (low, high). A null start (validity 0) never matches. */
  start: GPUVector<'uint32x2'>;
  /** Borrowed Int64 end words with the same chunk topology. Omit for point/open-ended rows. */
  end?: GPUVector<'uint32x2'>;
  startValidity?: GPUVector<'uint32'>;
  /** Validity 0 means the row is open ended (+infinity), unlike GPUKeplerTimeSelection. */
  endValidity?: GPUVector<'uint32'>;
  window: IntervalWindow;
  closure?: IntervalBoundaryClosure;
};

/**
 * Interval selection executing the generated time-contract-interval WGSL predicate (all 16
 * closure combinations, null start refusal, open-ended rows, reversed rows refused). The mask is a
 * GPUVector<'uint32'> with the source chunk topology, so it can serve as an external dimension mask
 * exactly like GPUKeplerTimeSelection.mask.
 */
export class ContractIntervalSelection {
  readonly mask: GPUVector<'uint32'>;
  readonly rowCount: number;
  private readonly graph: GPUCommandGraph;
  private readonly windowBuffer: Buffer;
  private windowWords: Uint32Array;
  private compiled?: CompiledGPUCommandGraph;
  private readonly device: Device;
  private destroyed = false;
  private emptyFlag = 0;

  constructor(props: ContractIntervalSelectionProps) {
    const {device, start} = props;
    const id = props.id ?? 'contract-interval-selection';
    this.device = device;
    this.rowCount = start.length;
    this.windowWords = makeWindowWords(props.window);
    this.windowBuffer = device.createBuffer({
      id: `${id}-window`,
      data: this.windowWords,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.mask = new GPUVector({
      type: 'data',
      name: `${id}-mask`,
      format: 'uint32',
      ownsData: true,
      data: start.data.map(
        chunk =>
          new GPUData({
            buffer: device.createBuffer({
              id: `${id}-mask-chunk`,
              byteLength: Math.max(4, chunk.length * 4),
              usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
            }),
            format: 'uint32',
            length: chunk.length,
            ownsBuffer: true
          })
      )
    });
    this.graph = new GPUCommandGraph(device, {id});
    const windowView = this.graph.createDataView(
      this.graph.importBuffer(
        {id: `${id}-window`, byteLength: 32, usage: this.windowBuffer.usage},
        this.windowBuffer
      ),
      {format: 'uint32', length: 8}
    );
    const startView = this.graph.importGPUVector(`${id}-start`, start);
    const endView = props.end ? this.graph.importGPUVector(`${id}-end`, props.end) : undefined;
    const startValidityView = props.startValidity
      ? this.graph.importGPUVector(`${id}-start-validity`, props.startValidity)
      : undefined;
    const endValidityView = props.endValidity
      ? this.graph.importGPUVector(`${id}-end-validity`, props.endValidity)
      : undefined;
    const maskView = this.graph.importGPUVector(`${id}-mask`, this.mask);
    const closure = props.closure ?? DEFAULT_INTERVAL_BOUNDARY_CLOSURE;
    for (const [chunkIndex, chunk] of startView.data.entries()) {
      if (!chunk.length) continue;
      const endChunk = endView?.data[chunkIndex];
      const startValid = startValidityView?.data[chunkIndex];
      const endValid = endValidityView?.data[chunkIndex];
      const output = maskView.data[chunkIndex];
      const dispatch = getBoundedDispatchLayout(
        'ContractInterval',
        chunk.length,
        256,
        device.limits.maxComputeWorkgroupsPerDimension
      );
      this.graph.addComputePass({
        id: `${id}-chunk-${chunkIndex}`,
        resources: [
          {buffer: chunk, usage: 'storage-read'},
          ...(endChunk ? [{buffer: endChunk, usage: 'storage-read' as const}] : []),
          ...(startValid ? [{buffer: startValid, usage: 'storage-read' as const}] : []),
          ...(endValid ? [{buffer: endValid, usage: 'storage-read' as const}] : []),
          {buffer: windowView, usage: 'storage-read'},
          {buffer: output, usage: 'storage-write'}
        ],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            source: `
${TIME_WORDS_WGSL}
${makeIntervalOverlapWgsl('isIntervalOverlappingWindow', closure)}
@group(0) @binding(0) var<storage, read> startWords: array<u32>;
@group(0) @binding(1) var<storage, read> windowWords: array<u32>;
@group(0) @binding(2) var<storage, read_write> maskOut: array<u32>;
${endChunk ? '@group(0) @binding(3) var<storage, read> endWords: array<u32>;' : ''}
${startValid ? '@group(0) @binding(4) var<storage, read> startValidity: array<u32>;' : ''}
${endValid ? '@group(0) @binding(5) var<storage, read> endValidity: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${chunk.length}u) {return;}
  let startOffset = ${chunk.byteOffset / 4}u + index * 2u;
  let start = vec2<u32>(startWords[startOffset], startWords[startOffset + 1u]);
  ${
    endChunk
      ? `let endOffset = ${endChunk.byteOffset / 4}u + index * 2u;
  var end = vec2<u32>(endWords[endOffset], endWords[endOffset + 1u]);
  var hasEnd = true;`
      : 'var end = vec2<u32>(0u, 0u);\n  var hasEnd = false;'
  }
  ${endValid ? `if (endValidity[${endValid.byteOffset / 4}u + index] == 0u) { hasEnd = false; }` : ''}
  let startPresent = ${startValid ? `startValidity[${startValid.byteOffset / 4}u + index] != 0u` : 'true'};
  let windowStart = vec2<u32>(windowWords[0], windowWords[1]);
  let windowEnd = vec2<u32>(windowWords[2], windowWords[3]);
  let selected = windowWords[4] == 0u && startPresent && isIntervalOverlappingWindow(start, end, hasEnd, windowStart, windowEnd);
  maskOut[${output.byteOffset / 4}u + index] = select(0u, 1u, selected);
}`,
            shaderLayout: {
              bindings: [
                {name: 'startWords', location: 0, group: 0, type: 'storage'},
                {name: 'windowWords', location: 1, group: 0, type: 'storage'},
                {name: 'maskOut', location: 2, group: 0, type: 'storage'},
                ...(endChunk
                  ? [{name: 'endWords', location: 3, group: 0, type: 'storage' as const}]
                  : []),
                ...(startValid
                  ? [{name: 'startValidity', location: 4, group: 0, type: 'storage' as const}]
                  : []),
                ...(endValid
                  ? [{name: 'endValidity', location: 5, group: 0, type: 'storage' as const}]
                  : [])
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {
                  startWords: getBuffer(chunk.buffer),
                  windowWords: getBuffer(windowView.buffer),
                  maskOut: getBuffer(output.buffer),
                  ...(endChunk ? {endWords: getBuffer(endChunk.buffer)} : {}),
                  ...(startValid ? {startValidity: getBuffer(startValid.buffer)} : {}),
                  ...(endValid ? {endValidity: getBuffer(endValid.buffer)} : {})
                },
                ...dispatch
              }),
            destroy: () => kernel.destroy()
          };
        }
      });
    }
  }

  /** Takes effect at the next encode. */
  setWindow(window: IntervalWindow): void {
    this.windowWords = makeWindowWords(window, this.emptyFlag);
  }

  /** Forces an empty mask regardless of the window (a reversed native window selects nothing). */
  setEmpty(isEmpty: boolean): void {
    this.emptyFlag = isEmpty ? 1 : 0;
    this.windowWords[4] = this.emptyFlag;
  }

  prepare(): void {
    this.compiled ??= this.graph.compile();
  }

  encode(commandEncoder: CommandEncoder): void {
    this.prepare();
    this.device.writeBufferViaCommandEncoder(commandEncoder, this.windowBuffer, this.windowWords);
    this.compiled!.encode(commandEncoder, {parameters: undefined});
  }

  /** Test and diagnostic readback of the concatenated 0/1 mask. */
  async readMask(): Promise<Uint8Array> {
    const result = new Uint8Array(this.rowCount);
    let offset = 0;
    for (const chunk of this.mask.data) {
      if (!chunk.length) continue;
      const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
      const words = new Uint32Array(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
      );
      for (let index = 0; index < chunk.length; index++) result[offset + index] = words[index];
      offset += chunk.length;
    }
    return result;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled?.destroy();
    this.windowBuffer.destroy();
    this.mask.destroy();
  }
}

function makeWindowWords(window: IntervalWindow, emptyFlag = 0): Uint32Array {
  return Uint32Array.from([
    ...splitInt64Words(window.start),
    ...splitInt64Words(window.end),
    emptyFlag,
    0,
    0,
    0
  ]);
}

/** The part of GPUKeplerTimeSelection that TimeRuntime consumes. */
export type TimeSelectionLike = {
  readonly mask: GPUVector<'uint32'>;
  setWindow(window: {start: number | bigint; end: number | bigint; unit: string}): unknown;
  encode(commandEncoder: CommandEncoder): unknown;
  readStatus(): Promise<{
    revision: number;
    stale: boolean;
    selectedCount: number;
    overflow: number;
    complete: boolean;
  }>;
  readonly accountedResidentByteLength: number;
  readonly readbackByteLength: number;
  destroy(): void;
};

/**
 * Adapts ContractIntervalSelection to the TimeRuntime selection surface so a native time control
 * can filter rows that are intervals (start and optional end columns) with contract closure,
 * open-ended rows and reversed-interval refusal. Windows are inclusive Int64 ticks in the source
 * unit; an empty window (end < start) selects nothing.
 */
export function makeContractIntervalTimeSelection(
  props: ContractIntervalSelectionProps
): TimeSelectionLike {
  const selection = new ContractIntervalSelection(props);
  let revision = 0;
  let encodedRevision = -1;
  let diagnosticBytes = 0;
  let emptyWindow = false;
  return {
    mask: selection.mask,
    setWindow(window) {
      revision++;
      const start = BigInt(window.start);
      const end = BigInt(window.end);
      emptyWindow = end < start;
      // A reversed window would throw in the oracle; selecting nothing is a closed zero-width window
      // under an all-open closure, which the predicate rejects, so clamp and force emptiness below.
      selection.setWindow(emptyWindow ? {start, end: start} : {start, end});
      selection.setEmpty(emptyWindow);
    },
    encode(commandEncoder) {
      selection.encode(commandEncoder);
      encodedRevision = revision;
      return encodedRevision;
    },
    async readStatus() {
      const mask = await selection.readMask();
      diagnosticBytes += mask.length * 4;
      let selectedCount = 0;
      for (const value of mask) selectedCount += value;
      return {
        revision: encodedRevision,
        stale: encodedRevision !== revision,
        selectedCount,
        overflow: 0,
        complete: encodedRevision === revision
      };
    },
    get accountedResidentByteLength() {
      return selection.rowCount * 4 + 16;
    },
    get readbackByteLength() {
      return diagnosticBytes;
    },
    destroy: () => selection.destroy()
  };
}
