// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import type {GPUKeplerTimeUnit} from '@luma-contributors/kepler';
import {makeResidentCalendarTimeView} from './resident-calendar-time';

/** Bytes read back per extent: biased minimum words, biased maximum words, row count, padding. */
export const TIME_EXTENT_READ_BYTES = 24;
const MAXIMUM_DATE_MILLISECONDS = 8640000000000000n;
const NATIVE_TARGET_TIME_BUCKETS = 200;
const HOUR = 3600000;
const DAY = 24 * HOUR;
const CANDIDATES = [
  ['hour', HOUR],
  ['day', DAY],
  ['week', 7 * DAY],
  ['month', 30 * DAY],
  ['year', 365 * DAY]
] as const;

/** Mirrors the pinned native chooseTimeInterval(spanMs, 200); an empty or zero span is day. */
export function chooseNativeTimeInterval(
  spanMilliseconds: number
): 'hour' | 'day' | 'week' | 'month' | 'year' {
  if (!(spanMilliseconds > 0)) return 'day';
  for (const [name, size] of CANDIDATES)
    if (spanMilliseconds / size <= NATIVE_TARGET_TIME_BUCKETS) return name;
  return 'year';
}

/**
 * Exact Int64 minimum and maximum of the calendar population as integer Date milliseconds.
 * The population matches the calendar key domain: selected rows whose valid instant is inside
 * Date's range, plus null rows, which native `new Date(null)` places at the epoch. Words never
 * pass through Float32. The minimum and maximum use two atomic phases (high word, then low word
 * among rows equal to the winning high word), so only 24 bytes are read back.
 */
export class ResidentTimeExtent {
  readonly byteLength: number;
  private readonly compiled: CompiledGPUCommandGraph;
  private readonly output: Buffer;
  constructor(props: {
    device: Device;
    timestamps: GPUVector<'uint32x2'>;
    validity: GPUVector<'uint32'>;
    selected?: GPUVector<'uint32'>;
    unit: GPUKeplerTimeUnit;
    sourceType: string;
  }) {
    const {device} = props;
    this.output = device.createBuffer({
      id: 'time-extent',
      byteLength: TIME_EXTENT_READ_BYTES,
      usage: Buffer.STORAGE | Buffer.COPY_SRC
    });
    try {
      const graph = new GPUCommandGraph(device, {id: 'chart-time-extent'});
      const output = graph.createDataView(
        graph.importBuffer(
          {id: 'extent', byteLength: this.output.byteLength, usage: this.output.usage},
          this.output
        ),
        {format: 'uint32', length: 6}
      );
      const timestamps = graph.importGPUVector('times', props.timestamps);
      const validity = graph.importGPUVector('validity', props.validity);
      const selected = props.selected
        ? graph.importGPUVector('selected', props.selected)
        : undefined;
      graph.addComputePass({
        id: 'reset-extent',
        resources: [{buffer: output, usage: 'storage-write'}],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            source:
              '@group(0) @binding(0) var<storage,read_write> extent: array<u32>; @compute @workgroup_size(1) fn main() {extent[0] = 0xffffffffu; extent[1] = 0xffffffffu; extent[2] = 0u; extent[3] = 0u; extent[4] = 0u; extent[5] = 0u;}',
            shaderLayout: {bindings: [{name: 'extent', group: 0, location: 0, type: 'storage'}]}
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {bindings: {extent: getBuffer(output.buffer)}, x: 1}),
            destroy: () => kernel.destroy()
          };
        }
      });
      const minimumHigh = Number(
        (BigInt.asUintN(64, -MAXIMUM_DATE_MILLISECONDS) >> 32n) ^ 0x80000000n
      );
      const minimumLow = Number(BigInt.asUintN(64, -MAXIMUM_DATE_MILLISECONDS) & 0xffffffffn);
      const maximumHigh = Number((MAXIMUM_DATE_MILLISECONDS >> 32n) ^ 0x80000000n);
      const maximumLow = Number(MAXIMUM_DATE_MILLISECONDS & 0xffffffffn);
      // Phase one finds the extreme biased high words; phase two refines the low words.
      for (const phase of [1, 2] as const)
        for (const [chunkIndex, chunk] of timestamps.data.entries()) {
          if (!chunk.length) continue;
          const source = makeResidentCalendarTimeView(graph, {
            id: `extent-time-${phase}-${chunkIndex}`,
            timestamps: chunk,
            unit: props.unit,
            sourceType: props.sourceType
          });
          const valid = validity.data[chunkIndex];
          const mask = selected?.data[chunkIndex];
          const dispatch = getBoundedDispatchLayout(
            'CalendarTimeExtent',
            source.length,
            256,
            device.limits.maxComputeWorkgroupsPerDimension
          );
          graph.addComputePass({
            id: `extent-${phase}-${chunkIndex}`,
            resources: [
              {buffer: source, usage: 'storage-read'},
              {buffer: valid, usage: 'storage-read'},
              ...(mask ? [{buffer: mask, usage: 'storage-read' as const}] : []),
              {buffer: output, usage: 'storage-read-write'}
            ],
            compile: ({device: compileDevice}) => {
              const kernel = new Kernel(compileDevice, {
                source: `
@group(0) @binding(0) var<storage,read> times: array<u32>;
@group(0) @binding(1) var<storage,read> valid: array<u32>;
@group(0) @binding(2) var<storage,read_write> extent: array<atomic<u32>>;
${mask ? '@group(0) @binding(3) var<storage,read> selected: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${source.length}u) {return;}
  if (${mask ? `selected[${mask.byteOffset / 4}u + index] == 0u` : 'false'}) {return;}
  let present = valid[${valid.byteOffset / 4}u + index] != 0u;
  let timeIndex = ${source.byteOffset / 4}u + index * 2u;
  // Null rows are native epoch-zero points; valid out-of-range instants are invalid dates.
  let instant = select(vec2<u32>(0u), vec2<u32>(times[timeIndex], times[timeIndex + 1u]), present);
  let high = instant.y ^ 0x80000000u;
  let inRange = (high > ${minimumHigh}u || (high == ${minimumHigh}u && instant.x >= ${minimumLow}u)) && (high < ${maximumHigh}u || (high == ${maximumHigh}u && instant.x <= ${maximumLow}u));
  if (!inRange) {return;}
  ${
    phase === 1
      ? `atomicMin(&extent[0], high);
  atomicMax(&extent[2], high);
  atomicAdd(&extent[4], 1u);`
      : `if (high == atomicLoad(&extent[0])) {atomicMin(&extent[1], instant.x);}
  if (high == atomicLoad(&extent[2])) {atomicMax(&extent[3], instant.x);}`
  }
}`,
                shaderLayout: {
                  bindings: [
                    {name: 'times', location: 0, group: 0, type: 'storage'},
                    {name: 'valid', location: 1, group: 0, type: 'storage'},
                    {name: 'extent', location: 2, group: 0, type: 'storage'},
                    ...(mask
                      ? [{name: 'selected', location: 3, group: 0, type: 'storage' as const}]
                      : [])
                  ]
                }
              });
              return {
                encode: ({computePass, getBuffer}) =>
                  kernel.dispatch(computePass, {
                    bindings: {
                      times: getBuffer(source.buffer),
                      valid: getBuffer(valid.buffer),
                      extent: getBuffer(output.buffer),
                      ...(mask ? {selected: getBuffer(mask.buffer)} : {})
                    },
                    ...dispatch
                  }),
                destroy: () => kernel.destroy()
              };
            }
          });
        }
      this.compiled = graph.compile();
      this.byteLength = this.output.byteLength + this.compiled.stats.physicalTransientResourceBytes;
    } catch (error) {
      this.output.destroy();
      throw error;
    }
  }
  encode(encoder: CommandEncoder): void {
    this.compiled.encode(encoder, {parameters: undefined});
  }
  /** Returns null when no row participates; otherwise integer millisecond extrema. */
  async read(): Promise<{minimum: number; maximum: number; count: number} | null> {
    const bytes = await this.output.readAsync(0, TIME_EXTENT_READ_BYTES);
    const words = new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 24));
    if (!words[4]) return null;
    const decode = (high: number, low: number) =>
      Number(BigInt.asIntN(64, (BigInt((high ^ 0x80000000) >>> 0) << 32n) | BigInt(low)));
    return {
      minimum: decode(words[0], words[1]),
      maximum: decode(words[2], words[3]),
      count: words[4]
    };
  }
  /** Caller has awaited reads and GPU completion before releasing borrowed source/scope buffers. */
  destroy(): void {
    this.compiled.destroy();
    this.output.destroy();
  }
}
