// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma-contributors/graph/gpu-core';
import type {ResidentCategorySummary} from './resident-chart-bins';

/** Groups kept beside the "Other" bucket (native chart point cap 1,000 = 999 + Other). */
export const MAXIMUM_RESIDENT_TOP_GROUPS = 999;
/** Largest dictionary the top-N fallback counts on the GPU (one dense group per entry). */
export const MAXIMUM_RESIDENT_TOP_DICTIONARY = 262_144;
export const INVALID_RESIDENT_GROUP_KEY = 0xffffffff;

export type ResidentTopGroupSelection = {
  /** Dictionary code to chart bucket; absent values map to INVALID_RESIDENT_GROUP_KEY. */
  bucketOfCode: Uint32Array;
  /** Bucket labels in rank order; the last one is the Other bucket when `hasOther`. */
  labels: string[];
  hasOther: boolean;
  /** Dictionary entries folded into Other. */
  otherCodeCount: number;
};

/**
 * Rank the per-code summary (aggregate value, then first row) and keep the top `limit` groups; every
 * other present code folds into one "Other" bucket that the GPU then re-aggregates exactly.
 * `summaries[code]` must describe dictionary entry `code`.
 */
export function selectResidentTopGroups(props: {
  summaries: readonly Pick<ResidentCategorySummary, 'value' | 'count' | 'firstRow'>[];
  labels: readonly string[];
  limit: number;
}): ResidentTopGroupSelection {
  const present: number[] = [];
  for (let code = 0; code < props.summaries.length; code++)
    if (props.summaries[code].count > 0) present.push(code);
  present.sort(
    (first, second) =>
      props.summaries[second].value - props.summaries[first].value ||
      props.summaries[first].firstRow - props.summaries[second].firstRow ||
      first - second
  );
  const hasOther = present.length > props.limit;
  const kept = hasOther ? present.slice(0, props.limit) : present;
  const bucketOfCode = new Uint32Array(props.labels.length).fill(INVALID_RESIDENT_GROUP_KEY);
  const labels: string[] = [];
  for (const code of kept) {
    bucketOfCode[code] = labels.length;
    labels.push(props.labels[code]);
  }
  if (hasOther) {
    for (const code of present.slice(props.limit)) bucketOfCode[code] = labels.length;
    labels.push(labels.includes('Other') ? 'Other (remaining groups)' : 'Other');
  }
  return {
    bucketOfCode,
    labels,
    hasOther,
    otherCodeCount: hasOther ? present.length - props.limit : 0
  };
}

/** Cell budget leaves room for the series cross product; at least one kept group remains. */
export function getResidentTopGroupLimit(
  seriesCategoryCount: number | undefined,
  cells: number
): number {
  if (seriesCategoryCount === undefined) return MAXIMUM_RESIDENT_TOP_GROUPS;
  return Math.max(
    1,
    Math.min(MAXIMUM_RESIDENT_TOP_GROUPS, Math.floor(cells / (seriesCategoryCount + 1)) - 1)
  );
}

/** Chunk-preserving derived bucket keys: keys[i] = bucketOfCode[codes[i]], invalid when missing. */
export class ResidentGroupRemapKeys {
  readonly keys: GPUVector<'uint32'>;
  readonly byteLength: number;
  private readonly compiled: CompiledGPUCommandGraph;
  private readonly table: Buffer;

  constructor(props: {
    device: Device;
    categories: GPUVector;
    validity?: GPUVector<'uint32'>;
    bucketOfCode: Uint32Array;
  }) {
    const {device} = props;
    this.table = device.createBuffer({
      id: 'chart-top-group-table',
      byteLength: Math.max(4, props.bucketOfCode.byteLength),
      usage: Buffer.STORAGE | Buffer.COPY_DST,
      data: props.bucketOfCode
    });
    const chunks: GPUData<'uint32'>[] = [];
    try {
      for (const chunk of props.categories.data)
        chunks.push(
          new GPUData({
            format: 'uint32',
            length: chunk.length,
            ownsBuffer: true,
            buffer: device.createBuffer({
              id: 'chart-top-group-keys',
              byteLength: Math.max(4, chunk.length * 4),
              usage: Buffer.STORAGE
            })
          })
        );
      this.keys = new GPUVector({
        name: 'chart-top-group-keys',
        type: 'data',
        format: 'uint32',
        data: chunks,
        ownsData: true
      });
      const graph = new GPUCommandGraph(device, {id: 'chart-top-group-keys'});
      const categories = graph.importGPUVector('categories', props.categories);
      const validity = props.validity
        ? graph.importGPUVector('validity', props.validity)
        : undefined;
      const output = graph.importGPUVector('output', this.keys);
      const table = graph.createDataView(
        graph.importBuffer(
          {id: 'top-group-table', byteLength: this.table.byteLength, usage: this.table.usage},
          this.table
        ),
        {format: 'uint32', length: Math.max(1, props.bucketOfCode.length)}
      );
      const codeCount = props.bucketOfCode.length;
      for (const [chunkIndex, category] of categories.data.entries()) {
        const valid = validity?.data[chunkIndex];
        if (valid && valid.length !== category.length)
          throw new Error('Chart top groups must preserve source chunk boundaries');
        if (!category.length) continue;
        const target = output.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          'ChartTopGroupKeys',
          category.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `top-group-keys-${chunkIndex}`,
          resources: [
            {buffer: category, usage: 'storage-read'},
            {buffer: table, usage: 'storage-read'},
            ...(valid ? [{buffer: valid, usage: 'storage-read' as const}] : []),
            {buffer: target, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> categories: array<u32>;
@group(0) @binding(1) var<storage,read> table: array<u32>;
@group(0) @binding(2) var<storage,read_write> output: array<u32>;
${valid ? '@group(0) @binding(3) var<storage,read> valid: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${category.length}u) {return;}
  let code = categories[${category.byteOffset / 4}u + index];
  let present = code < ${codeCount}u ${valid ? `&& valid[${valid.byteOffset / 4}u + index] != 0u` : ''};
  output[index] = select(0xffffffffu, table[select(0u, code, present)], present);
}`,
              shaderLayout: {
                bindings: [
                  {name: 'categories', location: 0, group: 0, type: 'storage'},
                  {name: 'table', location: 1, group: 0, type: 'storage'},
                  {name: 'output', location: 2, group: 0, type: 'storage'},
                  ...(valid
                    ? [{name: 'valid', location: 3, group: 0, type: 'storage' as const}]
                    : [])
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    categories: getBuffer(category.buffer),
                    table: getBuffer(table.buffer),
                    output: getBuffer(target.buffer),
                    ...(valid ? {valid: getBuffer(valid.buffer)} : {})
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      this.compiled = graph.compile();
      this.byteLength =
        this.table.byteLength +
        chunks.reduce(
          (total, chunk) =>
            total +
            (chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer).byteLength,
          0
        ) +
        this.compiled.stats.physicalTransientResourceBytes;
    } catch (error) {
      for (const chunk of chunks) chunk.destroy();
      this.table.destroy();
      throw error;
    }
  }
  encode(encoder: CommandEncoder): void {
    this.compiled.encode(encoder, {parameters: undefined});
  }
  destroy(): void {
    this.compiled.destroy();
    this.keys.destroy();
    this.table.destroy();
  }
}
