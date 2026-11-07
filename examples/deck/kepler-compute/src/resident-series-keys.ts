// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, getBoundedDispatchLayout, getBoundedInvocationIndexSource, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';

/** Chunk-preserving composite keys, with an extra series slot retaining rows without a series. */
export class ResidentSeriesKeys {
  readonly keys: GPUVector<'uint32'>;
  readonly byteLength: number;
  private readonly compiled: CompiledGPUCommandGraph;

  constructor(props: {
    device: Device;
    categories: GPUVector;
    categoryCount: number;
    categoryValidity?: GPUVector<'uint32'>;
    series: GPUVector;
    seriesLabels: readonly string[];
    seriesValidity?: GPUVector<'uint32'>;
  }) {
    const {device} = props;
    const chunks: GPUData<'uint32'>[] = [];
    try {
      for (const chunk of props.categories.data)
        chunks.push(
          new GPUData({
            format: 'uint32',
            length: chunk.length,
            ownsBuffer: true,
            buffer: device.createBuffer({
              id: 'chart-series-keys',
              byteLength: Math.max(4, chunk.length * 4),
              usage: Buffer.STORAGE
            })
          })
        );
      this.keys = new GPUVector({
        name: 'chart-series-keys',
        type: 'data',
        format: 'uint32',
        data: chunks,
        ownsData: true
      });
      const graph = new GPUCommandGraph(device, {id: 'chart-series-keys'});
      const categories = graph.importGPUVector('categories', props.categories);
      const series = graph.importGPUVector('series', props.series);
      const categoryValidity = props.categoryValidity
        ? graph.importGPUVector('category-validity', props.categoryValidity)
        : undefined;
      const seriesValidity = props.seriesValidity
        ? graph.importGPUVector('series-validity', props.seriesValidity)
        : undefined;
      const output = graph.importGPUVector('output', this.keys);
      const seriesCount = props.seriesLabels.length;
      const emptySeries = props.seriesLabels.indexOf('');
      for (const [chunkIndex, category] of categories.data.entries()) {
        const seriesChunk = series.data[chunkIndex];
        const categoryValid = categoryValidity?.data[chunkIndex];
        const seriesValid = seriesValidity?.data[chunkIndex];
        if (
          !seriesChunk ||
          seriesChunk.length !== category.length ||
          (categoryValid && categoryValid.length !== category.length) ||
          (seriesValid && seriesValid.length !== category.length)
        )
          throw new Error('Chart series must preserve source chunk boundaries');
        if (!category.length) continue;
        const target = output.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          'ChartSeriesKeys',
          category.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `series-keys-${chunkIndex}`,
          resources: [
            {buffer: category, usage: 'storage-read'},
            {buffer: seriesChunk, usage: 'storage-read'},
            ...(categoryValid ? [{buffer: categoryValid, usage: 'storage-read' as const}] : []),
            ...(seriesValid ? [{buffer: seriesValid, usage: 'storage-read' as const}] : []),
            {buffer: target, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> categories: array<u32>;
@group(0) @binding(1) var<storage,read> series: array<u32>;
@group(0) @binding(2) var<storage,read_write> output: array<u32>;
${categoryValid ? '@group(0) @binding(3) var<storage,read> categoryValid: array<u32>;' : ''}
${seriesValid ? '@group(0) @binding(4) var<storage,read> seriesValid: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${category.length}u) {return;}
  let category = categories[${category.byteOffset / 4}u + index];
  let seriesKey = series[${seriesChunk.byteOffset / 4}u + index];
  let categoryPresent = category < ${props.categoryCount}u ${categoryValid ? `&& categoryValid[${categoryValid.byteOffset / 4}u + index] != 0u` : ''};
  let seriesPresent = seriesKey < ${seriesCount}u ${emptySeries >= 0 ? `&& seriesKey != ${emptySeries}u` : ''} ${seriesValid ? `&& seriesValid[${seriesValid.byteOffset / 4}u + index] != 0u` : ''};
  output[index] = select(0xffffffffu, category * ${seriesCount + 1}u + select(${seriesCount}u, seriesKey, seriesPresent), categoryPresent);
}`,
              shaderLayout: {
                bindings: [
                  {name: 'categories', location: 0, group: 0, type: 'storage'},
                  {name: 'series', location: 1, group: 0, type: 'storage'},
                  {name: 'output', location: 2, group: 0, type: 'storage'},
                  ...(categoryValid
                    ? [{name: 'categoryValid', location: 3, group: 0, type: 'storage' as const}]
                    : []),
                  ...(seriesValid
                    ? [{name: 'seriesValid', location: 4, group: 0, type: 'storage' as const}]
                    : [])
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    categories: getBuffer(category.buffer),
                    series: getBuffer(seriesChunk.buffer),
                    output: getBuffer(target.buffer),
                    ...(categoryValid ? {categoryValid: getBuffer(categoryValid.buffer)} : {}),
                    ...(seriesValid ? {seriesValid: getBuffer(seriesValid.buffer)} : {})
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
        chunks.reduce(
          (total, chunk) =>
            total +
            (chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer).byteLength,
          0
        ) + this.compiled.stats.physicalTransientResourceBytes;
    } catch (error) {
      for (const chunk of chunks) chunk.destroy();
      throw error;
    }
  }
  encode(encoder: CommandEncoder): void {
    this.compiled.encode(encoder, {parameters: undefined});
  }
  destroy(): void {
    this.compiled.destroy();
    this.keys.destroy();
  }
}
