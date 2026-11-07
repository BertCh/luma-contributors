// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUGather, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import type {PreparedNativeIcons} from './icon-source';
import type {ResidentIconBinding} from './resident-layers';

export type NativeIconGPUGroup = {
  iconId: string;
  iconGeometry: GPUData<'float32x3'>;
  sourceRowIds: GPUVector<'uint32'>;
  resolveSourceRow(primitiveIndex: number): number;
};
export type NativeIconGPUVectors = {
  source: PreparedNativeIcons;
  groups: readonly NativeIconGPUGroup[];
  gpuByteLength: number;
  isCurrent(): boolean;
  destroy(completion: PromiseLike<unknown>): Promise<void>;
};
export type ResidentIconColumns = {lat: string; lng: string; icon: string; altitude?: string};

/** Upload exact native meshes and immutable row maps once; source batch boundaries are retained. */
export function makeGPUVectorsFromNativeIcons(
  device: Device,
  source: PreparedNativeIcons,
  options: {
    workingMemoryBudgetBytes: number;
    memoryRegistry?: KeplerDatasetRegistry;
  }
): NativeIconGPUVectors {
  const byteLength = source.groups.reduce(
    (total, group) =>
      total +
      group.vertices.byteLength +
      group.sourceRowIds.reduce((sum, chunk) => sum + Math.max(4, chunk.byteLength), 0),
    0
  );
  if (
    !Number.isSafeInteger(options.workingMemoryBudgetBytes) ||
    options.workingMemoryBudgetBytes < 0 ||
    byteLength > options.workingMemoryBudgetBytes
  )
    throw new Error('Native icon upload exceeds its admitted memory budget');
  const cpuReservation = options.memoryRegistry?.reserveMemory(
    source.typedStorageByteLength,
    'cpu',
    'source'
  );
  let reservation: KeplerMemoryReservation | undefined;
  const deviceLoss = device.lost;
  const buffers: Buffer[] = [];
  const allocate = (data: Float32Array | Uint32Array): Buffer => {
    const buffer = device.createBuffer({
      ...(data.byteLength ? {data} : {byteLength: 4}),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  };
  try {
    reservation = options.memoryRegistry?.reserveMemory(byteLength, 'gpu', 'source');
    const groups = source.groups.map(group => ({
      iconId: group.iconId,
      iconGeometry: new GPUData({
        format: 'float32x3',
        buffer: allocate(group.vertices),
        length: group.vertices.length / 3,
        ownsBuffer: false
      }),
      sourceRowIds: new GPUVector({
        type: 'data',
        name: `icon-${group.iconId}-source-rows`,
        format: 'uint32',
        ownsData: false,
        data: group.sourceRowIds.map(
          indices =>
            new GPUData({
              format: 'uint32',
              buffer: allocate(indices),
              length: indices.length,
              ownsBuffer: false
            })
        )
      }),
      resolveSourceRow: group.resolveSourceRow
    }));
    let current = true;
    let destruction: Promise<void> | undefined;
    return {
      source,
      groups,
      gpuByteLength: byteLength,
      isCurrent: () => current,
      destroy(completion) {
        if (!destruction) {
          current = false;
          destruction = (
            deviceLoss
              ? Promise.race([Promise.resolve(completion), deviceLoss])
              : Promise.resolve(completion)
          )
            .then(() => undefined)
            .finally(() => {
              for (const buffer of buffers) buffer.destroy();
              reservation?.release();
              cpuReservation?.release();
            });
        }
        return destruction;
      }
    };
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    reservation?.release();
    cpuReservation?.release();
    throw error;
  }
}

export type IconRuntimeProps = {
  device: Device;
  columns: ResidentIconColumns;
  icons: NativeIconGPUVectors;
  positions: GPUVector<'float32x2' | 'float32x3'>;
  selectionMask: GPUVector<'uint32'>;
  colors?: GPUVector<'unorm8x4'>;
  radii?: GPUVector<'float32'>;
  workingMemoryBudgetBytes: number;
  memoryRegistry?: KeplerDatasetRegistry;
};

/** GPU gathers replace native per-interaction row grouping and accessor scans. Inputs are borrowed. */
export class IconRuntime {
  readonly binding: ResidentIconBinding;
  private readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly reservation?: KeplerMemoryReservation;
  private readonly deviceLoss: Device['lost'];
  private compiled?: CompiledGPUCommandGraph;
  private destroyed = false;
  private released = false;
  private destruction?: Promise<void>;

  constructor(private readonly props: IconRuntimeProps) {
    this.deviceLoss = props.device.lost;
    // Source identities, device ownership, and fixed-width formats are checked before allocation.
    assert(props.icons.isCurrent());
    const channels = [props.positions, props.selectionMask, props.colors, props.radii].filter(
      vector => Boolean(vector)
    );
    for (const vector of channels) {
      assert(vector!.length === props.icons.source.rowCount);
      for (const chunk of vector!.data) assert(chunk.buffer.device === props.device);
    }
    for (const group of props.icons.groups) {
      assert(group.iconGeometry.buffer.device === props.device);
      for (const chunk of group.sourceRowIds.data) assert(chunk.buffer.device === props.device);
    }
    const outputBytes = props.icons.groups.reduce(
      (total, group) =>
        total +
        group.sourceRowIds.data.reduce(
          (sum, chunk) =>
            sum +
            channels.reduce(
              (bytes, vector) =>
                bytes +
                Math.max(4, chunk.length * getGPUVectorFormatInfo(vector!.format!).byteLength),
              0
            ),
          0
        ),
      0
    );
    if (
      !Number.isSafeInteger(props.workingMemoryBudgetBytes) ||
      outputBytes > props.workingMemoryBudgetBytes
    )
      throw new Error('Native icon gathers exceed their admitted memory budget');
    this.graph = new GPUCommandGraph(props.device, {id: 'native-icon-gathers'});
    this.reservation = props.memoryRegistry?.reserveMemory(
      props.workingMemoryBudgetBytes,
      'gpu',
      'scratch'
    );
    try {
      this.binding = {
        columns: props.columns,
        groups: props.icons.groups.map((group, index) => ({
          iconId: group.iconId,
          iconGeometry: group.iconGeometry,
          positions: this.gather(`positions-${index}`, props.positions, group.sourceRowIds),
          selectionMask: this.gather(`mask-${index}`, props.selectionMask, group.sourceRowIds),
          colors: props.colors
            ? this.gather(`colors-${index}`, props.colors, group.sourceRowIds)
            : undefined,
          radii: props.radii
            ? this.gather(`radii-${index}`, props.radii, group.sourceRowIds)
            : undefined,
          resolveSourceRow: group.resolveSourceRow
        }))
      };
    } catch (error) {
      this.release();
      throw error;
    }
  }
  prepare(): void {
    assert(!this.destroyed && this.props.icons.isCurrent());
    if (this.compiled) return;
    const compiled = this.graph.compile();
    if (
      this.ownedByteLength + compiled.stats.physicalTransientResourceBytes >
      this.props.workingMemoryBudgetBytes
    ) {
      compiled.destroy();
      throw new Error('Compiled native icon gathers exceed their admitted memory budget');
    }
    this.compiled = compiled;
  }
  encode(encoder: CommandEncoder): void {
    assert(encoder.device === this.props.device);
    this.prepare();
    this.compiled!.encode(encoder, {parameters: undefined});
  }
  get accountedResidentByteLength(): number {
    return this.released
      ? 0
      : this.ownedByteLength + (this.compiled?.stats.physicalTransientResourceBytes ?? 0);
  }
  get readbackByteLength(): number {
    return 0;
  }
  get resources(): readonly Buffer[] {
    return this.buffers;
  }
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.destruction) {
      this.destroyed = true;
      this.destruction = (
        this.deviceLoss
          ? Promise.race([Promise.resolve(completion), this.deviceLoss])
          : Promise.resolve(completion)
      )
        .then(() => undefined)
        .finally(() => this.release());
    }
    return this.destruction;
  }
  private get ownedByteLength(): number {
    return this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0);
  }
  private gather<Format extends 'float32x2' | 'float32x3' | 'float32' | 'uint32' | 'unorm8x4'>(
    name: string,
    source: GPUVector<Format>,
    indices: GPUVector<'uint32'>
  ): GPUVector<Format> {
    const format = source.format;
    assert(format);
    const rowByteLength = getGPUVectorFormatInfo(format).byteLength;
    const output = new GPUVector({
      type: 'data',
      name: `native-icon-${name}`,
      format,
      ownsData: false,
      data: indices.data.map(chunk => {
        const buffer = this.props.device.createBuffer({
          byteLength: Math.max(4, chunk.length * rowByteLength),
          usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        this.buffers.push(buffer);
        return new GPUData({format, buffer, length: chunk.length, ownsBuffer: false});
      })
    });
    this.graph.add(
      new GPUGather({
        id: name,
        source: this.graph.importGPUVector(`${name}-source`, source),
        indices: this.graph.importGPUVector(`${name}-indices`, indices),
        output: this.graph.importGPUVector(`${name}-output`, output)
      })
    );
    return output;
  }
  private release(): void {
    if (this.released) return;
    this.released = true;
    this.compiled?.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.reservation?.release();
  }
}
