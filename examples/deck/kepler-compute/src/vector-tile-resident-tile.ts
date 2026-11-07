// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU residency of one decoded vector tile.
 *
 * Geometry goes through the same one-time preparation and upload as resident GeoJSON sources
 * (`prepareGeometrySource` + `makeGeometryGPUVectors`). A filter + style compute kernel then writes
 * one packed mask word and one packed RGBA8 color per source feature, and the host's
 * `GeometryRuntime` gathers those into the point, path, polygon and polygon-outline primitive
 * spaces. The GPU layers read the gathered masks on every draw, so filtering is a GPU write of
 * the mask rather than a CPU re-selection of geometry.
 */

import {Buffer, type Device} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {prepareGeometrySource, makeGeometryGPUVectors} from '@luma.gl/engine/arrow';
import {GeometryRuntime} from './geometry-runtime';
import type {ResidentDatasetBinding} from './resident-layers';
import type {DecodedTileFeature, DecodedTileFilter} from './vector-tile-decoded-batch';
import {
  packVectorTileParameters,
  VECTOR_TILE_KERNEL_SOURCE,
  VECTOR_TILE_PARAMETER_WORD_COUNT,
  type VectorTileExtraChannels,
  type VectorTileGpuStyle
} from './vector-tile-resident-style';

const WORKGROUP_SIZE = 64;
const GEOMETRY_COLUMN = 'geometry';

export type VectorTileResidentTileOptions = {
  device: Device;
  tileKey: string;
  features: readonly DecodedTileFeature[];
  /** Float32 property columns uploaded for this tile; filters and styles address them by name. */
  fieldNames: readonly string[];
  filters: readonly DecodedTileFilter[];
  style: VectorTileGpuStyle;
  channels?: VectorTileExtraChannels;
  /**
   * Category codes (append-only, shared by the layer's tiles) of categorical property fields. A field
   * listed here is uploaded as its code (NaN when the label has no code) instead of a numeric value.
   */
  categories?: ReadonlyMap<string, ReadonlyMap<string | number, number>>;
};

export class VectorTileResidentTile {
  readonly tileKey: string;
  readonly rowCount: number;
  readonly fieldNames: readonly string[];
  readonly binding: NonNullable<ResidentDatasetBinding['geojson']>;
  readonly gpuByteLength: number;
  readonly featureTypes: {point: boolean; line: boolean; polygon: boolean};
  /** Rows whose geometry failed preparation; they stay as non-drawing source rows. */
  readonly invalidRowCount: number;
  private readonly device: Device;
  private readonly geometryRuntime: GeometryRuntime;
  private readonly computation: Computation;
  private readonly fieldIndexByName: Map<string, number>;
  private readonly buffers: Buffer[] = [];
  private readonly parametersBuffer: Buffer;
  private readonly maskBuffer: Buffer;
  private readonly colorBuffer: Buffer;
  private readonly strokeBuffer: Buffer;
  private readonly widthBuffer: Buffer;
  private readonly radiusBuffer: Buffer;
  private destroyed = false;
  private updateCount = 0;

  /** Returns null when the tile has no features. Throws when GPU upload or preparation fails. */
  static create(options: VectorTileResidentTileOptions): VectorTileResidentTile | null {
    return options.features.length ? new VectorTileResidentTile(options) : null;
  }

  private constructor(options: VectorTileResidentTileOptions) {
    const {device, features} = options;
    this.device = device;
    this.tileKey = options.tileKey;
    this.rowCount = features.length;
    this.fieldNames = options.fieldNames;
    this.fieldIndexByName = new Map(options.fieldNames.map((name, index) => [name, index]));
    const id = `kepler-vector-tile-${options.tileKey}`;
    const prepared = prepareGeometrySource([
      {id, rows: features.map(feature => feature.geometry ?? null)}
    ]);
    const geometry = makeGeometryGPUVectors(device, prepared);
    this.featureTypes = prepared.metadata.featureTypes;
    this.invalidRowCount = prepared.metadata.invalidRows.length;
    try {
      const columns = new Float32Array(Math.max(1, options.fieldNames.length * this.rowCount));
      options.fieldNames.forEach((name, fieldIndex) => {
        features.forEach((feature, row) => {
          const value = feature.properties?.[name];
          const codes = options.categories?.get(name);
          columns[fieldIndex * this.rowCount + row] = codes
            ? (codes.get(value as string | number) ?? Number.NaN)
            : typeof value === 'number'
              ? value
              : Number.NaN;
        });
      });
      const columnBuffer = this.allocate(`${id}-columns`, columns);
      this.parametersBuffer = this.allocate(
        `${id}-parameters`,
        new Uint32Array(VECTOR_TILE_PARAMETER_WORD_COUNT)
      );
      this.maskBuffer = this.allocate(`${id}-mask`, new Uint32Array(this.rowCount));
      this.colorBuffer = this.allocate(`${id}-colors`, new Uint32Array(this.rowCount));
      this.strokeBuffer = this.allocate(`${id}-stroke-colors`, new Uint32Array(this.rowCount));
      this.widthBuffer = this.allocate(`${id}-widths`, new Float32Array(this.rowCount));
      this.radiusBuffer = this.allocate(`${id}-radii`, new Float32Array(this.rowCount));
      const selectionMask = this.makeVector('uint32', this.maskBuffer);
      const colors = this.makeVector('unorm8x4', this.colorBuffer);
      const strokeColors = this.makeVector('unorm8x4', this.strokeBuffer);
      const widths = this.makeVector('float32', this.widthBuffer);
      const radii = this.makeVector('float32', this.radiusBuffer);
      this.computation = new Computation(device, {
        id: `${id}-filter-style`,
        source: VECTOR_TILE_KERNEL_SOURCE,
        shaderLayout: {
          bindings: [
            {name: 'parameters', type: 'read-only-storage', group: 0, location: 0},
            {name: 'columns', type: 'read-only-storage', group: 0, location: 1},
            {name: 'maskOutput', type: 'storage', group: 0, location: 2},
            {name: 'colorOutput', type: 'storage', group: 0, location: 3},
            {name: 'strokeOutput', type: 'storage', group: 0, location: 4},
            {name: 'widthOutput', type: 'storage', group: 0, location: 5},
            {name: 'radiusOutput', type: 'storage', group: 0, location: 6}
          ]
        },
        bindings: {
          parameters: this.parametersBuffer,
          columns: columnBuffer,
          maskOutput: this.maskBuffer,
          colorOutput: this.colorBuffer,
          strokeOutput: this.strokeBuffer,
          widthOutput: this.widthBuffer,
          radiusOutput: this.radiusBuffer
        }
      });
      this.geometryRuntime = new GeometryRuntime({
        device,
        column: GEOMETRY_COLUMN,
        geometry,
        selectionMask,
        colors,
        strokeColors,
        sizes: widths,
        pointSizes: radii,
        ownsGeometry: true,
        stylesResolved: true
      });
      if (this.geometryRuntime.reason) throw new Error(this.geometryRuntime.reason);
      this.binding = this.geometryRuntime.binding;
      this.geometryRuntime.prepare();
      this.gpuByteLength =
        geometry.gpuByteLength +
        this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
        this.geometryRuntime.accountedResidentByteLength;
      this.update(options.filters, options.style, options.channels);
    } catch (error) {
      for (const buffer of this.buffers) buffer.destroy();
      geometry.destroy();
      throw error;
    }
  }

  /** Writes the filter and style parameters and refreshes mask, colors and gathered primitive spaces. */
  update(
    filters: readonly DecodedTileFilter[],
    style: VectorTileGpuStyle,
    channels?: VectorTileExtraChannels
  ): void {
    if (this.destroyed) return;
    this.parametersBuffer.write(
      packVectorTileParameters({
        rowCount: this.rowCount,
        fieldIndexByName: this.fieldIndexByName,
        filters,
        style,
        channels
      })
    );
    const encoder = this.device.createCommandEncoder({id: `kepler-vector-tile-${this.tileKey}`});
    const computePass = encoder.beginComputePass({});
    // One workgroup per 64 features; tiles stay far below the 65535 workgroup dispatch limit.
    this.computation.dispatch(computePass, Math.ceil(this.rowCount / WORKGROUP_SIZE));
    computePass.end();
    this.geometryRuntime.encode(encoder);
    this.device.submit(encoder.finish());
    this.updateCount++;
  }

  getUpdateCount(): number {
    return this.updateCount;
  }

  /** Test and diagnostics readback of the GPU filter mask (one word per feature). */
  async readFilterMask(): Promise<Uint32Array> {
    const bytes = await this.maskBuffer.readAsync(0, this.rowCount * 4);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, this.rowCount).slice();
  }

  /** Test and diagnostics readback of the GPU packed colors (one word per feature). */
  async readColors(): Promise<Uint32Array> {
    const bytes = await this.colorBuffer.readAsync(0, this.rowCount * 4);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, this.rowCount).slice();
  }

  /** Test and diagnostics readback of the GPU line widths. */
  async readWidths(): Promise<Float32Array> {
    const bytes = await this.widthBuffer.readAsync(0, this.rowCount * 4);
    return new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + this.rowCount * 4)
    );
  }

  /** Test and diagnostics readback of the GPU point radii. */
  async readRadii(): Promise<Float32Array> {
    const bytes = await this.radiusBuffer.readAsync(0, this.rowCount * 4);
    return new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + this.rowCount * 4)
    );
  }

  /** Test and diagnostics readback of the GPU packed stroke colors. */
  async readStrokeColors(): Promise<Uint32Array> {
    const bytes = await this.strokeBuffer.readAsync(0, this.rowCount * 4);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, this.rowCount).slice();
  }

  /** Caller must have retired draws that reference this tile. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.geometryRuntime.destroy();
    this.computation.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }

  private allocate(id: string, data: Float32Array | Uint32Array): Buffer {
    const buffer = this.device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return buffer;
  }

  private makeVector<Format extends 'uint32' | 'unorm8x4' | 'float32'>(
    format: Format,
    buffer: Buffer
  ): GPUVector<Format> {
    return new GPUVector({
      type: 'data',
      name: `vector-tile-${format}`,
      format,
      ownsData: false,
      data: [new GPUData({format, length: this.rowCount, buffer, ownsBuffer: false})]
    });
  }
}
