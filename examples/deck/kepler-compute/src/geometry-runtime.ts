// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, type CommandEncoder} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUGather, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {
  PreparedKeplerGeometryGPUVectors,
  KeplerGeometryPrimitiveMapping,
  KeplerGeometryGPUPaths
} from './geometry-preparation';
import type {ResidentDatasetBinding, ResidentGeometryPathBinding} from './resident-layers';

export type GeometryRuntimeProps = {
  device: Device;
  column: string;
  geometry: PreparedKeplerGeometryGPUVectors;
  selectionMask: GPUVector<'uint32'>;
  /** Effective source-row fill colors, including any analysis override. */
  colors?: GPUVector<'unorm8x4'>;
  strokeColors?: GPUVector<'unorm8x4'>;
  /** Source-row line widths; independent from point radii. */
  sizes?: GPUVector<'float32'>;
  pointSizes?: GPUVector<'float32'>;
  /** Polygon height and unscaled base offset are independent of line width and point radius. */
  elevations?: number | GPUVector<'float32'>;
  elevationOffsets?: number | GPUVector<'float32'>;
  /** Default false: the registry/caller keeps geometry ownership. */
  ownsGeometry?: boolean;
  /** Host planner has resolved enabled property channels; inactive/field-bound properties are ignored. */
  stylesResolved?: boolean;
};

/**
 * Source masks and visual channels fan out to independent primitive spaces through GPUGather.
 * Geometry, indices and source-row maps are immutable. Per-frame encoding only updates borrowed
 * masks/styles and retains source identity; it never reads back or retessellates geometry.
 */
export class GeometryRuntime {
  readonly binding: NonNullable<ResidentDatasetBinding['geojson']>;
  readonly geometry: PreparedKeplerGeometryGPUVectors;
  readonly reason?: string;
  private readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private compiled?: CompiledGPUCommandGraph;
  private destroyed = false;

  constructor(private readonly props: GeometryRuntimeProps) {
    this.geometry = props.geometry;
    const metadata = props.geometry.metadata;
    const styleChannels = {
      fillColor: props.colors,
      lineColor: props.strokeColors,
      lineWidth: props.sizes,
      radius: props.pointSizes,
      elevation: props.elevations,
      elevationOffset: props.elevationOffsets
    };
    const unsupportedStyles = metadata.sourceStyleProperties.filter(
      property => styleChannels[property as keyof typeof styleChannels] === undefined
    );
    this.reason =
      metadata.crs !== undefined &&
      metadata.crs !== null &&
      metadata.crs !== 'EPSG:4326' &&
      metadata.crs !== 'OGC:CRS84'
        ? 'Geometry CRS requires explicit transformation to longitude/latitude'
        : unsupportedStyles.length && !props.stylesResolved
          ? `GeoJSON source properties require resident style channels: ${unsupportedStyles.join(', ')}`
          : undefined;
    const rowCount = props.geometry.metadata.rowCount;
    for (const vector of [
      props.selectionMask,
      props.colors,
      props.strokeColors,
      props.sizes,
      props.pointSizes,
      props.elevations,
      props.elevationOffsets
    ]) {
      if (typeof vector === 'number') {
        if (!Number.isFinite(Math.fround(vector)))
          throw new Error('Geometry vertical defaults require finite Float32 values');
        continue;
      }
      if (vector && vector.length !== rowCount)
        throw new Error('Geometry style/mask source row count mismatch');
      for (const chunk of vector?.data ?? [])
        if (chunk.buffer.device !== props.device)
          throw new Error('Geometry source device mismatch');
    }
    for (const buffer of props.geometry.buffers)
      if (buffer.device !== props.device) throw new Error('Geometry device mismatch');
    this.graph = new GPUCommandGraph(props.device, {id: `geometry-${props.column}`});
    try {
      const points = props.geometry.points;
      const polygonGeometry = props.geometry.polygons;
      const polygonRows = new GPUVector({
        type: 'data',
        name: 'polygon-vertex-source-rows',
        format: 'uint32',
        ownsData: false,
        data: polygonGeometry.rowIndices.data.map(
          chunk =>
            new GPUData({
              format: 'uint32',
              buffer: chunk.buffer,
              length: chunk.valueLength,
              byteOffset: chunk.byteOffset,
              ownsBuffer: false
            })
        )
      });
      const sourceRows = new GPUVector({
        type: 'data',
        name: 'geometry-source-row-identity',
        format: 'uint32',
        ownsData: false,
        data: props.geometry.batches.map(
          (batch, index) =>
            new GPUData({
              format: 'uint32',
              length: batch.rowCount,
              buffer: this.allocate(
                `geometry-source-rows-${index}`,
                Math.max(4, batch.rowCount * 4),
                Uint32Array.from({length: batch.rowCount}, (_, row) => batch.sourceRowOffset + row)
              ),
              ownsBuffer: false
            })
        )
      });
      const polygonMask = this.gather('polygon-mask', props.selectionMask, sourceRows);
      const polygonColors = props.colors
        ? this.gather('polygon-colors', props.colors, polygonRows)
        : undefined;
      const getFillColor = polygonColors
        ? new GPUVector<VertexList<'unorm8x4'>>({
            type: 'data',
            name: 'polygon-vertex-colors',
            format: 'vertex-list<unorm8x4>',
            ownsData: false,
            data: polygonColors.data.map(
              (chunk, index) =>
                new GPUData({
                  format: 'vertex-list<unorm8x4>',
                  buffer: chunk.buffer,
                  length: polygonGeometry.positions.data[index].length,
                  valueLength: chunk.length,
                  valueOffsets: polygonGeometry.positions.data[index].valueOffsets,
                  ownsBuffer: false
                })
            )
          })
        : undefined;
      this.binding = {
        column: props.column,
        // Native geojson labels sit on one representative anchor per source row.
        ...(props.geometry.representative
          ? {
              labelAnchors: {
                positions: props.geometry.representative.positions,
                validity: props.geometry.representative.validity
              }
            }
          : {}),
        ...(props.geometry.metadata.featureTypes.point
          ? {
              points: {
                positions: points.positions,
                positions3D: points.positions3D,
                selectionMask: this.gather('point-mask', props.selectionMask, points.sourceRowIds),
                resolveSourceRow: points.resolveSourceRow,
                findPrimitiveRange: makePrimitiveRangeFinder(points.sourceRows),
                colors: props.colors
                  ? this.gather('point-colors', props.colors, points.sourceRowIds)
                  : undefined,
                strokeColors: props.strokeColors
                  ? this.gather('point-stroke-colors', props.strokeColors, points.sourceRowIds)
                  : undefined,
                lineWidths: props.sizes
                  ? this.gather('point-line-widths', props.sizes, points.sourceRowIds)
                  : undefined,
                sizes: props.pointSizes
                  ? this.gather('point-radii', props.pointSizes, points.sourceRowIds)
                  : undefined
              }
            }
          : {}),
        ...(props.geometry.metadata.featureTypes.line
          ? {paths: this.makePaths('paths', props.geometry.paths)}
          : {}),
        ...(props.geometry.metadata.featureTypes.polygon
          ? {
              polygons: {
                ...polygonGeometry,
                selectionMask: polygonMask,
                getFillColor,
                getLineColor: props.strokeColors
                  ? this.gather('polygon-stroke-colors', props.strokeColors, sourceRows)
                  : undefined,
                getElevation:
                  typeof props.elevations === 'number'
                    ? props.elevations
                    : props.elevations
                      ? this.gather('polygon-elevations', props.elevations, sourceRows)
                      : undefined,
                getElevationOffset:
                  typeof props.elevationOffsets === 'number'
                    ? props.elevationOffsets
                    : props.elevationOffsets
                      ? this.gather('polygon-elevation-offsets', props.elevationOffsets, sourceRows)
                      : undefined,
                extrusionRings: {
                  positions: props.geometry.polygonOutlines.getPath,
                  positionsLow: props.geometry.polygonOutlines.getPathLow,
                  sourceRowIds: props.geometry.polygonOutlines.sourceRowIds,
                  sideOrientations: props.geometry.polygonOutlines.sideOrientations
                },
                resolveSourceRow: index =>
                  Number.isSafeInteger(index) && index >= 0 && index < rowCount ? index : -1,
                // Polygon primitives are source rows, so a feature is exactly its own row.
                findPrimitiveRange: row =>
                  Number.isSafeInteger(row) && row >= 0 && row < rowCount ? [row, row] : null
              },
              polygonOutlines: this.makePaths('outlines', props.geometry.polygonOutlines)
            }
          : {})
      };
    } catch (error) {
      for (const buffer of this.buffers) buffer.destroy();
      throw error;
    }
  }

  resolve(column: string): NonNullable<ResidentDatasetBinding['geojson']> | undefined {
    return !this.destroyed && !this.reason && column === this.props.column
      ? this.binding
      : undefined;
  }

  /** Uniform native defaults can change without rebuilding gather outputs or source geometry. */
  setVerticalDefaults(defaults: {elevation?: number; elevationOffset?: number}): void {
    this.assertAlive();
    for (const value of [defaults.elevation, defaults.elevationOffset])
      if (value !== undefined && !Number.isFinite(Math.fround(value)))
        throw new Error('Geometry vertical defaults require finite Float32 values');
    const polygons = this.binding.polygons;
    if (!polygons) return;
    if (!this.props.elevations || typeof this.props.elevations === 'number')
      polygons.getElevation = defaults.elevation;
    if (!this.props.elevationOffsets || typeof this.props.elevationOffsets === 'number')
      polygons.getElevationOffset = defaults.elevationOffset;
  }

  prepare(): void {
    this.assertAlive();
    this.compiled ??= this.graph.compile();
  }

  encode(encoder: CommandEncoder): void {
    this.assertAlive();
    if (encoder.device !== this.props.device) throw new Error('Geometry encoder device mismatch');
    this.prepare();
    this.compiled!.encode(encoder, {parameters: undefined});
  }

  /** Runtime-owned mappings/styles/scratch; separately account geometry once in the registry. */
  get accountedResidentByteLength(): number {
    return this.destroyed
      ? 0
      : this.buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) +
          (this.compiled?.stats.physicalTransientResourceBytes ?? 0);
  }
  get readbackByteLength(): number {
    return 0;
  }
  get resources(): readonly Buffer[] {
    return this.buffers;
  }

  /** Caller must await dependent submissions before release. Inputs are borrowed by default. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled?.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    if (this.props.ownsGeometry) this.geometry.destroy();
  }

  private makePaths(name: string, geometry: KeplerGeometryGPUPaths): ResidentGeometryPathBinding {
    return {
      getPath: geometry.getPath,
      getPathLow: geometry.getPathLow,
      selectionMask: this.gather(`${name}-mask`, this.props.selectionMask, geometry.sourceRowIds),
      resolveSourceRow: geometry.resolveSourceRow,
      findPrimitiveRange: makePrimitiveRangeFinder(geometry.sourceRows),
      getColor: this.props.strokeColors
        ? this.gather(`${name}-colors`, this.props.strokeColors, geometry.sourceRowIds)
        : undefined,
      getWidth: this.props.sizes
        ? this.gather(`${name}-widths`, this.props.sizes, geometry.sourceRowIds)
        : undefined
    };
  }

  private gather<Format extends 'uint32' | 'float32' | 'unorm8x4'>(
    name: string,
    source: GPUVector<Format>,
    indices: KeplerGeometryPrimitiveMapping['sourceRowIds']
  ): GPUVector<Format> {
    const output = new GPUVector({
      type: 'data',
      name,
      format: source.format,
      ownsData: false,
      data: indices.data.map(
        (chunk, index) =>
          new GPUData({
            format: source.format,
            length: chunk.length,
            buffer: this.allocate(`${name}-${index}`, Math.max(4, chunk.length * 4)),
            ownsBuffer: false
          })
      )
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

  private allocate(id: string, byteLength: number, data?: Uint32Array): Buffer {
    const buffer = this.props.device.createBuffer({
      id,
      ...(data?.length ? {data} : {byteLength}),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return buffer;
  }
  private assertAlive(): void {
    if (this.destroyed) throw new Error('Geometry runtime has been destroyed');
  }
}

/**
 * Bounded CPU scan of the retained primitive map (no GPU readback). Primitives are emitted in
 * source order, so the hovered feature's parts form one contiguous row range.
 */
function makePrimitiveRangeFinder(
  sourceRows: readonly Uint32Array[]
): (sourceRow: number) => readonly [number, number] | null {
  return sourceRow => {
    let first = -1;
    let last = -1;
    let offset = 0;
    for (const rows of sourceRows) {
      for (let index = 0; index < rows.length; index++)
        if (rows[index] === sourceRow) {
          if (first < 0) first = offset + index;
          last = offset + index;
        }
      offset += rows.length;
    }
    return first < 0 ? null : [first, last];
  };
}
