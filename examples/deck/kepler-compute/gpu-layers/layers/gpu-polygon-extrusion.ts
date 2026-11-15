// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import type {Color} from '@deck.gl/core';
import type {GPUData, GPUVector, VertexList} from '@luma.gl/graph/gpu-data';
import {getGPUDataBuffer} from './gpu-vector-layer-utils';

/** Closed ring paths retain original source batches and may contain holes and multipolygons. */
export type GPUPolygonExtrusionRings = {
  positions: GPUVector<VertexList<'float32x3'>>;
  /** Optional fp64 low part of ring XY (float32x2), vertex-aligned with `positions`. */
  positionsLow?: GPUVector<VertexList<'float32x2'>>;
  sourceRowIds: GPUVector<'uint32'>;
  /** Per ring: +1 multiplies [dy,-dx] for shell CCW / hole CW; -1 for the inverse. */
  sideOrientations?: GPUVector<'float32'>;
  /** Without explicit orientations, shells use this winding and holes the opposite. */
  windingOrder?: 'CW' | 'CCW';
};

export const GPU_POLYGON_EXTRUSION_SHADER = /* wgsl */ `
struct PolygonExtrusionConfig {
  rowBase: u32, rowCount: u32, elevationOffset: u32, elevationStride: u32,
  elevation: f32, elevationScale: f32, enabled: u32, useElevations: u32,
  color: vec4<f32>,
  colorMode: u32, colorOffset: u32, colorStride: u32, ringSourceOffset: u32,
  ringSourceStride: u32, ringPositionOffset: u32, ringPositionStride: u32, orientationOffset: u32,
  orientationStride: u32, useOrientations: u32, normalSign: f32, edgeOffset: u32,
  baseElevation: f32, baseOffset: u32, baseStride: u32, useBaseElevations: u32,
};
@group(0) @binding(auto) var<uniform> polygonExtrusion: PolygonExtrusionConfig;
@group(0) @binding(auto) var<storage, read> polygonElevations: array<f32>;
@group(0) @binding(auto) var<storage, read> polygonColors: array<u32>;
@group(0) @binding(auto) var<storage, read> polygonElevationOffsets: array<f32>;
@group(0) @binding(auto) var<storage, read> polygonRingPositions: array<f32>;
@group(0) @binding(auto) var<storage, read> polygonRingSources: array<u32>;
@group(0) @binding(auto) var<storage, read> polygonTopology: array<u32>;
@group(0) @binding(auto) var<storage, read> polygonOrientations: array<f32>;

fn isPolygonSourceRow(row: u32) -> bool {
  return row >= polygonExtrusion.rowBase && row - polygonExtrusion.rowBase < polygonExtrusion.rowCount;
}
fn getPolygonBaseElevation(row: u32) -> f32 {
  if (!isPolygonSourceRow(row)) { return 0.0; }
  if (polygonExtrusion.useBaseElevations != 0u) {
    return polygonElevationOffsets[polygonExtrusion.baseOffset + (row - polygonExtrusion.rowBase) * polygonExtrusion.baseStride];
  }
  return polygonExtrusion.baseElevation;
}
fn getPolygonEdge(index: u32) -> vec4<u32> {
  let offset = polygonExtrusion.edgeOffset + index * 4u;
  return vec4<u32>(polygonTopology[offset], polygonTopology[offset + 1u], polygonTopology[offset + 2u], polygonTopology[offset + 3u]);
}
fn getPolygonElevation(row: u32) -> f32 {
  if (polygonExtrusion.enabled == 0u || !isPolygonSourceRow(row)) { return 0.0; }
  var elevation = polygonExtrusion.elevation;
  if (polygonExtrusion.useElevations != 0u) {
    elevation = polygonElevations[polygonExtrusion.elevationOffset + (row - polygonExtrusion.rowBase) * polygonExtrusion.elevationStride];
  }
  return elevation * polygonExtrusion.elevationScale;
}
fn getPolygonColor(row: u32, vertexColor: vec4<f32>, side: bool) -> vec4<f32> {
  if (polygonExtrusion.colorMode == 0u) { return polygonExtrusion.color; }
  if (polygonExtrusion.colorMode == 2u && !side) { return vertexColor; }
  if (!isPolygonSourceRow(row)) { return vec4<f32>(0.0); }
  let localRow = row - polygonExtrusion.rowBase;
  var colorIndex = localRow;
  if (polygonExtrusion.colorMode == 2u) {
    if (polygonTopology[localRow] == polygonTopology[localRow + 1u]) { return vec4<f32>(0.0); }
    colorIndex = polygonTopology[localRow];
  }
  return unpack4x8unorm(polygonColors[polygonExtrusion.colorOffset + colorIndex * polygonExtrusion.colorStride]);
}
fn getPolygonRingPosition(index: u32) -> vec3<f32> {
  let offset = polygonExtrusion.ringPositionOffset + index * polygonExtrusion.ringPositionStride;
  return vec3<f32>(polygonRingPositions[offset], polygonRingPositions[offset + 1u], polygonRingPositions[offset + 2u]);
}
fn projectPolygonNormal(normal: vec3<f32>) -> vec3<f32> {
  if (project.coordinateFrame == COORDINATE_SYSTEM_GEODETIC) {
    return normalize(normal * project.commonUnitsPerWorldUnit);
  }
  return project_normal(normal);
}
fn shadePolygon(color: vec4<f32>, normal: vec3<f32>, commonPosition: vec4<f32>) -> vec4<f32> {
  if (polygonExtrusion.enabled == 0u) { return color; }
  return vec4<f32>(lighting_getLightColor2(color.rgb, project.cameraPosition, commonPosition.xyz, normal), color.a);
}
`;

/** Edge topology only: coordinates and per-feature values are never read or duplicated. */
export function makeGPUPolygonEdgeRecords(offsets: Int32Array): Uint32Array {
  let count = 0;
  for (let ring = 0; ring + 1 < offsets.length; ring++) {
    assert(offsets[ring] >= 0 && offsets[ring + 1] >= offsets[ring]);
    count += Math.max(0, offsets[ring + 1] - offsets[ring] - 1);
  }
  const records = new Uint32Array(count * 4);
  let edge = 0;
  for (let ring = 0; ring + 1 < offsets.length; ring++) {
    for (let vertex = offsets[ring]; vertex + 1 < offsets[ring + 1]; vertex++) {
      records.set([vertex, vertex + 1, ring, 0], edge++ * 4);
    }
  }
  return records;
}

/** Per-source-batch extrusion bindings. Owns topology/control buffers, borrows all input vectors. */
export class GPUPolygonExtrusionBinding {
  readonly edgeCount: number;
  readonly ownedBuffers: readonly Buffer[];
  private readonly configuration: Buffer;
  private readonly fallback: Buffer;
  private readonly topology: Buffer;
  private readonly edgeOffset: number;
  private readonly wireframeConfiguration?: Buffer;
  private destroyed = false;

  constructor(
    readonly device: Device,
    readonly id: string,
    readonly positions: GPUData<VertexList<'float32x4'>>,
    readonly rings?: GPUData<VertexList<'float32x3'>>,
    readonly ringSources?: GPUData<'uint32'>,
    readonly orientations?: GPUData<'float32'>,
    wireframe = false
  ) {
    // Offsets are retained producer metadata, not a GPU readback or implicit coordinate packing.
    assert(positions.valueOffsets && positions.valueOffsets.length === positions.length + 1);
    assert(
      positions.valueOffsets[0] === 0 &&
        positions.valueOffsets[positions.length] === positions.valueLength
    );
    if (rings) {
      assert(rings.format === 'vertex-list<float32x3>' && ringSources?.format === 'uint32');
      assert(!orientations || orientations.format === 'float32');
      assert(rings.valueOffsets && rings.valueOffsets.length === rings.length + 1);
      assert(rings.valueOffsets[0] === 0 && rings.valueOffsets[rings.length] === rings.valueLength);
      assert(ringSources?.length === rings.length);
      assert(!orientations || orientations.length === rings.length);
    }
    const records = rings ? makeGPUPolygonEdgeRecords(rings.valueOffsets!) : new Uint32Array(0);
    this.edgeCount = records.length / 4;
    this.edgeOffset = positions.length + 1;
    const topology = new Uint32Array(Math.max(4, this.edgeOffset + records.length));
    topology.set(positions.valueOffsets);
    topology.set(records, this.edgeOffset);
    const buffers: Buffer[] = [];
    const createBuffer = (name: string, data: Uint32Array, usage = Buffer.STORAGE) => {
      const buffer = device.createBuffer({id: `${id}-${name}`, data, usage});
      buffers.push(buffer);
      return buffer;
    };
    try {
      this.configuration = createBuffer(
        'configuration',
        new Uint32Array(28),
        Buffer.UNIFORM | Buffer.COPY_DST
      );
      this.fallback = createBuffer('fallback', new Uint32Array(4));
      this.topology = createBuffer('topology', topology);
      if (wireframe)
        this.wireframeConfiguration = createBuffer(
          'wireframe-configuration',
          new Uint32Array(28),
          Buffer.UNIFORM | Buffer.COPY_DST
        );
      this.ownedBuffers = buffers;
      for (const input of [rings, ringSources, orientations]) if (input) this.borrow(input);
    } catch (error) {
      buffers.forEach(buffer => buffer.destroy());
      throw error;
    }
  }

  get ownedByteLength(): number {
    if (this.destroyed) return 0;
    return this.ownedBuffers.reduce((total, buffer) => total + buffer.byteLength, 0);
  }

  update(
    props: {
      rowIndexBase: number;
      extruded: boolean;
      elevation: number | GPUData<'float32'>;
      elevationScale: number;
      elevationOffset?: number | GPUData<'float32'>;
      color: Color | GPUData<'unorm8x4' | VertexList<'unorm8x4'>>;
      windingOrder?: 'CW' | 'CCW';
    },
    wireframe = false
  ) {
    const elevationOffset =
      typeof props.elevationOffset === 'number' ? undefined : props.elevationOffset;
    if (elevationOffset)
      assert(
        elevationOffset.format === 'float32' && elevationOffset.length === this.positions.length
      );
    const elevation = typeof props.elevation === 'number' ? undefined : props.elevation;
    const color =
      Array.isArray(props.color) || ArrayBuffer.isView(props.color)
        ? undefined
        : (props.color as GPUData<'unorm8x4' | VertexList<'unorm8x4'>>);
    if (elevation)
      assert(elevation.format === 'float32' && elevation.length === this.positions.length);
    if (color)
      assert(
        color.length === this.positions.length &&
          (color.format === 'unorm8x4' || color.format === 'vertex-list<unorm8x4>')
      );
    const values = new Uint32Array(28);
    const floats = new Float32Array(values.buffer);
    values.set([
      props.rowIndexBase,
      this.positions.length,
      (elevation?.byteOffset ?? 0) / 4,
      (elevation?.byteStride ?? 4) / 4
    ]);
    floats[4] = typeof props.elevation === 'number' ? props.elevation : 0;
    floats[5] = props.elevationScale;
    values[6] = Number(props.extruded);
    values[7] = Number(Boolean(elevation));
    const rgba = color ? [0, 0, 0, 255] : (props.color as Color);
    floats.set([rgba[0] / 255, rgba[1] / 255, rgba[2] / 255, (rgba[3] ?? 255) / 255], 8);
    values.set(
      [
        color ? (color.format === 'unorm8x4' ? 1 : 2) : 0,
        (color?.byteOffset ?? 0) / 4,
        (color?.byteStride ?? 4) / 4,
        (this.ringSources?.byteOffset ?? 0) / 4,
        (this.ringSources?.byteStride ?? 4) / 4,
        (this.rings?.byteOffset ?? 0) / 4,
        (this.rings?.byteStride ?? 12) / 4,
        (this.orientations?.byteOffset ?? 0) / 4,
        (this.orientations?.byteStride ?? 4) / 4,
        Number(Boolean(this.orientations))
      ],
      12
    );
    floats[22] = props.windingOrder === 'CCW' ? 1 : -1;
    values[23] = this.edgeOffset;
    floats[24] = typeof props.elevationOffset === 'number' ? props.elevationOffset : 0;
    values[25] = (elevationOffset?.byteOffset ?? 0) / 4;
    values[26] = (elevationOffset?.byteStride ?? 4) / 4;
    values[27] = Number(Boolean(elevationOffset));
    const configuration = wireframe ? this.wireframeConfiguration : this.configuration;
    assert(configuration);
    configuration.write(values);
    return {
      polygonExtrusion: configuration,
      polygonElevations: elevation ? this.borrow(elevation) : this.fallback,
      polygonColors: color ? this.borrow(color) : this.fallback,
      polygonElevationOffsets: elevationOffset ? this.borrow(elevationOffset) : this.fallback,
      polygonRingPositions: this.rings ? this.borrow(this.rings) : this.fallback,
      polygonRingSources: this.ringSources ? this.borrow(this.ringSources) : this.fallback,
      polygonTopology: this.topology,
      polygonOrientations: this.orientations ? this.borrow(this.orientations) : this.fallback
    };
  }

  /** Owner must fence all consumers before releasing topology and control storage. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.ownedBuffers.forEach(buffer => buffer.destroy());
  }

  private borrow(input: GPUData): Buffer {
    const buffer = getGPUDataBuffer(input);
    // Source storage is word-addressable; callers explicitly split overlarge physical buffers.
    assert(buffer.device === this.device && Boolean(buffer.usage & Buffer.STORAGE));
    assert(buffer.byteLength <= this.device.limits.maxStorageBufferBindingSize);
    assert(input.byteOffset % 4 === 0 && input.byteStride % 4 === 0);
    return buffer;
  }
}
