// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import type {GPUData, VertexList} from '@luma.gl/graph/gpu-data';
import {getGPUDataBuffer} from './gpu-vector-layer-utils';

/** Marker replaced by {@link applyGPUPolygonLow}; the default variant adds only zero-returning stubs. */
export const GPU_POLYGON_LOW_MARKER = '/*POLYGON_LOW*/';

const GPU_POLYGON_LOW_DISABLED_WGSL = /* wgsl */ `
fn getPolygonTopLow(vertexIndex: u32) -> vec3<f32> { return vec3<f32>(0.0); }
fn getPolygonRingLow(ringPointIndex: u32) -> vec3<f32> { return vec3<f32>(0.0); }
`;

const GPU_POLYGON_LOW_ENABLED_WGSL = /* wgsl */ `
struct PolygonLowConfig { topOffset: u32, ringOffset: u32, useTop: u32, useRing: u32 };
@group(0) @binding(auto) var<uniform> polygonLowConfig: PolygonLowConfig;
@group(0) @binding(auto) var<storage, read> polygonLowPositions: array<f32>;
@group(0) @binding(auto) var<storage, read> polygonRingLowPositions: array<f32>;
// fp64 low parts of XY (Deck position64Low); the top table is indexed by the fetched vertex index.
fn getPolygonTopLow(vertexIndex: u32) -> vec3<f32> {
  if (polygonLowConfig.useTop == 0u) { return vec3<f32>(0.0); }
  let offset = polygonLowConfig.topOffset + vertexIndex * 2u;
  return vec3<f32>(polygonLowPositions[offset], polygonLowPositions[offset + 1u], 0.0);
}
fn getPolygonRingLow(ringPointIndex: u32) -> vec3<f32> {
  if (polygonLowConfig.useRing == 0u) { return vec3<f32>(0.0); }
  let offset = polygonLowConfig.ringOffset + ringPointIndex * 2u;
  return vec3<f32>(polygonRingLowPositions[offset], polygonRingLowPositions[offset + 1u], 0.0);
}
`;

/** Injects the fp64 low-part declarations; without them the shader keeps zero-valued low parts. */
export function applyGPUPolygonLow(
  source: string,
  enabled: boolean,
  includeRing = true,
  includeTop = true
): string {
  // The fill stage never reads ring lows and the wall stages never read top lows; dropping those
  // bindings keeps each stage within the 8 vertex storage buffer limit.
  let enabledSource = GPU_POLYGON_LOW_ENABLED_WGSL;
  if (!includeRing) {
    enabledSource = enabledSource
      .replace(/^.*polygonRingLowPositions: array.*\n/m, '')
      .replace(
        /fn getPolygonRingLow[\s\S]*?\n}\n/,
        'fn getPolygonRingLow(ringPointIndex: u32) -> vec3<f32> { return vec3<f32>(0.0); }\n'
      );
  }
  if (!includeTop) {
    enabledSource = enabledSource
      .replace(/^.*polygonLowPositions: array.*\n/m, '')
      .replace(
        /fn getPolygonTopLow[\s\S]*?\n}\n/,
        'fn getPolygonTopLow(vertexIndex: u32) -> vec3<f32> { return vec3<f32>(0.0); }\n'
      );
  }
  return source.replace(
    GPU_POLYGON_LOW_MARKER,
    enabled ? enabledSource : GPU_POLYGON_LOW_DISABLED_WGSL
  );
}

/** Per-source-batch fp64 low-part bindings. Owns two tiny buffers; borrows the low vectors. */
export class GPUPolygonLowBinding {
  private readonly configuration: Buffer;
  private readonly fallback: Buffer;
  private destroyed = false;

  constructor(
    device: Device,
    id: string,
    private readonly top?: GPUData<VertexList<'float32x2'>>,
    private readonly ring?: GPUData<VertexList<'float32x2'>>
  ) {
    // Low parts must be packed float32x2 words aligned with their high-part vertices.
    assert(!top || (top.format === 'vertex-list<float32x2>' && top.byteStride === 8));
    assert(!ring || (ring.format === 'vertex-list<float32x2>' && ring.byteStride === 8));
    assert(!top || top.byteOffset % 4 === 0);
    assert(!ring || ring.byteOffset % 4 === 0);
    this.configuration = device.createBuffer({
      id: `${id}-low-configuration`,
      data: new Uint32Array([
        (top?.byteOffset ?? 0) / 4,
        (ring?.byteOffset ?? 0) / 4,
        Number(Boolean(top)),
        Number(Boolean(ring))
      ]),
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    this.fallback = device.createBuffer({
      id: `${id}-low-fallback`,
      data: new Uint32Array(4),
      usage: Buffer.STORAGE
    });
  }

  getBindings() {
    return {
      polygonLowConfig: this.configuration,
      polygonLowPositions: this.top ? getGPUDataBuffer(this.top) : this.fallback,
      polygonRingLowPositions: this.ring ? getGPUDataBuffer(this.ring) : this.fallback
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.configuration.destroy();
    this.fallback.destroy();
  }
}
