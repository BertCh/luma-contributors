// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A lens snapshot drawn by luma.gl: the recorded raster and point layers, with an isoline band that
// sweeps through the value range. The static fallback is the same raster as a build-time PNG.

import type {Buffer, Device} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GpuScene, GpuSceneLoader} from '../../../gpu/types';
import type {LensMap} from '../../../lens/types';
import {
  assertWebGpu,
  createQuadModel,
  createStorageBuffer,
  createUniformBuffer,
  getSceneColors,
  RAMP_WGSL,
  STILL_FRAME_MILLISECONDS
} from './scene-kit';

const LOOK_FLOATS = 28;

const COMMON_WGSL = /* wgsl */ `
struct Look {
  grid: vec4f,
  clock: vec4f,
  size: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  ink: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
${RAMP_WGSL}
`;

const RASTER_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(1) var<storage, read> cells: array<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}

fn cellValue(column: i32, row: i32) -> f32 {
  let columns = i32(look.grid.x);
  let rows = i32(look.grid.y);
  return cells[u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1))];
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let shifted = input.uv * look.grid.xy - vec2f(0.5);
  let base = vec2i(floor(shifted));
  let fraction = shifted - floor(shifted);
  let value = mix(
    mix(cellValue(base.x, base.y), cellValue(base.x + 1, base.y), fraction.x),
    mix(cellValue(base.x, base.y + 1), cellValue(base.x + 1, base.y + 1), fraction.x),
    fraction.y
  );
  let level = fract(look.clock.x * 0.09);
  let band = exp(-pow((value - level) / 0.025, 2.0)) * look.clock.y;
  let body = sampleRamp(value);
  let visible = smoothstep(0.02, 0.28, value);
  var color = mix(look.background.rgb, body, visible);
  color = mix(color, look.ink.rgb, band * 0.6);
  return vec4f(color, 1.0);
}
`;

const POINT_WGSL = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(1) var<storage, read> points: array<vec4f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) corner: vec2f,
  @location(1) value: f32,
  @location(2) kind: f32
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let point = points[instanceIndex];
  let corner = corners[vertexIndex];
  let sweep = exp(-pow((point.x - fract(look.clock.x * 0.12)) / 0.07, 2.0));
  let radius = look.clock.z * (1.0 + 0.7 * sweep * select(0.0, 1.0, point.w > 0.5));
  let center = vec2f(point.x * 2.0 - 1.0, 1.0 - point.y * 2.0);
  var output: VertexOutput;
  output.position = vec4f(center + corner * radius * 2.0 / look.size.xy, 0.0, 1.0);
  output.corner = corner;
  output.value = point.z;
  output.kind = point.w;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.corner);
  if (distance > 1.0) {
    discard;
  }
  let edge = 1.0 - smoothstep(0.7, 1.0, distance);
  if (input.kind > 1.5) {
    return vec4f(look.signal.rgb, edge);
  }
  if (input.kind > 0.5) {
    return vec4f(sampleRamp(input.value), edge);
  }
  return vec4f(look.ink.rgb, edge * 0.5);
}
`;

export function createMapSceneLoader(map: LensMap): GpuSceneLoader {
  return async () => ({default: createMapScene(map)});
}

export function createMapScene(map: LensMap): GpuScene {
  return ({device, theme, width, height, reducedMotion}) =>
    new MapSceneInstance(device, map, theme, width, height, reducedMotion);
}

class MapSceneInstance {
  private readonly device: Device;
  private readonly colors: ReturnType<typeof getSceneColors>;
  private readonly look = new Float32Array(LOOK_FLOATS);
  private readonly lookBuffer: Buffer;
  private readonly buffers: Buffer[] = [];
  private readonly rasterModel: Model | undefined;
  private readonly pointModel: Model | undefined;
  private readonly rasterSize: [number, number];
  private readonly pointRadius: number;
  private readonly reducedMotion: boolean;
  private width: number;
  private height: number;

  constructor(
    device: Device,
    map: LensMap,
    theme: 'light' | 'dark',
    width: number,
    height: number,
    reducedMotion: boolean
  ) {
    assertWebGpu(device);
    this.device = device;
    this.colors = getSceneColors(theme);
    this.width = width;
    this.height = height;
    this.reducedMotion = reducedMotion;
    this.lookBuffer = createUniformBuffer(device, LOOK_FLOATS);

    this.rasterSize = map.raster ? [map.raster.width, map.raster.height] : [1, 1];
    if (map.raster) {
      const bytes = Uint8Array.from(atob(map.raster.grid), character => character.charCodeAt(0));
      const values = Float32Array.from(bytes, byte => byte / 255);
      const buffer = createStorageBuffer(device, values);
      this.buffers.push(buffer);
      this.rasterModel = createQuadModel(device, {
        id: 'map-raster',
        source: RASTER_WGSL,
        bindings: {look: this.lookBuffer, cells: buffer},
        instanceCount: 1,
        blendMode: 'opaque',
        theme
      });
    }

    const pointList: number[] = [];
    map.layers.forEach((layer, layerIndex) => {
      for (let index = 0; index < layer.positions.length / 2; index++) {
        const value = layer.values?.[index];
        // kind: 0 plain, 1 colored by value, 2 highlighted marker (small unvalued layers).
        const kind =
          value !== undefined ? 1 : layerIndex > 0 && layer.positions.length <= 64 ? 2 : 0;
        pointList.push(
          layer.positions[index * 2],
          layer.positions[index * 2 + 1],
          value ?? 0,
          kind
        );
      }
    });
    const pointCount = pointList.length / 4;
    this.pointRadius = Math.min(7, Math.max(2.2, 120 / Math.sqrt(Math.max(1, pointCount))));
    if (pointCount > 0) {
      const buffer = createStorageBuffer(device, Float32Array.from(pointList));
      this.buffers.push(buffer);
      this.pointModel = createQuadModel(device, {
        id: 'map-points',
        source: POINT_WGSL,
        bindings: {look: this.lookBuffer, points: buffer},
        instanceCount: pointCount,
        theme
      });
    }
  }

  render(timeMilliseconds: number): void {
    const seconds = (this.reducedMotion ? STILL_FRAME_MILLISECONDS : timeMilliseconds) / 1000;
    const look = this.look;
    look.set([this.rasterSize[0], this.rasterSize[1], 0, 0], 0);
    look.set(
      [
        seconds,
        this.reducedMotion ? 0 : 1,
        this.pointRadius * 0.55 * (this.width / 700 > 1 ? 1.3 : 1),
        0
      ],
      4
    );
    look.set([this.width, this.height, 0, 0], 8);
    look.set(this.colors.background, 12);
    look.set(this.colors.accent, 16);
    look.set(this.colors.signal, 20);
    look.set(this.colors.ink, 24);
    this.lookBuffer.write(look);
    const renderPass = this.device.beginRenderPass({clearColor: this.colors.background});
    this.rasterModel?.draw(renderPass);
    this.pointModel?.draw(renderPass);
    renderPass.end();
    this.device.submit();
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  destroy(): void {
    this.rasterModel?.destroy();
    this.pointModel?.destroy();
    this.lookBuffer.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

export default createMapScene({title: '', aspectRatio: 1, layers: []});
