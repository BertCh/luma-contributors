// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared building blocks of the flagship demos: seeded data, graph running, and three luma.gl
// layers (raster, dots, segments) that read analysis output buffers directly as WGSL storage
// buffers, so the picture is drawn from the very buffers the analysis wrote.

import {Buffer, type Device, type RenderPass} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {Model} from '@luma.gl/engine';
import type {GpuTheme} from '../types';

export type Color = readonly [number, number, number];

// ---------------------------------------------------------------------------------------------
// Seeded data

/** Small deterministic generator (mulberry32). Returns numbers in [0, 1). */
export function makeRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------------------------
// Theme and palettes

export type DemoTheme = {
  background: readonly [number, number, number, number];
  /** Sequential ramp, six stops from low to high. */
  ramp: readonly Color[];
  /** Diverging ramp, six stops from negative to positive. */
  diverging: readonly Color[];
  /** Categorical colors. */
  categories: readonly Color[];
  ink: Color;
  muted: Color;
};

function hex(value: string): Color {
  const number = Number.parseInt(value.slice(1), 16);
  return [((number >> 16) & 255) / 255, ((number >> 8) & 255) / 255, (number & 255) / 255];
}

const CATEGORIES_DARK = [
  '#5aa9ff',
  '#ffb454',
  '#ff6b6b',
  '#4fd1c5',
  '#8bd450',
  '#f5d547',
  '#c792ea',
  '#ff9ec7'
];
const CATEGORIES_LIGHT = [
  '#2563c9',
  '#d9730d',
  '#d1343b',
  '#12857a',
  '#4a8f1f',
  '#b8940b',
  '#8a4fc0',
  '#d1487f'
];

export function getDemoTheme(theme: GpuTheme): DemoTheme {
  if (theme === 'dark') {
    return {
      background: [0.043, 0.059, 0.09, 1],
      ramp: ['#0b0f1a', '#2a1a63', '#7a1f8f', '#d63f6b', '#ff9a3c', '#fff1a8'].map(hex),
      diverging: ['#3b7dff', '#7fb2ff', '#2a3350', '#2a3350', '#ff9a7a', '#ff4d4d'].map(hex),
      categories: CATEGORIES_DARK.map(hex),
      ink: hex('#e8ecf5'),
      muted: hex('#5b6580')
    };
  }
  return {
    background: [0.965, 0.957, 0.937, 1],
    ramp: ['#f6f4ef', '#fbe3b0', '#f6b26b', '#e5603b', '#a8204e', '#3b0f5e'].map(hex),
    diverging: ['#1f4fc4', '#7fa4e8', '#e7e3da', '#e7e3da', '#ee9a82', '#c4242b'].map(hex),
    categories: CATEGORIES_LIGHT.map(hex),
    ink: hex('#1c2230'),
    muted: hex('#9aa0ad')
  };
}

// ---------------------------------------------------------------------------------------------
// Buffers and graphs

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC;

/** Creates a storage buffer initialized with `values`. */
export function createStorageBuffer(
  device: Device,
  values: Float32Array | Uint32Array | Int32Array
): Buffer {
  return device.createBuffer({data: values, usage: STORAGE_USAGE});
}

/** Creates a zero-filled storage buffer of `length` 32-bit elements. */
export function createEmptyStorageBuffer(device: Device, length: number): Buffer {
  return device.createBuffer({byteLength: Math.max(length, 1) * 4, usage: STORAGE_USAGE});
}

/** Imports a caller-owned buffer into a graph as a typed view of `length` rows. */
export function importBufferView<Format extends GPUVectorFormat>(
  graph: GPUCommandGraph<void>,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {format, length});
}

/** A compiled command graph that can be encoded and submitted again and again. */
export class GraphRunner {
  readonly device: Device;
  readonly compiled: CompiledGPUCommandGraph<void>;

  private constructor(device: Device, compiled: CompiledGPUCommandGraph<void>) {
    this.device = device;
    this.compiled = compiled;
  }

  /** Builds the graph with `build`, then compiles it. */
  static async create(
    device: Device,
    id: string,
    build: (graph: GPUCommandGraph<void>) => void
  ): Promise<GraphRunner> {
    const graph = new GPUCommandGraph<void>(device, {id});
    build(graph);
    return new GraphRunner(device, await graph.compileAsync());
  }

  /** Encodes and submits the whole graph. Results are visible to the next render pass. */
  run(): void {
    const commandEncoder = this.device.createCommandEncoder({id: `${this.compiled.id}-encoder`});
    this.compiled.encode(commandEncoder, {parameters: undefined});
    this.device.submit(commandEncoder.finish());
  }

  destroy(): void {
    this.compiled.destroy();
  }
}

// ---------------------------------------------------------------------------------------------
// View: world rectangle [0, worldWidth] x [0, worldHeight] drawn "contained" in the canvas

export type DemoView = {
  worldWidth: number;
  worldHeight: number;
  /** Geographic runs draw in native longitude/latitude around this southwest origin. */
  origin?: readonly [number, number];
  longitudeScale?: number;
  width: number;
  height: number;
};

/** NDC scale that fits the world rectangle inside the canvas. */
export function getViewScale(view: DemoView): [number, number] {
  const worldAspect = (view.worldWidth * (view.longitudeScale ?? 1)) / view.worldHeight;
  const canvasAspect = view.width / view.height;
  return worldAspect > canvasAspect
    ? [1, canvasAspect / worldAspect]
    : [worldAspect / canvasAspect, 1];
}

/** Canvas pixel position to world coordinates (y up). */
export function getWorldPosition(
  view: DemoView,
  pixel: readonly [number, number]
): [number, number] {
  const [scaleX, scaleY] = getViewScale(view);
  const ndcX = (pixel[0] / view.width) * 2 - 1;
  const ndcY = 1 - (pixel[1] / view.height) * 2;
  return [
    (view.origin?.[0] ?? 0) + ((ndcX / scaleX + 1) / 2) * view.worldWidth,
    (view.origin?.[1] ?? 0) + ((ndcY / scaleY + 1) / 2) * view.worldHeight
  ];
}

// ---------------------------------------------------------------------------------------------
// Layers

export const PALETTE_STRUCT = /* wgsl */ `
struct Look {
  scale: vec2f,
  world: vec2f,
  size: vec2f,
  range: vec2f,
  mode: f32,
  radius: f32,
  alpha: f32,
  hillshade: f32,
  grid: vec2f,
  elevation: f32,
  smoothing: f32,
  background: vec4f,
  light: vec4f,
  stops: array<vec4f, 6>,
  colors: array<vec4f, 16>,
  // x: curve (0 linear, 1 log2), y: fade alpha with value (0 or 1), z: quantize to this many bands,
  // w: dots only, scale the radius with the value (0 or 1)
  extra: vec4f,
  origin: vec2f
};
@group(0) @binding(0) var<uniform> look: Look;

fn worldToClip(point: vec2f) -> vec2f {
  return ((point - look.origin) / look.world * 2.0 - 1.0) * look.scale;
}

fn isValid(value: f32) -> bool {
  return abs(value) < 1.0e20;
}

fn sampleRamp(position: f32) -> vec3f {
  let scaled = clamp(position, 0.0, 1.0) * 5.0;
  let index = min(u32(floor(scaled)), 4u);
  return mix(look.stops[index].rgb, look.stops[index + 1u].rgb, scaled - f32(index));
}

// mode 0: sequential ramp, 1: diverging ramp (range is symmetric about zero), 2: categorical,
// 3: cyclic hue wheel (angles).
fn colorize(value: f32) -> vec4f {
  if (!isValid(value)) {
    return vec4f(look.background.rgb, 0.0);
  }
  if (look.mode > 2.5) {
    // Cyclic hue wheel for angles: the range spans one full turn; values below it are muted.
    if (value < look.range.x) {
      return vec4f(look.background.rgb, 0.0);
    }
    let turn = fract((value - look.range.x) / max(look.range.y - look.range.x, 1.0e-6));
    return vec4f(0.55 + 0.4 * cos(6.28318 * (turn + vec3f(0.0, 0.3333, 0.6667))), 1.0);
  }
  if (look.mode > 1.5) {
    let index = u32(clamp(round(value), 0.0, 15.0));
    return vec4f(look.colors[index].rgb, look.colors[index].a);
  }
  var shaped = value;
  if (look.extra.x > 0.5) {
    shaped = log2(max(value, 1.0));
  }
  
  var span = look.range.y - look.range.x;
  if (abs(span) < 1.0e-12) {
    span = 1.0e-12;
  }
  var position = clamp((shaped - look.range.x) / span, 0.0, 1.0);
  if (look.extra.z > 0.5) {
    position = floor(position * look.extra.z) / look.extra.z;
  }
  let fade = select(1.0, position, look.extra.y > 0.5);
  return vec4f(sampleRamp(position), fade);
}
`;

const FLOAT_VALUES = (binding: number) => /* wgsl */ `
@group(0) @binding(${binding}) var<storage, read> values: array<f32>;
fn readValue(index: u32) -> f32 {
  return values[index];
}
`;

// Unsigned labels: wrapped into 16 categories; 0xffffffff (noise or unreached) reads as invalid.
const UINT_VALUES = (binding: number) => /* wgsl */ `
@group(0) @binding(${binding}) var<storage, read> values: array<u32>;
fn readValue(index: u32) -> f32 {
  let raw = values[index];
  return select(f32(raw % 16u), 1.0e30, raw == 0xffffffffu);
}
`;

export type ValueFormat = 'float32' | 'uint32';

export function getValues(format: ValueFormat, binding: number): string {
  return format === 'uint32' ? UINT_VALUES(binding) : FLOAT_VALUES(binding);
}

const getRasterShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
${getValues(format, 1)}

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
  output.position = vec4f((corner * 2.0 - 1.0) * look.scale, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}

fn cellValue(column: i32, row: i32) -> f32 {
  let columns = i32(look.grid.x);
  let rows = i32(look.grid.y);
  let clampedColumn = clamp(column, 0, columns - 1);
  let clampedRow = clamp(row, 0, rows - 1);
  return readValue(u32(clampedRow * columns + clampedColumn));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let cell = input.uv * look.grid;
  let column = i32(floor(cell.x));
  let row = i32(floor(cell.y));
  var value = cellValue(column, row);
  if (look.smoothing > 0.5 && isValid(value)) {
    let shifted = cell - vec2f(0.5);
    let base = vec2i(floor(shifted));
    let fraction = shifted - floor(shifted);
    let v00 = cellValue(base.x, base.y);
    let v10 = cellValue(base.x + 1, base.y);
    let v01 = cellValue(base.x, base.y + 1);
    let v11 = cellValue(base.x + 1, base.y + 1);
    if (isValid(v00) && isValid(v10) && isValid(v01) && isValid(v11)) {
      value = mix(mix(v00, v10, fraction.x), mix(v01, v11, fraction.x), fraction.y);
    }
  }
  var color = colorize(value);
  if (look.hillshade > 0.0 && color.a > 0.0) {
    let slopeX = (cellValue(column + 1, row) - cellValue(column - 1, row)) * look.elevation;
    let slopeY = (cellValue(column, row + 1) - cellValue(column, row - 1)) * look.elevation;
    let normal = normalize(vec3f(-slopeX, slopeY, 1.0));
    let shade = clamp(dot(normal, normalize(look.light.xyz)), 0.0, 1.0);
    color = vec4f(color.rgb * mix(1.0, 0.45 + 0.8 * shade, look.hillshade), color.a);
  }
  return vec4f(color.rgb, color.a * look.alpha);
}
`;

const getDotShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
${getValues(format, 2)}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) value: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let center = worldToClip(positions[instanceIndex]);
  let value = readValue(instanceIndex);
  var radius = look.radius;
  if (look.extra.w > 0.5) {
    // Size by value: from 30% of the radius at the low end of the range to all of it at the high end.
    var amount = 0.0;
    if (isValid(value)) {
      amount = clamp((value - look.range.x) / max(look.range.y - look.range.x, 1.0e-12), 0.0, 1.0);
    }
    radius = look.radius * (0.3 + 0.7 * amount);
  }
  var output: VertexOutput;
  output.position = vec4f(center + corner * radius * 2.0 / look.size, 0.0, 1.0);
  output.local = corner;
  output.value = value;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.local);
  if (distance > 1.0) {
    discard;
  }
  var color = colorize(input.value);
  // Invalid values take the solid color. A value faded to nothing stays invisible.
  if (look.mode < -0.5 || (color.a == 0.0 && look.extra.y < 0.5)) {
    color = look.light;
  }
  let edge = 1.0 - smoothstep(1.0 - 2.0 / max(look.radius, 1.0), 1.0, distance);
  let rim = smoothstep(0.7, 1.0, distance) * look.hillshade;
  let rgb = mix(color.rgb, look.background.rgb, rim * 0.6);
  return vec4f(rgb, color.a * edge * look.alpha);
}
`;

const SEGMENT_SHADER = /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> nodes: array<vec2f>;
@group(0) @binding(2) var<storage, read> edges: array<vec2u>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
fn readValue(index: u32) -> f32 {
  return values[index];
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) value: f32,
  @location(1) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let edge = edges[instanceIndex];
  let start = worldToClip(nodes[edge.x]);
  let end = worldToClip(nodes[edge.y]);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  var output: VertexOutput;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  let startValue = readValue(edge.x);
  let endValue = readValue(edge.y);
  output.value = select(max(startValue, endValue), mix(startValue, endValue, along[vertexIndex]), isValid(startValue) && isValid(endValue));
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (color.a == 0.0) {
    return vec4f(look.light.rgb, look.light.a * look.alpha);
  }
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(color.rgb, color.a * feather * look.alpha);
}
`;

// Segments given directly as (x0, y0, x1, y1), drawn up to a GPU-written count.
const SEGMENT_LIST_SHADER = /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> segments: array<vec4f>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.across = 0.0;
  if (instanceIndex >= counts[0]) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let segment = segments[instanceIndex];
  let start = worldToClip(segment.xy);
  let end = worldToClip(segment.zw);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`;

// A path of node ids (GPUNetworkPathExtraction output) over node positions: consecutive ids form
// segments, drawn up to a GPU-written id count.
const PATH_SHADER = /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> nodes: array<vec2f>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.across = 0.0;
  if (instanceIndex + 1u >= counts[0]) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let startPoint = nodes[ids[instanceIndex]];
  let endPoint = nodes[ids[instanceIndex + 1u]];
  let start = worldToClip(startPoint);
  let end = worldToClip(endPoint);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`;

// Triangle edges from packed uint32x3 triples (read as flat u32), drawn up to a GPU-written count.
const TRIANGLE_EDGE_SHADER = /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.across = 0.0;
  let triangle = instanceIndex / 3u;
  if (triangle >= counts[0]) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let corner = instanceIndex % 3u;
  let first = triangles[triangle * 3u + corner];
  let second = triangles[triangle * 3u + (corner + 1u) % 3u];
  let start = worldToClip(positions[first]);
  let end = worldToClip(positions[second]);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`;

export const LOOK_FLOATS = 120;

export const ALPHA_BLEND = {
  blend: true,
  blendColorOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaOperation: 'add',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

export type LookOptions = {
  /** 0 sequential, 1 diverging, 2 categorical, 3 cyclic, -1 solid color from `solid`. */
  mode?: 0 | 1 | 2 | 3 | -1;
  /** Value range mapped to the ramp. */
  range?: readonly [number, number];
  /** Dot radius or segment half width in canvas pixels. */
  radius?: number;
  alpha?: number;
  /** Raster only: 0 to 1 hillshade strength. Dots: rim strength. */
  hillshade?: number;
  /** Raster only: value change per cell that produces a 45 degree slope. */
  elevation?: number;
  /** Raster only: bilinear smoothing. */
  smoothing?: boolean;
  ramp?: readonly Color[];
  categories?: readonly (
    | readonly [number, number, number]
    | readonly [number, number, number, number]
  )[];
  /** Solid color, and the color of invalid values on segments. */
  solid?: readonly [number, number, number, number];
  lightDirection?: readonly [number, number, number];
  /** Log2 curve before the ramp (for skewed values such as flow accumulation). */
  logarithmic?: boolean;
  /** Fade alpha from 0 at the low end of the ramp to 1 at the high end. */
  fade?: boolean;
  /** Quantize the ramp into this many bands (isochrone steps). */
  bands?: number;
  /** Dots only: scale each dot's radius with its value, from 30% at the low end of the range. */
  sizeByValue?: boolean;
};

abstract class DemoLayer {
  protected readonly lookBuffer: Buffer;
  protected readonly model: Model;
  protected readonly lookValues = new Float32Array(LOOK_FLOATS);
  protected readonly device: Device;

  constructor(
    device: Device,
    id: string,
    source: string,
    bindings: Record<string, Buffer>,
    instanceCount: number
  ) {
    this.device = device;
    this.lookBuffer = device.createBuffer({
      byteLength: LOOK_FLOATS * 4,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    this.model = new Model(device, {
      id,
      source,
      topology: 'triangle-list',
      vertexCount: 6,
      instanceCount,
      bindings: {look: this.lookBuffer, ...bindings},
      parameters: {...ALPHA_BLEND}
    });
  }

  /** Uploads the look and view. Call before draw whenever either changes. */
  update(view: DemoView, theme: DemoTheme, options: LookOptions = {}): void {
    const values = this.lookValues;
    const [scaleX, scaleY] = getViewScale(view);
    const range = options.range ?? [0, 1];
    values.set([scaleX, scaleY, view.worldWidth, view.worldHeight, view.width, view.height], 0);
    values.set(range, 6);
    values[8] = options.mode ?? 0;
    values[9] = options.radius ?? 3;
    values[10] = options.alpha ?? 1;
    values[11] = options.hillshade ?? 0;
    this.writeGrid(values);
    values[14] = options.elevation ?? 1;
    values[15] = options.smoothing ? 1 : 0;
    values.set(theme.background, 16);
    const light = options.lightDirection ?? [-0.6, 0.6, 0.7];
    const solid = options.solid ?? [theme.ink[0], theme.ink[1], theme.ink[2], 1];
    // `light` doubles as the solid color for dots and the invalid-value color for segments.
    values.set(options.mode === -1 || this.usesSolid ? solid : [...light, 0], 20);
    const ramp = options.ramp ?? theme.ramp;
    for (let index = 0; index < 6; index++) {
      values.set([...ramp[Math.min(index, ramp.length - 1)], 1], 24 + index * 4);
    }
    const categories = options.categories ?? theme.categories;
    for (let index = 0; index < 16; index++) {
      const color = categories[index % categories.length];
      values.set([color[0], color[1], color[2], color.length === 4 ? color[3] : 1], 48 + index * 4);
    }
    values.set(
      [
        options.logarithmic ? 1 : 0,
        options.fade ? 1 : 0,
        options.bands ?? 0,
        options.sizeByValue ? 1 : 0
      ],
      112
    );
    values.set(view.origin ?? [0, 0], 116);
    this.lookBuffer.write(values);
  }

  protected get usesSolid(): boolean {
    return false;
  }

  /** Draws only the first `count` instances (dots, segments). */
  setInstanceCount(count: number): void {
    this.model.setInstanceCount(count);
  }

  protected writeGrid(_values: Float32Array): void {}

  draw(renderPass: RenderPass): void {
    this.model.draw(renderPass);
  }

  destroy(): void {
    this.model.destroy();
    this.lookBuffer.destroy();
  }
}

/** Colors a columns x rows float32 grid (row 0 at the top) straight from an analysis buffer. */
export class RasterLayer extends DemoLayer {
  private readonly columns: number;
  private readonly rows: number;

  constructor(
    device: Device,
    values: Buffer,
    columns: number,
    rows: number,
    valueFormat: ValueFormat = 'float32'
  ) {
    super(device, 'demo-raster', getRasterShader(valueFormat), {values}, 1);
    this.columns = columns;
    this.rows = rows;
  }

  protected override writeGrid(values: Float32Array): void {
    values[12] = this.columns;
    values[13] = this.rows;
  }
}

/** Draws one dot per position, colored by a per-dot float32 value or a solid color. */
export class DotLayer extends DemoLayer {
  constructor(
    device: Device,
    positions: Buffer,
    values: Buffer,
    count: number,
    valueFormat: ValueFormat = 'float32'
  ) {
    super(device, 'demo-dots', getDotShader(valueFormat), {positions, values}, count);
  }

  protected override get usesSolid(): boolean {
    return true;
  }
}

/** Draws edges between node positions, colored by a per-node float32 value (for example cost). */
export class SegmentLayer extends DemoLayer {
  constructor(device: Device, nodes: Buffer, edges: Buffer, values: Buffer, edgeCount: number) {
    super(device, 'demo-segments', SEGMENT_SHADER, {nodes, edges, values}, edgeCount);
  }

  protected override get usesSolid(): boolean {
    return true;
  }
}

/** Draws (x0, y0, x1, y1) segments in a solid color, as many as the GPU wrote to `counts[0]`. */
export class SegmentListLayer extends DemoLayer {
  constructor(device: Device, segments: Buffer, counts: Buffer, capacity: number) {
    super(device, 'demo-segment-list', SEGMENT_LIST_SHADER, {segments, counts}, capacity);
  }

  protected override get usesSolid(): boolean {
    return true;
  }
}

/** Draws the path `ids[0..counts[0])` of node ids as connected segments over `nodes`. */
export class PathLayer extends DemoLayer {
  constructor(device: Device, nodes: Buffer, ids: Buffer, counts: Buffer, capacity: number) {
    super(device, 'demo-path', PATH_SHADER, {nodes, ids, counts}, capacity);
  }

  protected override get usesSolid(): boolean {
    return true;
  }
}

/** Draws the three edges of every Delaunay triangle, as many as the GPU wrote to `counts[0]`. */
export class TriangleEdgeLayer extends DemoLayer {
  constructor(
    device: Device,
    positions: Buffer,
    triangles: Buffer,
    counts: Buffer,
    triangleCapacity: number
  ) {
    super(
      device,
      'demo-triangle-edges',
      TRIANGLE_EDGE_SHADER,
      {positions, triangles, counts},
      triangleCapacity * 3
    );
  }

  protected override get usesSolid(): boolean {
    return true;
  }
}

/** Clears the canvas to the theme background and draws the layers in order. */
export function drawLayers(device: Device, theme: DemoTheme, layers: readonly DemoLayer[]): void {
  const renderPass = device.beginRenderPass({
    clearColor: [...theme.background] as [number, number, number, number]
  });
  for (const layer of layers) {
    layer.draw(renderPass);
  }
  renderPass.end();
  device.submit();
}

export {DemoLayer};

/**
 * Coalesces redraw requests to one per animation frame. Controls call `request()`; the scene's own
 * render loop calls `flush()`, which cancels a pending request so a frame never draws twice.
 */
export class DrawScheduler {
  private readonly draw: () => void;
  private handle = 0;

  constructor(draw: () => void) {
    this.draw = draw;
  }

  request(): void {
    if (this.handle) return;
    this.handle = requestAnimationFrame(() => {
      this.handle = 0;
      this.draw();
    });
  }

  flush(): void {
    this.cancel();
    this.draw();
  }

  cancel(): void {
    if (this.handle) cancelAnimationFrame(this.handle);
    this.handle = 0;
  }
}

/** Throws unless the device runs WebGPU; the canvas keeps its fallback otherwise. */
export function assertWebGPU(device: Device): void {
  if (device.type !== 'webgpu') {
    // Demos run real compute kernels, which need WebGPU.
    throw new Error('WebGPU required');
  }
}
