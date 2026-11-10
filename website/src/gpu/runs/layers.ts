// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Layers for declarative runs that the flagship demo kit does not have: filled polygon meshes,
// segments with typed values (categorical labels), and charts (bars, line, scatter) for scalar and
// table outputs. They extend the kit's DemoLayer, so they read analysis output buffers directly
// and share its look (ramps, categories, solid colors).

import {Buffer, type Device} from '@luma.gl/core';
import {DemoLayer, getValues, PALETTE_STRUCT, type ValueFormat} from '../demos/kit';

// Triangle list over `positions` (vec2f) given by `indices`; triangle t takes the value of row
// `owners[t]` (the polygon the triangle belongs to). Invalid values draw in the solid color.
const getMeshShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> owners: array<u32>;
${getValues(format, 4)}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) value: f32
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let point = positions[indices[vertexIndex]];
  var output: VertexOutput;
  output.position = vec4f(worldToClip(point), 0.0, 1.0);
  output.value = readValue(owners[vertexIndex / 3u]);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (look.mode < -0.5 || color.a == 0.0) {
    color = look.light;
  }
  return vec4f(color.rgb, color.a * look.alpha);
}
`;

/** Filled polygons: a triangle mesh whose triangles are colored by their owner polygon's value. */
export class MeshLayer extends DemoLayer {
  constructor(
    device: Device,
    positions: Buffer,
    indices: Buffer,
    owners: Buffer,
    values: Buffer,
    triangleCount: number,
    valueFormat: ValueFormat
  ) {
    super(device, 'run-mesh', getMeshShader(valueFormat), {positions, indices, owners, values}, 1);
    this.model.setVertexCount(Math.max(triangleCount, 1) * 3);
  }

  protected get usesSolid(): boolean {
    return true;
  }
}

// Network edges (node pairs) colored by a per-node value. Categorical labels are not interpolated.
const getEdgeShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
@group(0) @binding(1) var<storage, read> nodes: array<vec2f>;
@group(0) @binding(2) var<storage, read> edges: array<vec2u>;
${getValues(format, 3)}

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
  if (look.mode > 1.5) {
    output.value = select(endValue, startValue, isValid(startValue));
  } else {
    output.value = select(max(startValue, endValue), mix(startValue, endValue, along[vertexIndex]), isValid(startValue) && isValid(endValue));
  }
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (look.mode < -0.5 || color.a == 0.0) {
    return vec4f(look.light.rgb, look.light.a * look.alpha);
  }
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(color.rgb, color.a * feather * look.alpha);
}
`;

/** Edges between node positions, colored by a float32 or uint32 value per node. */
export class EdgeLayer extends DemoLayer {
  constructor(
    device: Device,
    nodes: Buffer,
    edges: Buffer,
    values: Buffer,
    edgeCount: number,
    valueFormat: ValueFormat
  ) {
    super(device, 'run-edges', getEdgeShader(valueFormat), {nodes, edges, values}, edgeCount);
  }

  protected get usesSolid(): boolean {
    return true;
  }
}

export type ChartKind = 'bars' | 'line' | 'scatter';

const CHART_PLOT = /* wgsl */ `
// plot[0]: rectangle in world units (x0, y0, x1, y1); plot[1]: x range (min, max) for scatter.
@group(0) @binding(2) var<uniform> plot: array<vec4f, 2>;

fn plotPoint(x: f32, normalizedY: f32) -> vec2f {
  let world = vec2f(mix(plot[0].x, plot[0].z, x), mix(plot[0].y, plot[0].w, normalizedY));
  return worldToClip(world);
}

fn normalizeValue(value: f32) -> f32 {
  var span = look.range.y - look.range.x;
  if (abs(span) < 1.0e-12) {
    span = 1.0e-12;
  }
  return clamp((value - look.range.x) / span, 0.0, 1.0);
}
`;

const getBarsShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
${getValues(format, 1)}
${CHART_PLOT}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) value: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let count = f32(arrayLength(&values));
  let value = readValue(instanceIndex);
  let height = select(0.0, normalizeValue(value), isValid(value));
  let x = (f32(instanceIndex) + 0.1 + corner.x * 0.8) / count;
  var output: VertexOutput;
  output.position = vec4f(plotPoint(x, corner.y * height), 0.0, 1.0);
  output.value = value;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (look.mode < -0.5 || color.a == 0.0) {
    color = look.light;
  }
  return vec4f(color.rgb, color.a * look.alpha);
}
`;

const getLineShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
${getValues(format, 1)}
${CHART_PLOT}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let count = f32(arrayLength(&values));
  let first = readValue(instanceIndex);
  let second = readValue(instanceIndex + 1u);
  let start = plotPoint((f32(instanceIndex) + 0.5) / count, normalizeValue(first));
  let end = plotPoint((f32(instanceIndex) + 1.5) / count, normalizeValue(second));
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
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`;

const getScatterShader = (format: ValueFormat) => /* wgsl */ `
${PALETTE_STRUCT}
${getValues(format, 1)}
${CHART_PLOT}
@group(0) @binding(3) var<storage, read> xValues: array<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f
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
  var spanX = plot[1].y - plot[1].x;
  if (abs(spanX) < 1.0e-12) {
    spanX = 1.0e-12;
  }
  let x = clamp((xValues[instanceIndex] - plot[1].x) / spanX, 0.0, 1.0);
  let center = plotPoint(x, normalizeValue(readValue(instanceIndex)));
  var output: VertexOutput;
  output.position = vec4f(center + corner * look.radius * 2.0 / look.size, 0.0, 1.0);
  output.local = corner;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.local);
  if (distance > 1.0) {
    discard;
  }
  let edge = 1.0 - smoothstep(1.0 - 2.0 / max(look.radius, 1.0), 1.0, distance);
  return vec4f(look.light.rgb, look.light.a * edge * look.alpha);
}
`;

/**
 * A chart of a scalar or table output, drawn from its GPU buffer. Bars and line plot `values[i]`
 * at position i; scatter plots `values[i]` against `xValues[i]`. The y range comes from the look's
 * `range`; call `setPlot` for the rectangle (world units) and the scatter x range.
 */
export class ChartLayer extends DemoLayer {
  private readonly plotBuffer: Buffer;
  private readonly plotValues = new Float32Array(8);

  constructor(
    device: Device,
    kind: ChartKind,
    values: Buffer,
    xValues: Buffer,
    count: number,
    valueFormat: ValueFormat
  ) {
    const plotBuffer = device.createBuffer({
      byteLength: 32,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const shader =
      kind === 'bars'
        ? getBarsShader(valueFormat)
        : kind === 'line'
          ? getLineShader(valueFormat)
          : getScatterShader(valueFormat);
    super(
      device,
      `run-chart-${kind}`,
      shader,
      kind === 'scatter' ? {values, plot: plotBuffer, xValues} : {values, plot: plotBuffer},
      kind === 'line' ? Math.max(count - 1, 0) : count
    );
    this.plotBuffer = plotBuffer;
  }

  protected get usesSolid(): boolean {
    return true;
  }

  setPlot(rectangle: readonly [number, number, number, number], xRange: readonly [number, number]) {
    this.plotValues.set(rectangle, 0);
    this.plotValues.set(xRange, 4);
    this.plotBuffer.write(this.plotValues);
  }

  destroy(): void {
    super.destroy();
    this.plotBuffer.destroy();
  }
}
