// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A table on the GPU: every column is a vector, every chunk of a column is one buffer. Chunks
// upload one by one (streaming keeps batch boundaries), then a dispatch reads rows across columns.

import type {Buffer, Device} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GpuScene, GpuSceneLoader} from '../../../gpu/types';
import {
  assertWebGpu,
  createQuadModel,
  createStorageBuffer,
  createUniformBuffer,
  getSceneColors,
  RAMP_WGSL,
  STILL_FRAME_MILLISECONDS
} from './scene-kit';

export type ColumnCell = {
  x: number;
  y: number;
  width: number;
  height: number;
  /** 0..1, colors the cell. */
  value: number;
  /** Upload order of the chunk (GPUData) this cell lives in. */
  chunkOrder: number;
  /** Row from the top of the table, for the dispatch sweep. */
  row: number;
};

export type ColumnsLayout = {width: number; height: number; rowCount: number; cells: ColumnCell[]};

const LOOK_FLOATS = 24;
const CYCLE_SECONDS = 12;

const WGSL = /* wgsl */ `
struct Look {
  world: vec2f,
  size: vec2f,
  clock: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
@group(0) @binding(1) var<storage, read> rects: array<vec4f>;
@group(0) @binding(2) var<storage, read> styles: array<vec4f>;
${RAMP_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) corner: vec2f,
  @location(1) style: vec4f
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let rect = rects[instanceIndex];
  let corner = corners[vertexIndex];
  let position = rect.xy + corner * rect.zw;
  var output: VertexOutput;
  output.position = vec4f(position.x / look.world.x * 2.0 - 1.0, 1.0 - position.y / look.world.y * 2.0, 0.0, 1.0);
  output.corner = corner;
  output.style = styles[instanceIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let time = look.clock.x;
  let chunkOrder = input.style.y;
  let row = input.style.z;
  // Upload phase: chunk n appears at 0.45 * n seconds. Dispatch phase: a band sweeps down the rows.
  let appear = smoothstep(0.0, 0.35, time - chunkOrder * 0.45);
  let band = exp(-pow((row - (time - 5.2) * 5.5) / 1.6, 2.0));
  let rampColor = sampleRamp(input.style.x);
  var color = mix(rampColor, look.accent.rgb, band * 0.8);
  let inset = min(min(input.corner.x, 1.0 - input.corner.x), min(input.corner.y, 1.0 - input.corner.y));
  let fill = smoothstep(0.0, 0.12, inset);
  let flash = exp(-pow((time - chunkOrder * 0.45 - 0.2) / 0.25, 2.0));
  color = mix(color, vec3f(1.0), flash * 0.5);
  let strength = (0.5 + 0.5 * input.style.x) * (0.8 + 0.2 * band) * look.clock.w;
  return vec4f(color, appear * fill * clamp(strength, 0.0, 1.0));
}
`;

export function createColumnsSceneLoader(layout: ColumnsLayout): GpuSceneLoader {
  return async () => ({default: createColumnsScene(layout)});
}

export function createColumnsScene(layout: ColumnsLayout): GpuScene {
  return ({device, theme, width, height, reducedMotion}) =>
    new ColumnsSceneInstance(device, layout, theme, width, height, reducedMotion);
}

class ColumnsSceneInstance {
  private readonly device: Device;
  private readonly layout: ColumnsLayout;
  private readonly colors: ReturnType<typeof getSceneColors>;
  private readonly look = new Float32Array(LOOK_FLOATS);
  private readonly lookBuffer: Buffer;
  private readonly buffers: Buffer[] = [];
  private readonly model: Model;
  private readonly reducedMotion: boolean;
  private width: number;
  private height: number;

  constructor(
    device: Device,
    layout: ColumnsLayout,
    theme: 'light' | 'dark',
    width: number,
    height: number,
    reducedMotion: boolean
  ) {
    assertWebGpu(device);
    this.device = device;
    this.layout = layout;
    this.colors = getSceneColors(theme);
    this.width = width;
    this.height = height;
    this.reducedMotion = reducedMotion;
    this.lookBuffer = createUniformBuffer(device, LOOK_FLOATS);
    const rects = new Float32Array(Math.max(1, layout.cells.length) * 4);
    const styles = new Float32Array(Math.max(1, layout.cells.length) * 4);
    layout.cells.forEach((cell, index) => {
      rects.set([cell.x, cell.y, cell.width, cell.height], index * 4);
      styles.set([cell.value, cell.chunkOrder, cell.row, 0], index * 4);
    });
    const rectsBuffer = createStorageBuffer(device, rects);
    const stylesBuffer = createStorageBuffer(device, styles);
    this.buffers.push(rectsBuffer, stylesBuffer);
    this.model = createQuadModel(device, {
      id: 'columns',
      source: WGSL,
      bindings: {look: this.lookBuffer, rects: rectsBuffer, styles: stylesBuffer},
      instanceCount: layout.cells.length,
      theme
    });
    if (layout.cells.length === 0) this.model.setInstanceCount(0);
  }

  render(timeMilliseconds: number): void {
    const seconds = this.reducedMotion
      ? (STILL_FRAME_MILLISECONDS * 3.2) / 1000
      : (timeMilliseconds / 1000) % CYCLE_SECONDS;
    // Fade out in the last second so the loop restarts from an empty table.
    const fade = this.reducedMotion ? 1 : Math.min(1, Math.max(0, CYCLE_SECONDS - seconds));
    const look = this.look;
    look.set([this.layout.width, this.layout.height, this.width, this.height], 0);
    look.set([seconds, 0, 0, fade], 4);
    look.set(this.colors.background, 8);
    look.set(this.colors.accent, 12);
    look.set(this.colors.signal, 16);
    this.lookBuffer.write(look);
    const renderPass = this.device.beginRenderPass({clearColor: this.colors.background});
    this.model.draw(renderPass);
    renderPass.end();
    this.device.submit();
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  destroy(): void {
    this.model.destroy();
    this.lookBuffer.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

export default createColumnsScene({width: 1, height: 1, rowCount: 1, cells: []});
