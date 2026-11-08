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
  ScenePointer,
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

const LOOK_FLOATS = 32;
const CYCLE_SECONDS = 12;

const WGSL = /* wgsl */ `
struct Look {
  world: vec2f,
  size: vec2f,
  clock: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  ink: vec4f,
  // x: hovered chunk order, y: hovered row, z: pointer presence
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
@group(0) @binding(1) var<storage, read> rects: array<vec4f>;
@group(0) @binding(2) var<storage, read> styles: array<vec4f>;
${RAMP_WGSL}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) corner: vec2f,
  @location(1) style: vec4f,
  @location(2) cellPixels: vec2f
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
  output.cellPixels = rect.zw / look.world * look.size;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let time = look.clock.x;
  let presence = look.pointer.z;
  let chunkOrder = input.style.y;
  let row = input.style.z;
  // Hovering shows the finished table, so the upload and the closing fade are overridden.
  let appear = mix(smoothstep(0.0, 0.35, time - chunkOrder * 0.45), 1.0, presence);
  let fade = mix(look.clock.w, 1.0, presence);
  // Dispatch phase: a band sweeps down the rows, then locks onto the hovered row.
  let bandRow = mix((time - 5.2) * 5.5, look.pointer.y, presence);
  let bandWidth = mix(1.6, 0.7, presence);
  let band = exp(-pow((row - bandRow) / bandWidth, 2.0));
  let rampColor = sampleRamp(input.style.x);
  var color = mix(rampColor, look.accent.rgb, band * 0.8);
  let inset = min(min(input.corner.x, 1.0 - input.corner.x), min(input.corner.y, 1.0 - input.corner.y));
  let fill = smoothstep(0.0, 0.12, inset);
  let flash = exp(-pow((time - chunkOrder * 0.45 - 0.2) / 0.25, 2.0)) * (1.0 - presence);
  color = mix(color, vec3f(1.0), flash * 0.5);
  // The hovered chunk is one GPU buffer: brighten and outline it; dim other cells off the band row.
  let hovered = select(0.0, 1.0, abs(chunkOrder - look.pointer.x) < 0.5) * presence;
  // Ink is light on dark surfaces only, so the tint brightens there and fades out on light ones.
  color = mix(color, look.ink.rgb, hovered * 0.25 * dot(look.ink.rgb, vec3f(1.0 / 3.0)));
  let edgeDistances = min(input.corner, vec2f(1.0) - input.corner) * input.cellPixels;
  let outline = (1.0 - smoothstep(0.75, 1.5, min(edgeDistances.x, edgeDistances.y))) * hovered;
  var strength = (0.5 + 0.5 * input.style.x) * (0.8 + 0.2 * band) * fade;
  strength = clamp(strength + hovered * 0.35, 0.0, 1.0) * (1.0 - 0.65 * (presence - hovered) * (1.0 - band));
  let alpha = max(appear * fill * strength, outline * appear);
  return vec4f(mix(color, look.ink.rgb, outline), alpha);
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
  private readonly pointer: ScenePointer;
  private hoveredChunkOrder = 0;
  private hoveredRow = 0;
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
    this.pointer = new ScenePointer(reducedMotion);
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

  setPointer(position: [number, number] | null): void {
    this.pointer.set(position);
  }

  /** Finds the cell nearest the pointer, so crossing the gaps between cells never flickers. */
  private updateHoveredCell(): void {
    const {cells, width: worldWidth, height: worldHeight} = this.layout;
    const worldX = (this.pointer.x / this.width) * worldWidth;
    const worldY = (this.pointer.y / this.height) * worldHeight;
    let nearestDistance = Infinity;
    for (const cell of cells) {
      const deltaX = Math.max(cell.x - worldX, 0, worldX - (cell.x + cell.width));
      const deltaY = Math.max(cell.y - worldY, 0, worldY - (cell.y + cell.height));
      const distance = deltaX * deltaX + deltaY * deltaY;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        this.hoveredChunkOrder = cell.chunkOrder;
        this.hoveredRow = cell.row;
      }
    }
  }

  render(timeMilliseconds: number): void {
    this.pointer.update(timeMilliseconds / 1000);
    // Keep the last hovered cell while the pointer fades out.
    if (this.pointer.isActive) this.updateHoveredCell();
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
    look.set(this.colors.ink, 20);
    look.set([this.hoveredChunkOrder, this.hoveredRow, this.pointer.presence, 0], 24);
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
