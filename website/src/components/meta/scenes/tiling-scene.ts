// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A dispatch mapped onto data: each square is one invocation and the data element it owns, colored
// by workgroup. The workgroup size comes from a shared store, so the grid re-tiles as a slider moves;
// the hovered workgroup stays bright. Lanes past the element count are guarded out (hollow).

import type {Buffer, Device} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GpuScene, GpuSceneLoader} from '../../../gpu/types';
import {KERNEL_GRID} from '../kernel-source';
import {
  assertWebGpu,
  createQuadModel,
  createUniformBuffer,
  getSceneColors,
  RAMP_WGSL,
  STILL_FRAME_MILLISECONDS
} from './scene-kit';

/** State shared between the React controls and the scene. Plain data: the scene reads it each frame. */
export type TilingStore = {
  workgroupSize: number;
  /** Hovered workgroup index, or -1. */
  hovered: number;
};

export function createTilingStore(workgroupSize: number): TilingStore {
  return {workgroupSize, hovered: -1};
}

const LOOK_FLOATS = 28;
const SWEEP_SECONDS = 4.5;
const CYCLE_SECONDS = 6.5;

const WGSL = /* wgsl */ `
struct Look {
  grid: vec4f,
  clock: vec4f,
  view: vec4f,
  view2: vec4f,
  background: vec4f,
  signal: vec4f,
  ink: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
${RAMP_WGSL}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  return vec4f(corners[vertexIndex] * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let columns = look.grid.x;
  let rows = look.grid.y;
  let size = look.grid.z;
  let count = look.grid.w;
  let hovered = look.clock.y;
  let front = look.clock.z;
  let concurrency = look.clock.w;
  let local = (position.xy - vec2f(look.view.w, look.view2.x)) / look.view.z;
  if (local.x < 0.0 || local.y < 0.0 || local.x >= columns || local.y >= rows) {
    return vec4f(0.0);
  }
  let cell = floor(local);
  let inCell = fract(local);
  let index = cell.y * columns + cell.x;
  let workgroup = floor(index / size);
  let workgroupCount = ceil(count / size);
  let offset = abs(inCell - vec2f(0.5));
  let distance = max(offset.x, offset.y);
  let body = 1.0 - smoothstep(0.38, 0.45, distance);
  let hole = 1.0 - smoothstep(0.27, 0.34, distance);

  if (index >= count) {
    return vec4f(look.signal.rgb, (body - hole) * 0.85);
  }

  var color = sampleRamp(workgroup / max(workgroupCount - 1.0, 1.0));
  color = mix(color, look.ink.rgb, select(0.0, 0.32, (u32(workgroup) & 1u) == 1u));
  var alpha = 0.82;
  if (hovered >= 0.0) {
    alpha = select(0.16, 1.0, abs(hovered - workgroup) < 0.5);
  } else {
    let age = front - workgroup;
    if (age >= 0.0 && age < concurrency) {
      color = mix(color, vec3f(1.0), 0.55);
      alpha = 1.0;
    } else if (age < 0.0) {
      alpha = 0.3;
    }
  }
  // The first invocation of a workgroup carries a bar on its left edge.
  let isStart = (u32(index) % u32(size)) == 0u;
  let bar = select(0.0, 1.0 - smoothstep(0.1, 0.16, inCell.x), isStart);
  color = mix(color, look.ink.rgb, bar * 0.9);
  return vec4f(color, body * alpha);
}
`;

export function createTilingSceneLoader(store: TilingStore): GpuSceneLoader {
  return async () => ({default: createTilingScene(store)});
}

export function createTilingScene(store: TilingStore): GpuScene {
  return ({device, theme, width, height, reducedMotion}) =>
    new TilingSceneInstance(device, store, theme, width, height, reducedMotion);
}

class TilingSceneInstance {
  private readonly device: Device;
  private readonly store: TilingStore;
  private readonly colors: ReturnType<typeof getSceneColors>;
  private readonly look = new Float32Array(LOOK_FLOATS);
  private readonly lookBuffer: Buffer;
  private readonly model: Model;
  private readonly reducedMotion: boolean;
  private width: number;
  private height: number;

  constructor(
    device: Device,
    store: TilingStore,
    theme: 'light' | 'dark',
    width: number,
    height: number,
    reducedMotion: boolean
  ) {
    assertWebGpu(device);
    this.device = device;
    this.store = store;
    this.colors = getSceneColors(theme);
    this.width = width;
    this.height = height;
    this.reducedMotion = reducedMotion;
    this.lookBuffer = createUniformBuffer(device, LOOK_FLOATS);
    this.model = createQuadModel(device, {
      id: 'kernel-tiling',
      source: WGSL,
      bindings: {look: this.lookBuffer},
      instanceCount: 1,
      theme
    });
  }

  render(timeMilliseconds: number): void {
    const {columns, rows, invocationCount} = KERNEL_GRID;
    const size = this.store.workgroupSize;
    const workgroupCount = Math.ceil(invocationCount / size);
    const seconds = (this.reducedMotion ? STILL_FRAME_MILLISECONDS : timeMilliseconds) / 1000;
    const cycle = seconds % CYCLE_SECONDS;
    const concurrency = Math.max(2, Math.round(workgroupCount / 8));
    // Past the end of the sweep every workgroup counts as finished.
    const front =
      this.reducedMotion || cycle >= SWEEP_SECONDS
        ? 1e6
        : (cycle / SWEEP_SECONDS) * (workgroupCount + concurrency);
    const cellSize = Math.min(this.width / columns, this.height / rows);
    const originX = (this.width - cellSize * columns) / 2;
    const originY = (this.height - cellSize * rows) / 2;
    const look = this.look;
    look.set([columns, rows, size, invocationCount], 0);
    look.set([seconds, this.store.hovered, front, concurrency], 4);
    look.set([this.width, this.height, cellSize, originX], 8);
    look.set([originY, 0, 0, 0], 12);
    look.set(this.colors.background, 16);
    look.set(this.colors.signal, 20);
    look.set(this.colors.ink, 24);
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
  }
}
