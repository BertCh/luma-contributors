// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A dispatch as a picture: a grid of workgroups, each a grid of lanes (invocations). A front of
// concurrent workgroups sweeps the grid in launch order; lanes past the dispatch's invocation count
// stay hollow, because kernels guard `if (index >= count) return;`.

import type {Buffer, Device} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GpuScene, GpuSceneLoader} from '../../../gpu/types';
import {
  assertWebGpu,
  createQuadModel,
  createUniformBuffer,
  getSceneColors,
  STILL_FRAME_MILLISECONDS
} from './scene-kit';

export type WorkgroupGrid = {
  columns: number;
  rows: number;
  /** Lanes per side inside a workgroup (8 means 64 invocations). */
  laneSide: number;
  /** Invocations the kernel guards to; the rest of the last workgroups are idle. */
  invocationCount: number;
  /** Workgroups running at the same time. */
  concurrency: number;
};

export const DEFAULT_WORKGROUP_GRID: WorkgroupGrid = {
  columns: 16,
  rows: 8,
  laneSide: 8,
  invocationCount: 16 * 8 * 64 - 411,
  concurrency: 7
};

const LOOK_FLOATS = 24;
const CYCLE_SECONDS = 10;
const SWEEP_SECONDS = 6.5;

const WGSL = /* wgsl */ `
struct Look {
  grid: vec4f,
  clock: vec4f,
  size: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;

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

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let columns = look.grid.x;
  let rows = look.grid.y;
  let laneSide = look.grid.z;
  let invocationCount = look.grid.w;
  let progress = look.clock.y;
  let concurrency = look.clock.z;
  let fade = look.clock.w;

  let cellUv = input.uv * vec2f(columns, rows);
  let cell = floor(cellUv);
  let inCell = fract(cellUv);
  let workgroup = cell.y * columns + cell.x;
  let age = progress - workgroup;

  let gap = 0.09;
  let inner = (inCell - vec2f(gap)) / (1.0 - 2.0 * gap);
  var color = look.background.rgb;
  var alpha = 0.0;

  // Workgroup frame.
  let frameDistance = max(abs(inCell.x - 0.5), abs(inCell.y - 0.5));
  let frame = smoothstep(0.5 - gap * 0.55, 0.5 - gap * 0.45, frameDistance) *
    (1.0 - smoothstep(0.5 - gap * 0.2, 0.5 - gap * 0.05, frameDistance));
  var frameStrength = 0.14;
  if (age >= 0.0 && age < concurrency) {
    frameStrength = 0.9;
  } else if (age >= concurrency) {
    frameStrength = 0.3 * fade;
  }
  color = look.accent.rgb;
  alpha = frame * frameStrength;

  if (inner.x >= 0.0 && inner.x < 1.0 && inner.y >= 0.0 && inner.y < 1.0) {
    let lanePosition = inner * laneSide;
    let lane = floor(lanePosition);
    let inLane = fract(lanePosition) - vec2f(0.5);
    let laneIndex = lane.y * laneSide + lane.x;
    let laneCount = laneSide * laneSide;
    let invocation = workgroup * laneCount + laneIndex;
    let radius = length(inLane);
    let disc = 1.0 - smoothstep(0.27, 0.36, radius);
    let ring = disc - (1.0 - smoothstep(0.17, 0.26, radius));
    let guarded = invocation >= invocationCount;

    if (guarded) {
      color = look.signal.rgb;
      alpha = max(alpha, ring * 0.7 * select(0.4, 1.0, age >= 0.0));
    } else if (age < 0.0) {
      color = look.accent.rgb;
      alpha = max(alpha, disc * 0.16);
    } else if (age < concurrency) {
      let reach = clamp(age / concurrency, 0.0, 1.0);
      let lit = select(0.0, 1.0, laneIndex / laneCount <= reach);
      color = mix(look.accent.rgb, vec3f(1.0), 0.35 * lit);
      alpha = max(alpha, disc * mix(0.2, 1.0, lit));
    } else {
      color = look.accent.rgb;
      alpha = max(alpha, disc * 0.5 * fade);
    }
  }
  return vec4f(color, clamp(alpha, 0.0, 1.0));
}
`;

export function createWorkgroupSceneLoader(grid: WorkgroupGrid): GpuSceneLoader {
  return async () => ({default: createWorkgroupScene(grid)});
}

export function createWorkgroupScene(grid: WorkgroupGrid): GpuScene {
  return ({device, theme, width, height, reducedMotion}) =>
    new WorkgroupSceneInstance(device, grid, theme, width, height, reducedMotion);
}

class WorkgroupSceneInstance {
  private readonly device: Device;
  private readonly grid: WorkgroupGrid;
  private readonly colors: ReturnType<typeof getSceneColors>;
  private readonly look = new Float32Array(LOOK_FLOATS);
  private readonly lookBuffer: Buffer;
  private readonly model: Model;
  private readonly reducedMotion: boolean;
  private width: number;
  private height: number;

  constructor(
    device: Device,
    grid: WorkgroupGrid,
    theme: 'light' | 'dark',
    width: number,
    height: number,
    reducedMotion: boolean
  ) {
    assertWebGpu(device);
    this.device = device;
    this.grid = grid;
    this.colors = getSceneColors(theme);
    this.width = width;
    this.height = height;
    this.reducedMotion = reducedMotion;
    this.lookBuffer = createUniformBuffer(device, LOOK_FLOATS);
    this.model = createQuadModel(device, {
      id: 'workgroup-grid',
      source: WGSL,
      bindings: {look: this.lookBuffer},
      instanceCount: 1,
      theme
    });
  }

  render(timeMilliseconds: number): void {
    const seconds = (this.reducedMotion ? STILL_FRAME_MILLISECONDS * 1.4 : timeMilliseconds) / 1000;
    const cycle = seconds % CYCLE_SECONDS;
    const total = this.grid.columns * this.grid.rows;
    const sweep = Math.min(1, cycle / SWEEP_SECONDS);
    const progress = sweep * (total + this.grid.concurrency);
    // After the sweep the finished grid holds, then fades before the next launch.
    const fade = cycle < CYCLE_SECONDS - 1 ? 1 : Math.max(0, CYCLE_SECONDS - cycle);
    const look = this.look;
    look.set([this.grid.columns, this.grid.rows, this.grid.laneSide, this.grid.invocationCount], 0);
    look.set([seconds, progress, this.grid.concurrency, fade], 4);
    look.set([this.width, this.height, 0, 0], 8);
    look.set(this.colors.background, 12);
    look.set(this.colors.accent, 16);
    look.set(this.colors.signal, 20);
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

export default createWorkgroupScene(DEFAULT_WORKGROUP_GRID);
