// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A dispatch as a picture: a grid of workgroups, each a grid of lanes (invocations). A front of
// concurrent workgroups sweeps the grid in launch order; lanes past the dispatch's invocation count
// stay hollow, because kernels guard `if (index >= count) return;`.
//
// Optional pointer interactions (`interaction` option, off by default): 'dispatch' launches the
// hovered workgroup lane by lane, 'trail' leaves a fading trail of redrawn workgroups.

import type {Buffer, Device} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import type {GpuScene, GpuSceneLoader} from '../../../gpu/types';
import {
  assertWebGpu,
  type Color4,
  createQuadModel,
  createUniformBuffer,
  getSceneColors,
  type SceneColors,
  createStorageBuffer,
  ScenePointer,
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

export type WorkgroupSceneOptions = {
  /** 'dispatch' launches the hovered workgroup; 'trail' marks workgroups the pointer crossed. */
  interaction?: 'none' | 'dispatch' | 'trail';
};

const LOOK_FLOATS = 32;
const INTERACTION_MODES = {none: 0, dispatch: 1, trail: 2} as const;
/** Seconds per replay of the hovered workgroup's lane fill, and the share of it spent filling. */
const LAUNCH_PERIOD_SECONDS = 0.6;
const LAUNCH_FILL_SECONDS = 0.45;
/** Largest step, in cells, when walking the pointer path so fast motion skips no cell. */
const TRAIL_STEP_CELLS = 0.25;
const UNTOUCHED_SECONDS = -1e6;
const CYCLE_SECONDS = 10;
const SWEEP_SECONDS = 6.5;

const WGSL = /* wgsl */ `
struct Look {
  grid: vec4f,
  clock: vec4f,
  size: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  highlight: vec4f,
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
@group(0) @binding(1) var<storage, read> touched: array<f32>;

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

  // Pointer interaction state; everything below is gated by mode, so mode 0 is the plain picture.
  let mode = look.size.z;
  let isDispatch = mode > 0.5 && mode < 1.5;
  let isTrail = mode > 1.5;
  let presence = look.pointer.z;
  let isHovered = isDispatch && workgroup == look.size.w;
  let pointerOffset = cell + vec2f(0.5) - look.pointer.xy;
  let glow = select(0.0, exp(-dot(pointerOffset, pointerOffset) / 1.6) * presence, isDispatch);
  var heat = 0.0;
  if (isTrail) {
    heat = exp(-max(look.pointer.w - touched[u32(workgroup)], 0.0) / 0.8);
    frameStrength = max(frameStrength * (1.0 - 0.3 * presence), heat);
  }
  if (isDispatch) {
    frameStrength = frameStrength * (1.0 - 0.5 * presence) + glow * 0.6;
    if (isHovered) {
      frameStrength = mix(frameStrength, 1.0, presence);
    }
  }
  color = look.accent.rgb;
  alpha = frame * frameStrength;
  let frameAlpha = alpha;

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

    if (isDispatch) {
      if (!guarded) {
        alpha = alpha * (1.0 - 0.5 * presence) + disc * glow * 0.35;
      }
      if (isHovered) {
        // Lanes fill in launch order up to reach; guarded lanes show as hollow signal rings.
        let reach = look.pointer.w;
        let lanePosition = (laneIndex + 1.0) / laneCount;
        let isLit = lanePosition <= reach;
        let isFront = isLit && reach < 1.0 && lanePosition > reach - 1.0 / laneCount;
        var hoverColor = look.signal.rgb;
        var hoverAlpha = max(frameAlpha, ring * 0.95);
        if (!guarded) {
          hoverColor = mix(look.accent.rgb, look.highlight.rgb, select(0.0, select(0.35, 0.75, isFront), isLit));
          hoverAlpha = max(frameAlpha, disc * select(0.2, 1.0, isLit));
        }
        color = mix(color, hoverColor, presence);
        alpha = mix(alpha, hoverAlpha, presence);
      }
    }

    if (isTrail) {
      if (guarded) {
        color = mix(color, mix(look.signal.rgb, look.highlight.rgb, 0.4), heat);
        alpha = mix(alpha, max(alpha, ring), heat);
      } else {
        alpha = mix(alpha * (1.0 - 0.3 * presence), max(alpha, disc), heat);
        color = mix(color, mix(look.accent.rgb, look.highlight.rgb, 0.55), heat);
      }
    }
  }
  return vec4f(color, clamp(alpha, 0.0, 1.0));
}
`;

export function createWorkgroupSceneLoader(
  grid: WorkgroupGrid,
  options: WorkgroupSceneOptions = {}
): GpuSceneLoader {
  return async () => ({default: createWorkgroupScene(grid, options)});
}

export function createWorkgroupScene(
  grid: WorkgroupGrid,
  options: WorkgroupSceneOptions = {}
): GpuScene {
  return ({device, theme, width, height, reducedMotion}) =>
    new WorkgroupSceneInstance(
      device,
      grid,
      options.interaction ?? 'none',
      theme,
      width,
      height,
      reducedMotion
    );
}

function getHighlightColor(colors: SceneColors, theme: 'light' | 'dark'): Color4 {
  if (theme === 'dark') return [1, 1, 1, 1];
  const [red, green, blue] = colors.accent;
  const [inkRed, inkGreen, inkBlue] = colors.ink;
  return [(red + inkRed) / 2, (green + inkGreen) / 2, (blue + inkBlue) / 2, 1];
}

class WorkgroupSceneInstance {
  private readonly device: Device;
  private readonly grid: WorkgroupGrid;
  private readonly interaction: keyof typeof INTERACTION_MODES;
  private readonly colors: ReturnType<typeof getSceneColors>;
  /** Lit-lane tint of the pointer effects: white glow on dark, deep accent on light. */
  private readonly highlight: Color4;
  private readonly look = new Float32Array(LOOK_FLOATS);
  private readonly lookBuffer: Buffer;
  private readonly lastTouched: Float32Array;
  private readonly touchedBuffer: Buffer;
  private readonly model: Model;
  private readonly reducedMotion: boolean;
  private readonly pointer: ScenePointer;
  private width: number;
  private height: number;
  // Dispatch state: the last hovered workgroup (kept while presence fades) and when it launched.
  private hoveredIndex = -1;
  private launchSeconds = 0;
  private wasActive = false;
  // Trail state: the previous eased pointer position in cell units, and whether any cell is marked.
  private previousCell: [number, number] | null = null;
  private hasTouches = false;

  constructor(
    device: Device,
    grid: WorkgroupGrid,
    interaction: keyof typeof INTERACTION_MODES,
    theme: 'light' | 'dark',
    width: number,
    height: number,
    reducedMotion: boolean
  ) {
    assertWebGpu(device);
    this.device = device;
    this.grid = grid;
    this.interaction = interaction;
    this.colors = getSceneColors(theme);
    this.highlight = getHighlightColor(this.colors, theme);
    this.width = width;
    this.height = height;
    this.reducedMotion = reducedMotion;
    this.pointer = new ScenePointer(reducedMotion);
    this.lookBuffer = createUniformBuffer(device, LOOK_FLOATS);
    this.lastTouched = new Float32Array(grid.columns * grid.rows).fill(UNTOUCHED_SECONDS);
    this.touchedBuffer = createStorageBuffer(device, this.lastTouched);
    this.model = createQuadModel(device, {
      id: 'workgroup-grid',
      source: WGSL,
      bindings: {look: this.lookBuffer, touched: this.touchedBuffer},
      instanceCount: 1,
      theme
    });
  }

  setPointer(position: [number, number] | null): void {
    this.pointer.set(position);
  }

  render(timeMilliseconds: number): void {
    const rawSeconds = timeMilliseconds / 1000;
    this.pointer.update(rawSeconds);
    const seconds = (this.reducedMotion ? STILL_FRAME_MILLISECONDS * 1.4 : timeMilliseconds) / 1000;
    const cycle = seconds % CYCLE_SECONDS;
    const total = this.grid.columns * this.grid.rows;
    const sweep = Math.min(1, cycle / SWEEP_SECONDS);
    const progress = sweep * (total + this.grid.concurrency);
    // After the sweep the finished grid holds, then fades before the next launch.
    const fade = cycle < CYCLE_SECONDS - 1 ? 1 : Math.max(0, CYCLE_SECONDS - cycle);

    let pointerExtra = 0;
    if (this.interaction === 'dispatch') {
      pointerExtra = this.updateDispatch(rawSeconds);
    } else if (this.interaction === 'trail') {
      this.updateTrail(rawSeconds);
      pointerExtra = rawSeconds;
    }

    const look = this.look;
    look.set([this.grid.columns, this.grid.rows, this.grid.laneSide, this.grid.invocationCount], 0);
    look.set([seconds, progress, this.grid.concurrency, fade], 4);
    look.set([this.width, this.height, INTERACTION_MODES[this.interaction], this.hoveredIndex], 8);
    look.set(this.colors.background, 12);
    look.set(this.colors.accent, 16);
    look.set(this.colors.signal, 20);
    look.set(this.highlight, 24);
    look.set(
      [
        (this.pointer.x / this.width) * this.grid.columns,
        (this.pointer.y / this.height) * this.grid.rows,
        this.pointer.presence,
        pointerExtra
      ],
      28
    );
    this.lookBuffer.write(look);
    const renderPass = this.device.beginRenderPass({clearColor: this.colors.background});
    this.model.draw(renderPass);
    renderPass.end();
    this.device.submit();
  }

  /** Tracks the hovered workgroup and returns how far its lanes have filled (0..1). */
  private updateDispatch(rawSeconds: number): number {
    const {columns, rows} = this.grid;
    const isActive = this.pointer.isActive;
    if (this.pointer.target) {
      const column = Math.floor((this.pointer.target[0] / this.width) * columns);
      const row = Math.floor((this.pointer.target[1] / this.height) * rows);
      if (column >= 0 && column < columns && row >= 0 && row < rows) {
        const index = row * columns + column;
        if (index !== this.hoveredIndex || !this.wasActive) this.launchSeconds = rawSeconds;
        this.hoveredIndex = index;
      }
    }
    this.wasActive = isActive;
    if (this.reducedMotion) return 1;
    const phase = (rawSeconds - this.launchSeconds) % LAUNCH_PERIOD_SECONDS;
    return Math.min(1, phase / LAUNCH_FILL_SECONDS);
  }

  /** Marks the workgroups the eased pointer crossed this frame with the time they were redrawn. */
  private updateTrail(rawSeconds: number): void {
    const {columns, rows} = this.grid;
    let isChanged = false;
    const touchCell = (cellX: number, cellY: number): void => {
      const column = Math.floor(cellX);
      const row = Math.floor(cellY);
      if (column < 0 || column >= columns || row < 0 || row >= rows) return;
      this.lastTouched[row * columns + column] = rawSeconds;
      isChanged = true;
      this.hasTouches = true;
    };

    if (this.reducedMotion) {
      // No decay without animation: only the cell under the pointer is marked.
      if (this.hasTouches) {
        this.lastTouched.fill(UNTOUCHED_SECONDS);
        this.hasTouches = false;
        isChanged = true;
      }
      if (this.pointer.isActive) {
        touchCell((this.pointer.x / this.width) * columns, (this.pointer.y / this.height) * rows);
      }
    } else if (this.pointer.isActive) {
      const currentX = (this.pointer.x / this.width) * columns;
      const currentY = (this.pointer.y / this.height) * rows;
      const [startX, startY] = this.previousCell ?? [currentX, currentY];
      const stepCount = Math.max(
        1,
        Math.ceil(Math.hypot(currentX - startX, currentY - startY) / TRAIL_STEP_CELLS)
      );
      for (let step = 0; step <= stepCount; step++) {
        const fraction = step / stepCount;
        touchCell(startX + (currentX - startX) * fraction, startY + (currentY - startY) * fraction);
      }
      this.previousCell = [currentX, currentY];
    } else {
      this.previousCell = null;
    }

    if (this.reducedMotion) this.previousCell = null;
    if (isChanged) this.touchedBuffer.write(this.lastTouched);
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  destroy(): void {
    this.model.destroy();
    this.lookBuffer.destroy();
    this.touchedBuffer.destroy();
  }
}

export default createWorkgroupScene(DEFAULT_WORKGROUP_GRID);
