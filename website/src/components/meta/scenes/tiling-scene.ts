// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A dispatch mapped onto data: each square is one invocation and the data element it owns, colored
// by workgroup. The workgroup size comes from a shared store, so the grid re-tiles as a slider moves;
// the hovered workgroup stays bright. Lanes past the element count are guarded out (hollow). A pointer
// over a cell also hovers its workgroup, fading the rest back and outlining the workgroup's span.

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
  STILL_FRAME_MILLISECONDS,
  ScenePointer
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

const LOOK_FLOATS = 32;
const SWEEP_SECONDS = 4.5;
const CYCLE_SECONDS = 6.5;
/** Seconds for the pointer hover to fade in and out. */
const FOCUS_IN_SECONDS = 0.12;
const FOCUS_OUT_SECONDS = 0.22;
/** Longest frame gap that easing integrates; longer gaps snap. */
const MAXIMUM_FOCUS_STEP_SECONDS = 0.1;

const WGSL = /* wgsl */ `
struct Look {
  grid: vec4f,
  clock: vec4f,
  view: vec4f,
  view2: vec4f,
  background: vec4f,
  signal: vec4f,
  ink: vec4f,
  pointer: vec4f
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
  let focus = look.pointer.x;
  let outlineStrength = look.pointer.y;
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
  let baseColor = color;
  var alpha = 0.82;
  if (hovered >= 0.0 && focus >= 1.0) {
    alpha = select(0.16, 1.0, abs(hovered - workgroup) < 0.5);
  } else {
    let age = front - workgroup;
    if (age >= 0.0 && age < concurrency) {
      color = mix(color, vec3f(1.0), 0.55);
      alpha = 1.0;
    } else if (age < 0.0) {
      alpha = 0.3;
    }
    // A fading pointer hover blends from the sweep toward the hovered look.
    if (hovered >= 0.0) {
      let hoveredAlpha = select(0.16, 1.0, abs(hovered - workgroup) < 0.5);
      alpha = mix(alpha, hoveredAlpha, focus);
      color = mix(color, baseColor, focus);
    }
  }
  // The first invocation of a workgroup carries a bar on its left edge.
  let isStart = (u32(index) % u32(size)) == 0u;
  let bar = select(0.0, 1.0 - smoothstep(0.1, 0.16, inCell.x), isStart);
  color = mix(color, look.ink.rgb, bar * 0.9);
  // Pointer hover outlines the hovered workgroup's span: a thin line on each cell edge whose
  // neighbour belongs to another workgroup, is guarded, or lies outside the grid.
  var outline = 0.0;
  if (outlineStrength > 0.0 && abs(hovered - workgroup) < 0.5) {
    for (var side = 0; side < 4; side++) {
      let isHorizontal = side < 2;
      let direction = select(-1.0, 1.0, (side & 1) == 1);
      let neighbor = cell + select(vec2f(0.0, direction), vec2f(direction, 0.0), isHorizontal);
      let neighborIndex = neighbor.y * columns + neighbor.x;
      let isOutside = neighbor.x < 0.0 || neighbor.y < 0.0 || neighbor.x >= columns || neighbor.y >= rows;
      let isOther = isOutside || neighborIndex >= count || floor(neighborIndex / size) != workgroup;
      let along = select(inCell.y, inCell.x, isHorizontal);
      let edgeDistance = select(along, 1.0 - along, direction > 0.0);
      let line = 1.0 - smoothstep(0.03, 0.09, edgeDistance);
      outline = max(outline, select(0.0, line, isOther));
    }
  }
  let lineAlpha = outline * outlineStrength * 0.9;
  return vec4f(mix(color, look.ink.rgb, lineAlpha), max(body * alpha, lineAlpha));
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
  private readonly pointer: ScenePointer;
  /** Eased pointer hover amount (0..1) and the workgroup it last hovered. */
  private focus = 0;
  private pointerWorkgroup = -1;
  private lastFocusSeconds = -1;
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
    this.pointer = new ScenePointer(reducedMotion);
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
    const pointerSeconds = timeMilliseconds / 1000;
    this.pointer.update(pointerSeconds);
    this.updateFocus(pointerSeconds, size, cellSize, originX, originY);
    // The store (kernel-demo) wins; otherwise the pointer hover, otherwise the sweep.
    const isStoreHover = this.store.hovered >= 0;
    const hovered = isStoreHover ? this.store.hovered : this.focus > 0 ? this.pointerWorkgroup : -1;
    const focus = isStoreHover ? 1 : this.focus;
    const look = this.look;
    look.set([columns, rows, size, invocationCount], 0);
    look.set([seconds, hovered, front, concurrency], 4);
    look.set([this.width, this.height, cellSize, originX], 8);
    look.set([originY, 0, 0, 0], 12);
    look.set(this.colors.background, 16);
    look.set(this.colors.signal, 20);
    look.set(this.colors.ink, 24);
    look.set([focus, isStoreHover ? 0 : focus, 0, 0], 28);
    this.lookBuffer.write(look);
    const renderPass = this.device.beginRenderPass({clearColor: this.colors.background});
    this.model.draw(renderPass);
    renderPass.end();
    this.device.submit();
  }

  setPointer(position: [number, number] | null): void {
    this.pointer.set(position);
  }

  /** Hit-tests the raw pointer against the grid and eases `focus` toward hovering that workgroup. */
  private updateFocus(
    seconds: number,
    size: number,
    cellSize: number,
    originX: number,
    originY: number
  ): void {
    const {columns, rows, invocationCount} = KERNEL_GRID;
    const target = this.pointer.target;
    let isOverCell = false;
    if (target) {
      const column = Math.floor((target[0] - originX) / cellSize);
      const row = Math.floor((target[1] - originY) / cellSize);
      const index = row * columns + column;
      if (column >= 0 && column < columns && row >= 0 && row < rows && index < invocationCount) {
        isOverCell = true;
        this.pointerWorkgroup = Math.floor(index / size);
      }
    }
    const step = this.lastFocusSeconds < 0 ? 0 : seconds - this.lastFocusSeconds;
    this.lastFocusSeconds = seconds;
    const goal = isOverCell ? 1 : 0;
    if (this.reducedMotion || step <= 0 || step > MAXIMUM_FOCUS_STEP_SECONDS) {
      this.focus = goal;
      return;
    }
    const fade = 1 - Math.exp(-step / (goal > this.focus ? FOCUS_IN_SECONDS : FOCUS_OUT_SECONDS));
    this.focus += (goal - this.focus) * fade;
    if (Math.abs(goal - this.focus) < 0.002) this.focus = goal;
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
