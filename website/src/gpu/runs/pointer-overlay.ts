// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// What a pointer gesture draws over a picture: its glyphs (segments and dots with a halo) and,
// for probes, a spotlight that dims everything outside the hovered neighbourhood. Shared by
// defineOperationRun (runs) and createPointerDriver (hand-written demos).

import type {Buffer, Device} from '@luma.gl/core';
import {
  createEmptyStorageBuffer,
  createStorageBuffer,
  DemoLayer,
  DotLayer,
  getViewScale,
  getWorldPosition,
  PALETTE_STRUCT,
  SegmentListLayer,
  type DemoTheme,
  type DemoView
} from '../demos/kit';
import {
  createGlyphGeometry,
  GLYPH_DOT_CAPACITY,
  GLYPH_SEGMENT_CAPACITY,
  tessellateGlyphs,
  type RunGlyph,
  type Spotlight
} from './interaction';

/** Glyph sizes are designed for a picture this many device pixels wide and scale with it. */
const GLYPH_REFERENCE_WIDTH = 360;
/** How much of the picture outside the spotlight is covered by the background color. */
const SPOTLIGHT_STRENGTH = 0.58;

export type {Spotlight};

/** World position to canvas pixels (y down); the inverse of getWorldPosition. */
export function getPixelOfWorld(view: DemoView, x: number, y: number): [number, number] {
  const [scaleX, scaleY] = getViewScale(view);
  const ndcX = (((x - (view.origin?.[0] ?? 0)) / view.worldWidth) * 2 - 1) * scaleX;
  const ndcY = (((y - (view.origin?.[1] ?? 0)) / view.worldHeight) * 2 - 1) * scaleY;
  return [((ndcX + 1) / 2) * view.width, ((1 - ndcY) / 2) * view.height];
}

/** Glyph scale, world units per glyph pixel (vertically) and the horizontal stretch of world x. */
export function getGlyphMetrics(view: DemoView): {
  glyphScale: number;
  pixelSize: number;
  aspectX: number;
} {
  const [scaleX, scaleY] = getViewScale(view);
  const glyphScale = view.width / GLYPH_REFERENCE_WIDTH;
  const worldPerPixelY = view.worldHeight / (view.height * scaleY);
  const worldPerPixelX = view.worldWidth / (view.width * scaleX);
  return {
    glyphScale,
    pixelSize: worldPerPixelY * glyphScale,
    aspectX: worldPerPixelX / worldPerPixelY
  };
}

// A full-canvas quad in the background color with a soft-edged hole. The look uniform carries the
// hole: `range` is its center, `elevation` its radius (world units), `hillshade` the horizontal
// stretch of world x and `alpha` the strength.
const SPOTLIGHT_SHADER = /* wgsl */ `
${PALETTE_STRUCT}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) clip: vec2f
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var output: VertexOutput;
  output.position = vec4f(corners[vertexIndex], 0.0, 1.0);
  output.clip = corners[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let world = (input.clip / look.scale + 1.0) * 0.5 * look.world + look.origin;
  let offset = (world - look.range) / vec2f(max(look.hillshade, 1.0e-6), 1.0);
  let amount = smoothstep(look.elevation, look.elevation * 1.8, length(offset));
  return vec4f(look.background.rgb, amount * look.alpha);
}
`;

class SpotlightLayer extends DemoLayer {
  constructor(device: Device) {
    super(device, 'pointer-spotlight', SPOTLIGHT_SHADER, {}, 1);
  }
}

/**
 * The spotlight a probe gets when it does not set one: a circle around its glyphs, never smaller
 * than a sixth of the picture height, so the hovered neighbourhood stays bright and the rest dims.
 */
function getSpotlightOfGlyphs(
  segments: Float32Array,
  segmentCount: number,
  dots: Float32Array,
  dotCount: number,
  aspectX: number,
  minimumRadius: number
): Spotlight | null {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  const include = (x: number, y: number) => {
    if (x < minimumX) minimumX = x;
    if (x > maximumX) maximumX = x;
    if (y < minimumY) minimumY = y;
    if (y > maximumY) maximumY = y;
  };
  for (let index = 0; index < segmentCount; index++) {
    include(segments[index * 4], segments[index * 4 + 1]);
    include(segments[index * 4 + 2], segments[index * 4 + 3]);
  }
  for (let index = 0; index < dotCount; index++) include(dots[index * 2], dots[index * 2 + 1]);
  if (minimumX > maximumX) return null;
  const halfWidth = (maximumX - minimumX) / 2 / aspectX;
  const halfHeight = (maximumY - minimumY) / 2;
  return {
    at: [(minimumX + maximumX) / 2, (minimumY + maximumY) / 2],
    radius: Math.max(minimumRadius, Math.hypot(halfWidth, halfHeight) * 1.15)
  };
}

export class PointerOverlay {
  private readonly device: Device;
  private readonly geometry = createGlyphGeometry();
  private resources: {
    segments: Buffer;
    count: Buffer;
    dots: Buffer;
    values: Buffer;
    halo: SegmentListLayer;
    ink: SegmentListLayer;
    dotLayer: DotLayer;
    spotlight: SpotlightLayer;
  } | null = null;
  private spotlight: Spotlight | null = null;
  /** 0 to 1: the spotlight fades in when the reader starts probing. */
  private spotlightAmount = 0;
  private aspectX = 1;

  constructor(device: Device) {
    this.device = device;
  }

  /**
   * Sets the glyphs to draw. `spotlight` is an explicit circle, `'auto'` to derive one from the
   * glyphs, or null for none.
   */
  set(glyphs: readonly RunGlyph[], view: DemoView, spotlight: Spotlight | 'auto' | null): void {
    if (glyphs.length === 0 && !this.resources && (spotlight === null || spotlight === 'auto')) {
      return;
    }
    const {pixelSize, aspectX} = getGlyphMetrics(view);
    const [minimumX, maximumY] = getWorldPosition(view, [0, 0]);
    const [maximumX, minimumY] = getWorldPosition(view, [view.width, view.height]);
    tessellateGlyphs(
      glyphs,
      this.geometry,
      pixelSize,
      {minimumX, minimumY, maximumX, maximumY},
      aspectX
    );
    this.aspectX = aspectX;
    this.spotlight =
      spotlight === 'auto'
        ? getSpotlightOfGlyphs(
            this.geometry.segments,
            this.geometry.segmentCount,
            this.geometry.dots,
            this.geometry.dotCount,
            aspectX,
            (maximumY - minimumY) / 6
          )
        : spotlight;
    const resources = this.getResources();
    resources.segments.write(this.geometry.segments);
    resources.count.write(Uint32Array.of(this.geometry.segmentCount));
    resources.dots.write(this.geometry.dots);
    resources.dotLayer.setInstanceCount(this.geometry.dotCount);
  }

  /**
   * Moves the spotlight's fade toward on (while the reader probes) or off. Returns true while it
   * is still changing, so the caller keeps drawing.
   */
  fadeSpotlight(isOn: boolean, elapsedMilliseconds: number): boolean {
    const goal = isOn && this.spotlight ? 1 : 0;
    if (this.spotlightAmount === goal) return false;
    const step = elapsedMilliseconds / 160;
    this.spotlightAmount =
      goal > this.spotlightAmount
        ? Math.min(goal, this.spotlightAmount + step)
        : Math.max(goal, this.spotlightAmount - step);
    return true;
  }

  /** Jumps the spotlight's fade to on or off (reduced motion, verification). */
  setSpotlightOn(isOn: boolean): void {
    this.spotlightAmount = isOn && this.spotlight ? 1 : 0;
  }

  /** The layers to draw last, already updated for `view`: spotlight, then glyphs. */
  getLayers(view: DemoView, theme: DemoTheme): DemoLayer[] {
    const resources = this.resources;
    if (!resources) return [];
    const layers: DemoLayer[] = [];
    if (this.spotlight && this.spotlightAmount > 0) {
      resources.spotlight.update(view, theme, {
        range: this.spotlight.at,
        elevation: this.spotlight.radius,
        hillshade: this.aspectX,
        alpha: SPOTLIGHT_STRENGTH * this.spotlightAmount
      });
      layers.push(resources.spotlight);
    }
    if (this.geometry.segmentCount === 0 && this.geometry.dotCount === 0) return layers;
    const {glyphScale} = getGlyphMetrics(view);
    const ink = [theme.ink[0], theme.ink[1], theme.ink[2], 0.96] as const;
    const halo = [theme.background[0], theme.background[1], theme.background[2], 0.7] as const;
    resources.halo.update(view, theme, {mode: -1, solid: halo, radius: 2.1 * glyphScale});
    resources.ink.update(view, theme, {mode: -1, solid: ink, radius: 0.85 * glyphScale});
    resources.dotLayer.update(view, theme, {
      mode: -1,
      solid: ink,
      radius: 2.8 * glyphScale,
      hillshade: 1
    });
    if (this.geometry.segmentCount > 0) layers.push(resources.halo, resources.ink);
    if (this.geometry.dotCount > 0) layers.push(resources.dotLayer);
    return layers;
  }

  destroy(): void {
    const resources = this.resources;
    if (!resources) return;
    resources.halo.destroy();
    resources.ink.destroy();
    resources.dotLayer.destroy();
    resources.spotlight.destroy();
    resources.segments.destroy();
    resources.count.destroy();
    resources.dots.destroy();
    resources.values.destroy();
    this.resources = null;
  }

  private getResources() {
    if (this.resources) return this.resources;
    const segments = createEmptyStorageBuffer(this.device, GLYPH_SEGMENT_CAPACITY * 4);
    const count = createStorageBuffer(this.device, Uint32Array.of(0));
    const dots = createEmptyStorageBuffer(this.device, GLYPH_DOT_CAPACITY * 2);
    const values = createEmptyStorageBuffer(this.device, GLYPH_DOT_CAPACITY);
    this.resources = {
      segments,
      count,
      dots,
      values,
      halo: new SegmentListLayer(this.device, segments, count, GLYPH_SEGMENT_CAPACITY),
      ink: new SegmentListLayer(this.device, segments, count, GLYPH_SEGMENT_CAPACITY),
      dotLayer: new DotLayer(this.device, dots, values, GLYPH_DOT_CAPACITY, 'float32'),
      spotlight: new SpotlightLayer(this.device)
    };
    return this.resources;
  }
}
