// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPU side of picture text: one instanced quad per glyph, swatch or rule, drawn from a signed
// distance field atlas held in a storage buffer. Glyphs get a halo in the background color, so
// labels read over data in both themes, and stay sharp at any scale because the outline is
// computed per pixel from the field. Because the scene draws its own text, the poster (a capture
// of the canvas) shows exactly the labels of the live picture.
//
// The layer extends the demo kit's DemoLayer only so the run framework can hold it with its other
// layers. It has its own uniform block (`TextUniforms`) and never reads the kit's `Look`.

import {Buffer, type Device, type RenderPass} from '@luma.gl/core';
import {
  createEmptyStorageBuffer,
  createStorageBuffer,
  DemoLayer,
  type DemoTheme,
  type DemoView,
  type LookOptions
} from '../../demos/kit';
import {getPixelOfWorld} from '../pointer-overlay';
import {getFontPixels, loadFontPixels} from './font';
import {FONT_METRICS} from './font-metrics';
import {
  getTextScale,
  INSTANCE_FLOATS,
  layoutText,
  type RunText,
  type TextLayout,
  type TextPalette
} from './text-layout';

/** Halo width around glyphs and rules, in text pixels. */
const HALO_WIDTH = 1.4;
/** Halo opacity: how much of the picture under a label is covered by the background color. */
const HALO_STRENGTH = 0.82;
/** Text opacity of context (input) text. */
const CONTEXT_ALPHA = 0.72;
/** Floats in TextUniforms. */
const UNIFORM_FLOATS = 12;

const TEXT_SHADER = /* wgsl */ `
struct TextUniforms {
  // Canvas size in pixels.
  canvas: vec2f,
  // Atlas size in texels.
  atlas: vec2f,
  // Distance field reach in texels, halo width in pixels, halo strength, layer opacity.
  spread: f32,
  halo: f32,
  haloStrength: f32,
  alpha: f32,
  background: vec4f
};

struct TextInstance {
  // Pivot in canvas pixels (y down).
  pivot: vec4f,
  // Quad in the label's frame relative to the pivot: x, y (y down), width, height in pixels.
  box: vec4f,
  // Glyph cell in the atlas: x, y, width, height in texels.
  cell: vec4f,
  // Angle (radians, counter-clockwise on screen), kind (0 glyph, 1 rectangle), weight (pixels),
  // pass (1 halo, 2 ink).
  style: vec4f,
  color: vec4f
};

// DemoLayer always binds its look buffer at binding 0. Text never reads it (it has TextUniforms);
// the declaration only keeps the binding in the shader layout.
@group(0) @binding(0) var<uniform> look: vec4f;
@group(0) @binding(1) var<storage, read> instances: array<TextInstance>;
@group(0) @binding(2) var<storage, read> atlas: array<u32>;
@group(0) @binding(3) var<uniform> textUniforms: TextUniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) instance: u32
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
  let item = instances[instanceIndex];
  // Rectangles grow by the halo; glyph cells already hold the field's padding.
  let margin = select(0.0, textUniforms.halo + 1.0, item.style.y > 0.5);
  let local = corners[vertexIndex] * (item.box.zw + 2.0 * margin) - margin;
  let frame = item.box.xy + local;
  let cosine = cos(item.style.x);
  let sine = sin(item.style.x);
  let pixel = item.pivot.xy + vec2f(frame.x * cosine + frame.y * sine, -frame.x * sine + frame.y * cosine);
  var output: VertexOutput;
  output.position = vec4f(pixel.x / textUniforms.canvas.x * 2.0 - 1.0, 1.0 - pixel.y / textUniforms.canvas.y * 2.0, 0.0, 1.0);
  output.local = local;
  output.instance = instanceIndex;
  return output;
}

fn readTexel(x: i32, y: i32) -> f32 {
  let index = u32(y) * u32(textUniforms.atlas.x) + u32(x);
  let word = atlas[index >> 2u];
  return f32((word >> ((index & 3u) * 8u)) & 255u) / 255.0;
}

/** Bilinear sample of the field inside one glyph cell; texel centers are at half texels. */
fn sampleField(cell: vec4f, texel: vec2f) -> f32 {
  let position = clamp(texel - 0.5, vec2f(0.0), cell.zw - 1.0);
  let base = floor(position);
  let fraction = position - base;
  let x0 = i32(cell.x + base.x);
  let y0 = i32(cell.y + base.y);
  let x1 = min(x0 + 1, i32(cell.x + cell.z) - 1);
  let y1 = min(y0 + 1, i32(cell.y + cell.w) - 1);
  let top = mix(readTexel(x0, y0), readTexel(x1, y0), fraction.x);
  let bottom = mix(readTexel(x0, y1), readTexel(x1, y1), fraction.x);
  return mix(top, bottom, fraction.y);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let item = instances[input.instance];
  var distance: f32;
  var halo = textUniforms.halo;
  if (item.style.y < 0.5) {
    let pixelsPerTexel = item.box.z / max(item.cell.z, 1.0);
    var field: f32;
    if (pixelsPerTexel < 0.75) {
      // Small text (device pixel ratio 1): one pixel covers more than a texel. Four taps over the
      // pixel average the field, so thin stems do not alias.
      field = 0.25 * (
        sampleField(item.cell, (input.local + vec2f(-0.25, -0.25)) / pixelsPerTexel) +
        sampleField(item.cell, (input.local + vec2f(0.25, -0.25)) / pixelsPerTexel) +
        sampleField(item.cell, (input.local + vec2f(-0.25, 0.25)) / pixelsPerTexel) +
        sampleField(item.cell, (input.local + vec2f(0.25, 0.25)) / pixelsPerTexel)
      );
    } else {
      field = sampleField(item.cell, input.local / pixelsPerTexel);
    }
    // Signed distance to the outline in screen pixels, positive inside, thickened by the weight.
    distance = (field - 0.5) * 2.0 * textUniforms.spread * pixelsPerTexel + item.style.z;
    // The field reaches only so far: keep the halo inside it.
    halo = min(halo, textUniforms.spread * pixelsPerTexel - 1.0);
  } else {
    let size = item.box.zw;
    let inside = min(min(input.local.x, size.x - input.local.x), min(input.local.y, size.y - input.local.y));
    distance = inside + 0.5 * min(min(size.x, size.y), 1.0) - 0.5;
  }
  // Halo pass (style.w 1): the background color around the outline. Ink pass (2): the glyph.
  if (item.style.w < 1.5) {
    let haloAlpha = clamp(distance + halo + 0.5, 0.0, 1.0) * textUniforms.haloStrength * item.color.a;
    if (haloAlpha < 0.002) {
      discard;
    }
    return vec4f(textUniforms.background.rgb, haloAlpha * textUniforms.alpha);
  }
  let inkAlpha = clamp(distance + 0.5, 0.0, 1.0) * item.color.a;
  if (inkAlpha < 0.002) {
    discard;
  }
  return vec4f(item.color.rgb, inkAlpha * textUniforms.alpha);
}
`;

// One atlas buffer per device, shared by every text layer on it.
const atlasBuffers = new WeakMap<Device, {buffer: Buffer; users: number}>();

function acquireAtlas(device: Device, pixels: Uint32Array): Buffer {
  const shared = atlasBuffers.get(device);
  if (shared) {
    shared.users++;
    return shared.buffer;
  }
  const buffer = createStorageBuffer(device, pixels);
  atlasBuffers.set(device, {buffer, users: 1});
  return buffer;
}

function releaseAtlas(device: Device): void {
  const shared = atlasBuffers.get(device);
  if (!shared) return;
  shared.users--;
  if (shared.users <= 0) {
    shared.buffer.destroy();
    atlasBuffers.delete(device);
  }
}

export type TextLayerOptions = {
  /**
   * CSS width the picture is shown at. Default: the canvas width over the device pixel ratio. A
   * poster rendered for a card passes `CARD_DISPLAY_WIDTH`.
   */
  displayWidth?: number;
  /** Input text (context): drawn fainter, and kept in the inputs-only render. */
  context?: boolean;
  alpha?: number;
};

/** Mixes `color` toward `target` by `amount`. */
function mixColor(
  color: readonly [number, number, number],
  target: readonly [number, number, number, number] | readonly [number, number, number],
  amount: number
): [number, number, number] {
  return [
    color[0] + (target[0] - color[0]) * amount,
    color[1] + (target[1] - color[1]) * amount,
    color[2] + (target[2] - color[2]) * amount
  ];
}

/** The text colors of a theme: ink, a muted ink that keeps about 5:1 contrast, the categories. */
export function getTextPalette(theme: DemoTheme): TextPalette {
  return {
    ink: theme.ink,
    muted: mixColor(theme.ink, theme.background, 0.33),
    background: [theme.background[0], theme.background[1], theme.background[2]],
    categories: theme.categories
  };
}

/**
 * Labels, swatches and rules over a picture. Call `setText`, then `update` and `draw` as usual.
 * The atlas texels load lazily (./font.ts): until they arrive the layer draws nothing and then
 * calls `onReady`, so the owner can draw again. A scene that awaits `loadFontPixels()` before its
 * first frame never shows a frame without its labels.
 */
export class TextLayer extends DemoLayer {
  private instanceBuffer: Buffer;
  private readonly uniformBuffer: Buffer;
  private readonly uniformValues = new Float32Array(UNIFORM_FLOATS);
  private readonly placeholder: Buffer;
  private capacity: number;
  private text: RunText = {};
  private options: TextLayerOptions = {};
  private layoutKey = '';
  private layout: TextLayout | null = null;
  private hasAtlas = false;
  private isDestroyed = false;

  constructor(device: Device, onReady?: () => void) {
    const capacity = 256;
    const instanceBuffer = createEmptyStorageBuffer(device, capacity * INSTANCE_FLOATS);
    const uniformBuffer = device.createBuffer({
      byteLength: UNIFORM_FLOATS * 4,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const loaded = getFontPixels();
    const placeholder = createEmptyStorageBuffer(device, 1);
    super(
      device,
      'run-text',
      TEXT_SHADER,
      {
        instances: instanceBuffer,
        atlas: loaded ? acquireAtlas(device, loaded) : placeholder,
        textUniforms: uniformBuffer
      },
      0
    );
    this.instanceBuffer = instanceBuffer;
    this.uniformBuffer = uniformBuffer;
    this.placeholder = placeholder;
    this.capacity = capacity;
    this.hasAtlas = Boolean(loaded);
    if (!loaded) {
      void loadFontPixels().then(pixels => {
        if (this.isDestroyed) return;
        this.model.setBindings({atlas: acquireAtlas(device, pixels)});
        this.hasAtlas = true;
        onReady?.();
      });
    }
  }

  /** The labels and rules to draw. Cheap to call every frame: layout reruns only on change. */
  setText(text: RunText, options: TextLayerOptions = {}): void {
    this.text = text;
    this.options = options;
  }

  /** What the last layout drew and dropped. */
  getLayout(): TextLayout | null {
    return this.layout;
  }

  /** Lays the text out for `view` and `theme`. The kit's look options do not apply to text. */
  override update(view: DemoView, theme: DemoTheme, _options: LookOptions = {}): void {
    const displayWidth =
      this.options.displayWidth ?? view.width / (globalThis.devicePixelRatio || 1);
    const scale = getTextScale(view.width, displayWidth);
    const alpha = (this.options.alpha ?? 1) * (this.options.context ? CONTEXT_ALPHA : 1);
    this.uniformValues.set(
      [
        view.width,
        view.height,
        FONT_METRICS.width,
        FONT_METRICS.height,
        FONT_METRICS.spread,
        HALO_WIDTH * scale,
        HALO_STRENGTH,
        1,
        ...theme.background
      ],
      0
    );
    this.uniformBuffer.write(this.uniformValues);
    const key = JSON.stringify([
      this.text,
      view.width,
      view.height,
      view.worldWidth,
      view.worldHeight,
      view.origin,
      view.longitudeScale,
      displayWidth,
      alpha,
      theme.ink,
      theme.background,
      theme.categories
    ]);
    if (key === this.layoutKey) return;
    this.layoutKey = key;
    const layout = layoutText(
      this.text,
      {
        width: view.width,
        height: view.height,
        toPixel: (x, y) => getPixelOfWorld(view, x, y)
      },
      scale,
      getTextPalette(theme),
      alpha
    );
    this.layout = layout;
    if (layout.instanceCount > this.capacity) {
      this.instanceBuffer.destroy();
      this.capacity = Math.max(layout.instanceCount, this.capacity * 2);
      this.instanceBuffer = createEmptyStorageBuffer(this.device, this.capacity * INSTANCE_FLOATS);
      this.model.setBindings({instances: this.instanceBuffer});
    }
    if (layout.instanceCount > 0) {
      this.instanceBuffer.write(
        layout.instances.subarray(0, layout.instanceCount * INSTANCE_FLOATS)
      );
    }
    this.model.setInstanceCount(layout.instanceCount);
  }

  override draw(renderPass: RenderPass): void {
    if (!this.hasAtlas || !this.layout || this.layout.instanceCount === 0) return;
    super.draw(renderPass);
  }

  override destroy(): void {
    this.isDestroyed = true;
    super.destroy();
    this.instanceBuffer.destroy();
    this.uniformBuffer.destroy();
    this.placeholder.destroy();
    if (this.hasAtlas) releaseAtlas(this.device);
  }
}
