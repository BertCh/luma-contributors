// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Live luma.gl renderer for preview data. Imported lazily (see index.ts), never during prerendering.

import {Buffer, type Device, type RenderPipelineParameters, type Texture} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {PreviewKind} from '../../catalog/schema.ts';
import type {GpuScene, GpuSceneContext, GpuSceneInstance} from '../types.ts';
import {buildPreview} from './builders.ts';
import {clamp} from './data.ts';
import {getRampColor, readPalette, type PreviewPalette, type Rgb} from './palette.ts';
import {
  FIELD_FRAGMENT_GLSL,
  FIELD_VERTEX_GLSL,
  FIELD_WGSL,
  MESH_FRAGMENT_GLSL,
  MESH_VERTEX_GLSL,
  MESH_WGSL,
  SPRITE_FRAGMENT_GLSL,
  SPRITE_VERTEX_GLSL,
  SPRITE_WGSL
} from './shaders.ts';
import {
  getDefaultParameterValues,
  POINTER_KINDS,
  type PreviewParameterValues
} from './parameters.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type FieldLayer,
  type FillLayer,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewLayer,
  type SpriteLayer,
  type StrokeLayer
} from './types.ts';

const ADDITIVE_BLEND: RenderPipelineParameters = {
  blend: true,
  blendColorOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one',
  blendAlphaOperation: 'add',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one'
};

const ALPHA_BLEND: RenderPipelineParameters = {
  ...ADDITIVE_BLEND,
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaDstFactor: 'one-minus-src-alpha'
};

const FIELD_STYLES = {ramp: 0, contours: 1, hillshade: 2, expand: 3} as const;
const STROKE_MODES = {static: 0, flow: 1, draw: 2, pulse: 3} as const;
const RAMP_WIDTH = 64;

type LayerRenderer = {
  /** Writes uniforms for the current frame. */
  update(timeSeconds: number, still: boolean, viewX: number, viewY: number): void;
  model: Model;
  destroy(): void;
};

/** Triangulates a simple polygon by ear clipping; falls back to a fan for degenerate rings. */
function triangulateRing(ring: PolygonItem['ring']): number[] {
  const count = ring.length;
  let area = 0;
  for (let index = 0; index < count; index++) {
    const next = ring[(index + 1) % count];
    area += ring[index][0] * next[1] - next[0] * ring[index][1];
  }
  const orientation = area >= 0 ? 1 : -1;
  const cross = (a: number, b: number, c: number) =>
    (ring[b][0] - ring[a][0]) * (ring[c][1] - ring[a][1]) -
    (ring[b][1] - ring[a][1]) * (ring[c][0] - ring[a][0]);
  const remaining = ring.map((_, index) => index);
  const triangles: number[] = [];
  let guard = count * count;
  while (remaining.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let position = 0; position < remaining.length; position++) {
      const previous = remaining[(position + remaining.length - 1) % remaining.length];
      const current = remaining[position];
      const next = remaining[(position + 1) % remaining.length];
      if (cross(previous, current, next) * orientation <= 1e-9) continue;
      const containsOther = remaining.some(other => {
        if (other === previous || other === current || other === next) return false;
        return (
          cross(previous, current, other) * orientation >= 0 &&
          cross(current, next, other) * orientation >= 0 &&
          cross(next, previous, other) * orientation >= 0
        );
      });
      if (containsOther) continue;
      triangles.push(previous, current, next);
      remaining.splice(position, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;
  }
  if (remaining.length === 3) triangles.push(...remaining);
  else
    for (let index = 1; index + 1 < remaining.length; index++)
      triangles.push(remaining[0], remaining[index], remaining[index + 1]);
  return triangles;
}

function createFillVertices(items: PolygonItem[], palette: PreviewPalette): Float32Array {
  const values: number[] = [];
  for (const item of items) {
    const color = palette.roles[item.role];
    for (const index of triangulateRing(item.ring)) {
      values.push(
        item.ring[index][0],
        item.ring[index][1],
        color[0],
        color[1],
        color[2],
        item.alpha ?? 0.5,
        0,
        item.phase ?? 0,
        0
      );
    }
  }
  return new Float32Array(values);
}

/** Mitered ribbon triangles: position, color, (along, phase, across). */
function createStrokeVertices(items: PathItem[], palette: PreviewPalette): Float32Array {
  const values: number[] = [];
  for (const item of items) {
    const points = item.points.filter(
      (point, index, all) =>
        index === 0 || point[0] !== all[index - 1][0] || point[1] !== all[index - 1][1]
    );
    if (points.length < 2) continue;
    const color = palette.roles[item.role];
    const halfWidth = Math.max(item.width * 0.5, 0.18) * 1.25;
    const lengths = [0];
    for (let index = 1; index < points.length; index++) {
      lengths.push(
        lengths[index - 1] +
          Math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1]
          )
      );
    }
    const total = lengths[lengths.length - 1] || 1;
    const sides: number[][] = [];
    for (let index = 0; index < points.length; index++) {
      const before = points[Math.max(index - 1, 0)];
      const after = points[Math.min(index + 1, points.length - 1)];
      const current = points[index];
      const directionIn =
        index > 0
          ? [current[0] - before[0], current[1] - before[1]]
          : [after[0] - current[0], after[1] - current[1]];
      const directionOut =
        index < points.length - 1 ? [after[0] - current[0], after[1] - current[1]] : directionIn;
      const lengthIn = Math.hypot(directionIn[0], directionIn[1]) || 1;
      const lengthOut = Math.hypot(directionOut[0], directionOut[1]) || 1;
      const normalIn = [-directionIn[1] / lengthIn, directionIn[0] / lengthIn];
      const normalOut = [-directionOut[1] / lengthOut, directionOut[0] / lengthOut];
      let miterX = normalIn[0] + normalOut[0];
      let miterY = normalIn[1] + normalOut[1];
      const miterLength = Math.hypot(miterX, miterY) || 1;
      miterX /= miterLength;
      miterY /= miterLength;
      const scale = halfWidth / Math.max(miterX * normalIn[0] + miterY * normalIn[1], 0.5);
      sides.push([
        current[0] + miterX * scale,
        current[1] + miterY * scale,
        current[0] - miterX * scale,
        current[1] - miterY * scale,
        lengths[index] / total
      ]);
    }
    const pushVertex = (x: number, y: number, along: number, across: number) =>
      values.push(
        x,
        y,
        color[0],
        color[1],
        color[2],
        item.alpha ?? 0.9,
        along,
        item.phase ?? 0,
        across
      );
    for (let index = 0; index + 1 < sides.length; index++) {
      const a = sides[index];
      const b = sides[index + 1];
      pushVertex(a[0], a[1], a[4], 1);
      pushVertex(a[2], a[3], a[4], -1);
      pushVertex(b[0], b[1], b[4], 1);
      pushVertex(b[0], b[1], b[4], 1);
      pushVertex(a[2], a[3], a[4], -1);
      pushVertex(b[2], b[3], b[4], -1);
    }
  }
  return new Float32Array(values);
}

function splitMeshVertices(vertices: Float32Array) {
  const count = vertices.length / 9;
  const position = new Float32Array(count * 2);
  const color = new Float32Array(count * 4);
  const track = new Float32Array(count * 3);
  for (let index = 0; index < count; index++) {
    position.set(vertices.subarray(index * 9, index * 9 + 2), index * 2);
    color.set(vertices.subarray(index * 9 + 2, index * 9 + 6), index * 4);
    track.set(vertices.subarray(index * 9 + 6, index * 9 + 9), index * 3);
  }
  return {position, color, track, count};
}

class PreviewRenderer {
  private readonly renderers: LayerRenderer[] = [];
  private readonly buffers: Buffer[] = [];
  private readonly textures: Texture[] = [];
  private readonly cornerBuffer: Buffer;
  private readonly uniformData = new Float32Array(16);

  constructor(
    private readonly device: Device,
    layers: PreviewLayer[],
    private readonly palette: PreviewPalette
  ) {
    this.cornerBuffer = this.createVertexBuffer(new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
    for (const layer of layers) {
      if (layer.type === 'field') this.renderers.push(this.createFieldRenderer(layer));
      else if (layer.type === 'sprites') this.renderers.push(this.createSpriteRenderer(layer));
      else {
        const renderer = this.createMeshRenderer(layer);
        if (renderer) this.renderers.push(renderer);
      }
    }
  }

  private createVertexBuffer(data: Float32Array): Buffer {
    const buffer = this.device.createBuffer({usage: Buffer.VERTEX | Buffer.COPY_DST, data});
    this.buffers.push(buffer);
    return buffer;
  }

  private createUniformBuffer(): Buffer {
    const buffer = this.device.createBuffer({
      usage: Buffer.UNIFORM | Buffer.COPY_DST,
      data: new Float32Array(16)
    });
    this.buffers.push(buffer);
    return buffer;
  }

  private createModel(
    id: string,
    shaders: {wgsl: string; vertex: string; fragment: string},
    props: {
      bufferLayout: {
        name: string;
        format: 'float32x2' | 'float32x3' | 'float32x4';
        stepMode?: 'instance';
      }[];
      attributes: Record<string, Buffer>;
      topology: 'triangle-list' | 'triangle-strip';
      vertexCount: number;
      instanceCount?: number;
      bindings: Record<string, Buffer | Texture>;
      parameters: RenderPipelineParameters;
    }
  ): Model {
    const common = {
      id,
      ...props,
      isInstanced: props.instanceCount !== undefined,
      disableWarnings: true
    };
    return this.device.type === 'webgpu'
      ? new Model(this.device, {...common, source: shaders.wgsl})
      : new Model(this.device, {...common, vs: shaders.vertex, fs: shaders.fragment});
  }

  private createLayerRenderer(
    model: Model,
    uniformBuffer: Buffer,
    getAnimation: () => [number, number, number, number],
    extraResources: {destroy(): void}[] = [],
    getExtra: () => [number, number] = () => [0, 0]
  ): LayerRenderer {
    return {
      model,
      update: (timeSeconds, still, viewX, viewY) => {
        const data = this.uniformData;
        data.set([
          still ? 0 : timeSeconds,
          0,
          still ? 1 : 0,
          0,
          viewX,
          viewY,
          0,
          0,
          ...getAnimation(),
          ...getExtra(),
          0,
          0
        ]);
        uniformBuffer.write(data);
      },
      destroy: () => {
        model.destroy();
        for (const resource of extraResources) resource.destroy();
      }
    };
  }

  private createSpriteRenderer(layer: SpriteLayer): LayerRenderer {
    const count = layer.items.length;
    const center = new Float32Array(count * 3);
    const color = new Float32Array(count * 4);
    const shape = new Float32Array(count * 2);
    layer.items.forEach((item, index) => {
      const rgb = this.palette.roles[item.role];
      center.set([item.x, item.y, item.z ?? 0], index * 3);
      color.set([rgb[0], rgb[1], rgb[2], item.alpha ?? 0.9], index * 4);
      shape.set([item.radius, item.phase ?? 0], index * 2);
    });
    const uniformBuffer = this.createUniformBuffer();
    const model = this.createModel(
      'preview-sprites',
      {wgsl: SPRITE_WGSL, vertex: SPRITE_VERTEX_GLSL, fragment: SPRITE_FRAGMENT_GLSL},
      {
        bufferLayout: [
          {name: 'corner', format: 'float32x2'},
          {name: 'center', format: 'float32x3', stepMode: 'instance'},
          {name: 'color', format: 'float32x4', stepMode: 'instance'},
          {name: 'shape', format: 'float32x2', stepMode: 'instance'}
        ],
        attributes: {
          corner: this.cornerBuffer,
          center: this.createVertexBuffer(center),
          color: this.createVertexBuffer(color),
          shape: this.createVertexBuffer(shape)
        },
        topology: 'triangle-strip',
        vertexCount: 4,
        instanceCount: count,
        bindings: {sceneUniforms: uniformBuffer},
        parameters: ADDITIVE_BLEND
      }
    );
    const turntable = layer.turntable;
    return this.createLayerRenderer(model, uniformBuffer, () => [
      layer.pulse ?? 0,
      turntable?.speed ?? 0,
      turntable?.angle ?? 0,
      turntable?.perspective ?? 0
    ]);
  }

  private createMeshRenderer(layer: StrokeLayer | FillLayer): LayerRenderer | null {
    const vertices =
      layer.type === 'strokes'
        ? createStrokeVertices(layer.items, this.palette)
        : createFillVertices(layer.items, this.palette);
    const mesh = splitMeshVertices(vertices);
    if (mesh.count === 0) return null;
    const uniformBuffer = this.createUniformBuffer();
    const model = this.createModel(
      `preview-${layer.type}`,
      {wgsl: MESH_WGSL, vertex: MESH_VERTEX_GLSL, fragment: MESH_FRAGMENT_GLSL},
      {
        bufferLayout: [
          {name: 'position', format: 'float32x2'},
          {name: 'color', format: 'float32x4'},
          {name: 'track', format: 'float32x3'}
        ],
        attributes: {
          position: this.createVertexBuffer(mesh.position),
          color: this.createVertexBuffer(mesh.color),
          track: this.createVertexBuffer(mesh.track)
        },
        topology: 'triangle-list',
        vertexCount: mesh.count,
        bindings: {sceneUniforms: uniformBuffer},
        parameters: layer.blend === 'alpha' ? ALPHA_BLEND : ADDITIVE_BLEND
      }
    );
    const mode = STROKE_MODES[layer.animation ?? 'static'];
    const speed = layer.type === 'strokes' ? (layer.speed ?? 0.3) : 0.3;
    const trail = layer.type === 'strokes' ? (layer.trail ?? 1) : 1;
    return this.createLayerRenderer(model, uniformBuffer, () => [mode, speed, trail, 0]);
  }

  private createFieldRenderer(layer: FieldLayer): LayerRenderer {
    const {columns, rows, values} = layer;
    const pixels = new Uint8Array(columns * rows * 4);
    const strength = layer.strength ?? 0.05;
    const at = (column: number, row: number) =>
      values[clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1)];
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const offset = (row * columns + column) * 4;
        const value = at(column, row);
        pixels[offset] = Math.round(value * 255);
        if (layer.style === 'hillshade') {
          const reliefStrength = layer.strength ?? 5;
          pixels[offset + 1] = Math.round(
            (clamp((at(column + 1, row) - at(column - 1, row)) * reliefStrength, -0.5, 0.5) + 0.5) *
              255
          );
          pixels[offset + 2] = Math.round(
            (clamp((at(column, row + 1) - at(column, row - 1)) * reliefStrength, -0.5, 0.5) + 0.5) *
              255
          );
        } else {
          pixels[offset + 1] = Math.round((value * 2.5 - Math.floor(value * 2.5)) * 255);
        }
        pixels[offset + 3] = 255;
      }
    }
    const rampPixels = new Uint8Array(RAMP_WIDTH * 4);
    for (let index = 0; index < RAMP_WIDTH; index++) {
      const color: Rgb = getRampColor(layer.ramp, index / (RAMP_WIDTH - 1));
      rampPixels.set([color[0] * 255, color[1] * 255, color[2] * 255, 255], index * 4);
    }
    const filter = layer.filter === 'nearest' ? 'nearest' : 'linear';
    const fieldTexture = this.device.createTexture({
      width: columns,
      height: rows,
      format: 'rgba8unorm',
      data: pixels,
      sampler: {
        minFilter: filter,
        magFilter: filter,
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    const rampTexture = this.device.createTexture({
      width: RAMP_WIDTH,
      height: 1,
      format: 'rgba8unorm',
      data: rampPixels,
      sampler: {
        minFilter: 'linear',
        magFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    this.textures.push(fieldTexture, rampTexture);
    const uniformBuffer = this.createUniformBuffer();
    const model = this.createModel(
      'preview-field',
      {wgsl: FIELD_WGSL, vertex: FIELD_VERTEX_GLSL, fragment: FIELD_FRAGMENT_GLSL},
      {
        bufferLayout: [{name: 'corner', format: 'float32x2'}],
        attributes: {corner: this.cornerBuffer},
        topology: 'triangle-strip',
        vertexCount: 4,
        bindings: {sceneUniforms: uniformBuffer, fieldTexture, rampTexture},
        parameters: {}
      }
    );
    const style = FIELD_STYLES[layer.style];
    return this.createLayerRenderer(
      model,
      uniformBuffer,
      () => [style, layer.levels ?? 1, layer.speed ?? 0.1, layer.style === 'ramp' ? strength : 0],
      [fieldTexture, rampTexture],
      () =>
        layer.style === 'hillshade' && layer.light !== undefined
          ? [layer.light, 1]
          : layer.style === 'expand' && layer.cutoff !== undefined
            ? [layer.cutoff, 1]
            : [0, 0]
    );
  }

  render(timeSeconds: number, still: boolean, width: number, height: number): void {
    const scale = Math.max(width / SCENE_WIDTH, height / SCENE_HEIGHT);
    const viewX = (scale / width) * 2;
    const viewY = (scale / height) * 2;
    for (const renderer of this.renderers) renderer.update(timeSeconds, still, viewX, viewY);
    const background = this.palette.background;
    const pass = this.device.beginRenderPass({
      clearColor: [background[0], background[1], background[2], 1]
    });
    for (const renderer of this.renderers) renderer.model.draw(pass);
    pass.end();
    this.device.submit();
  }

  destroy(): void {
    for (const renderer of this.renderers) renderer.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}

/** Scene units per canvas pixel is 1 / scale; scenes cover the canvas, so the larger ratio wins. */
function getSceneScale(width: number, height: number): number {
  return Math.max(width / SCENE_WIDTH, height / SCENE_HEIGHT);
}

/** Pointer positions snap to this many scene units, so a rebuild is not triggered by every pixel. */
const POINTER_QUANTUM = 2;

/** Kinds whose pointer choice (a source or focus node) stays after the pointer leaves. */
const STICKY_POINTER_KINDS: ReadonlySet<PreviewKind> = new Set<PreviewKind>([
  'network',
  'shortest-path',
  'isochrones',
  'spatial-weights'
]);

/** Creates the luma.gl scene for one preview kind. */
export function createPreviewScene(kind: PreviewKind): GpuScene {
  return (context: GpuSceneContext): GpuSceneInstance => {
    const canvasContext = context.device.getDefaultCanvasContext();
    const palette = readPalette();
    const usesPointer = POINTER_KINDS.has(kind);
    const parameters: PreviewParameterValues = {
      ...getDefaultParameterValues(kind),
      ...context.parameters
    };
    let pointer: Point | null = null;
    let pointerKey = '';
    let isDirty = false;
    let data = buildPreview(kind, context.seed, parameters, null);
    let renderer = new PreviewRenderer(context.device, data.layers, palette);
    // Point clouds turn on the CPU: spin integrates every frame, the pointer adds an offset.
    let turnAngle = 0;
    let turnOffset = 0;
    let lastSeconds = -1;

    const rebuild = () => {
      isDirty = false;
      const next = buildPreview(kind, context.seed, parameters, pointer);
      renderer.destroy();
      data = next;
      renderer = new PreviewRenderer(context.device, data.layers, palette);
    };

    return {
      render(timeMilliseconds) {
        const [width, height] = canvasContext.getDrawingBufferSize();
        if (width === 0 || height === 0) return;
        const seconds = timeMilliseconds / 1000;
        if (kind === 'point-cloud') {
          const elapsed = lastSeconds < 0 || context.reducedMotion ? 0 : seconds - lastSeconds;
          turnAngle += ((parameters.spin * Math.PI) / 180) * elapsed;
          lastSeconds = seconds;
          // Parameters change only the perspective here, so no rebuild is needed for spin.
          for (const layer of data.layers) {
            if (layer.type === 'sprites' && layer.turntable) {
              layer.turntable.angle = 0.5 + turnAngle + turnOffset;
              layer.turntable.perspective = parameters.depth / 1000;
            }
          }
        }
        if (isDirty) rebuild();
        renderer.render(seconds, context.reducedMotion, width, height);
      },
      setPointer(position) {
        if (!usesPointer) return;
        const [width, height] = canvasContext.getDrawingBufferSize();
        if (kind === 'point-cloud') {
          turnOffset = position && width > 0 ? (position[0] / width - 0.5) * Math.PI : 0;
          return;
        }
        let key = '';
        let next: Point | null = null;
        if (position && width > 0 && height > 0) {
          const scale = getSceneScale(width, height);
          next = [
            Math.round((SCENE_WIDTH / 2 + (position[0] - width / 2) / scale) / POINTER_QUANTUM) *
              POINTER_QUANTUM,
            Math.round((SCENE_HEIGHT / 2 + (position[1] - height / 2) / scale) / POINTER_QUANTUM) *
              POINTER_QUANTUM
          ];
          key = next.join(',');
        } else if (STICKY_POINTER_KINDS.has(kind)) {
          return;
        }
        if (key === pointerKey) return;
        pointerKey = key;
        pointer = next;
        isDirty = true;
      },
      setParameters(values) {
        Object.assign(parameters, values);
        // Point clouds read spin and perspective every frame; nothing needs rebuilding.
        if (kind !== 'point-cloud') isDirty = true;
      },
      destroy() {
        renderer.destroy();
      }
    };
  };
}
