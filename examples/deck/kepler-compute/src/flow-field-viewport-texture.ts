// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Buffer,
  Texture,
  type Device,
  type Framebuffer,
  type RenderPipeline,
  type VertexArray,
  type ShaderLayout
} from '@luma.gl/core';
import type {FlowFieldBounds} from '@luma-contributors/analysis/raster-analysis';

const SHADER_LAYOUT: ShaderLayout = {
  attributes: [],
  bindings: [
    {name: 'parameters', type: 'uniform', group: 0, location: 0},
    {name: 'sourceField', type: 'texture', group: 0, location: 1, sampleType: 'unfilterable-float'}
  ]
};

const VERTEX_GLSL = `#version 300 es
void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`;
const FRAGMENT_GLSL = `#version 300 es
precision highp float;
precision highp int;
layout(std140) uniform parameters { vec4 region; vec4 dimensions; };
uniform highp sampler2D sourceField;
out vec4 color;
void main() {
  vec2 normalized = (gl_FragCoord.xy - 0.5) / (dimensions.xy - 1.0);
  vec2 coordinate = mix(region.xy, region.zw, normalized) * (dimensions.xy - 1.0);
  ivec2 lower = clamp(ivec2(floor(coordinate)), ivec2(0), ivec2(dimensions.xy) - 2);
  ivec2 upper = lower + 1;
  vec2 fraction = coordinate - vec2(lower);
  vec4 values[4];
  values[0] = texelFetch(sourceField, lower, 0);
  values[1] = texelFetch(sourceField, ivec2(upper.x, lower.y), 0);
  values[2] = texelFetch(sourceField, ivec2(lower.x, upper.y), 0);
  values[3] = texelFetch(sourceField, upper, 0);
  vec4 weights = vec4((1.0-fraction.x)*(1.0-fraction.y), fraction.x*(1.0-fraction.y),
    (1.0-fraction.x)*fraction.y, fraction.x*fraction.y);
  vec4 total = vec4(0.0);
  bool valid = true;
  for (int index = 0; index < 4; index++) {
    valid = valid && values[index].z >= 0.5;
    total += values[index] * weights[index];
  }
  color = valid ? vec4(total.xy, 1.0, total.w) : vec4(0.0);
}`;
const SHADER_WGSL = `
struct Parameters { region: vec4f, dimensions: vec4f };
@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var sourceField: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(corner * 2.0 - vec2f(1.0), 0.0, 1.0);
}
@fragment fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let normalized = (position.xy - vec2f(0.5)) / (parameters.dimensions.xy - vec2f(1.0));
  let coordinate = mix(parameters.region.xy, parameters.region.zw, normalized) * (parameters.dimensions.xy - vec2f(1.0));
  let lower = clamp(vec2i(floor(coordinate)), vec2i(0), vec2i(parameters.dimensions.xy) - vec2i(2));
  let upper = lower + vec2i(1);
  let fraction = coordinate - vec2f(lower);
  let values = array<vec4f, 4>(textureLoad(sourceField, lower, 0),
    textureLoad(sourceField, vec2i(upper.x, lower.y), 0),
    textureLoad(sourceField, vec2i(lower.x, upper.y), 0), textureLoad(sourceField, upper, 0));
  let weights = vec4f((1.0-fraction.x)*(1.0-fraction.y), fraction.x*(1.0-fraction.y),
    (1.0-fraction.x)*fraction.y, fraction.x*fraction.y);
  var total = vec4f(0.0);
  var valid = true;
  for (var index = 0u; index < 4u; index++) {
    valid = valid && values[index].z >= 0.5;
    total += values[index] * weights[index];
  }
  if (valid) { return vec4f(total.xy, 1.0, total.w); }
  return vec4f(0.0);
}`;

/**
 * Rendering adapter for endpoint-spaced FlowParticleField textures. Crop, with validity-aware
 * bilinear resampling, into a caller-selected geographic rectangle. Row zero stays south on both
 * backends: fragment coordinates index texture rows directly, without a presentation Y flip.
 * The source field (and its GPU filter/grid pipeline) stays untouched. No readback or CPU grid
 * scan is needed, including on WebGL. All four source corners must be valid, preserving holes.
 * These shaders only resample fields; luma owns advection.
 */
export class FlowFieldViewportTexture {
  readonly texture: Texture;
  readonly byteLength: number;
  private readonly framebuffer: Framebuffer;
  private readonly parameters: Buffer;
  private readonly pipeline: RenderPipeline;
  private readonly vertexArray: VertexArray;
  private readonly resources: {destroy(): void}[] = [];

  constructor(
    readonly device: Device,
    columns: number,
    rows: number
  ) {
    try {
      this.texture = this.ownResource(
        device.createTexture({
          id: 'flow-field-viewport',
          width: columns,
          height: rows,
          format: 'rgba32float',
          samples: 1,
          usage: Texture.RENDER | Texture.SAMPLE,
          sampler: {minFilter: 'nearest', magFilter: 'nearest'}
        })
      );
      this.byteLength = columns * rows * 16 + 32;
      this.framebuffer = this.ownResource(
        device.createFramebuffer({
          id: 'flow-field-viewport-crop-framebuffer',
          width: columns,
          height: rows,
          colorAttachments: [this.texture],
          depthStencilAttachment: null
        })
      );
      this.parameters = this.ownResource(
        device.createBuffer({
          usage: Buffer.UNIFORM | Buffer.COPY_DST,
          byteLength: 32
        })
      );
      const isWebGPU = device.type === 'webgpu';
      const vertex = this.ownResource(
        device.createShader({
          stage: 'vertex',
          source: isWebGPU ? SHADER_WGSL : VERTEX_GLSL
        })
      );
      const fragment = this.ownResource(
        device.createShader({
          stage: 'fragment',
          source: isWebGPU ? SHADER_WGSL : FRAGMENT_GLSL
        })
      );
      this.pipeline = this.ownResource(
        device.createRenderPipeline({
          id: 'flow-field-viewport-crop-pipeline',
          vs: vertex,
          fs: fragment,
          ...(isWebGPU ? {vertexEntryPoint: 'vertexMain', fragmentEntryPoint: 'fragmentMain'} : {}),
          shaderLayout: SHADER_LAYOUT,
          topology: 'triangle-list',
          colorAttachmentFormats: ['rgba32float'],
          // This framebuffer has no depth attachment. Even depthWriteEnabled:false/depthCompare:always
          // makes published luma infer a depth24plus pipeline, incompatible with this color-only pass.
          // Alpha stores speed/altitude metadata, not opacity. Never inherit Deck's blending.
          parameters: {cullMode: 'none', sampleCount: 1, blend: false}
        })
      );
      this.vertexArray = this.ownResource(
        device.createVertexArray({shaderLayout: SHADER_LAYOUT, bufferLayout: []})
      );
    } catch (error) {
      try {
        this.destroy();
      } catch (cleanupError) {
        throw new AggregateError([error, cleanupError], 'Flow field viewport construction failed');
      }
      throw error;
    }
  }

  update(source: Texture, region: FlowFieldBounds): boolean {
    if (this.pipeline.isErrored) throw new Error('Flow field viewport crop pipeline failed');
    if (this.pipeline.isPending) return false;
    this.parameters.write(new Float32Array([...region, source.width, source.height, 0, 0]));
    const encoder = this.device.createCommandEncoder({id: 'flow-field-viewport-crop-encoder'});
    const pass = encoder.beginRenderPass({
      id: 'flow-field-viewport-crop-pass',
      framebuffer: this.framebuffer,
      parameters: {viewport: [0, 0, this.texture.width, this.texture.height]},
      clearColor: [0, 0, 0, 0],
      clearDepth: false
    });
    pass.setPipeline(this.pipeline);
    pass.setBindings({parameters: this.parameters, sourceField: source});
    pass.setVertexArray(this.vertexArray);
    const drawn = pass.draw({vertexCount: 3});
    pass.end();
    this.device.submit(encoder.finish());
    return drawn;
  }

  private ownResource<ResourceType extends {destroy(): void}>(
    resource: ResourceType
  ): ResourceType {
    this.resources.push(resource);
    return resource;
  }

  destroy(): void {
    const errors: unknown[] = [];
    for (const resource of this.resources.splice(0).reverse()) {
      try {
        resource.destroy();
      } catch (error) {
        errors.push(error);
      }
    }
    if (errors.length) throw new AggregateError(errors, 'Flow field viewport cleanup failed');
  }
}
