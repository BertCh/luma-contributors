// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {project32, type LayerProps, type Viewport} from '@deck.gl/core';
import {assert, type Device, type Texture} from '@luma.gl/core';
import {TextureTransform, getGPUVectorModelBatches} from '@luma.gl/engine';
import {GPUConstant, type GPUVector} from '@luma.gl/graph/gpu-data';
import type {ShaderModule} from '@luma.gl/shaders';

/** Borrowed VERTEX buffers with identical source chunk boundaries, including empty chunks.
 * Changing values/masks in place requires only a semantic revision update. */
export type ResidentHeatmapInputs = {
  positions: GPUVector<'float32x2' | 'float32x3'>;
  selectionMask: GPUVector<'uint32'>;
  /** Native getWeight results. Kepler field weights must first map weightDomain to [0, 1]. */
  weights?: GPUVector<'float32'>;
  /** Null weights become zero, but still contribute Gaussian count to native MEAN. */
  weightValidity?: GPUVector<'uint32'>;
  revision: number;
};

const weightUniforms: ShaderModule = {
  name: 'weight',
  source: /* wgsl */ `
struct WeightUniforms {
  commonBounds: vec4<f32>,
  radiusPixels: f32,
  textureWidth: f32,
  weightsScale: f32,
};
@group(0) @binding(auto) var<uniform> weight: WeightUniforms;
`,
  uniformTypes: {
    commonBounds: 'vec4<f32>',
    radiusPixels: 'f32',
    textureWidth: 'f32',
    weightsScale: 'f32'
  }
};

/**
 * Native Deck 9.4 WebGPU Gaussian splat, adapted only for borrowed vectors and masks.
 * Kepler's GLSL /8.5 kernel and forced SUM maximum patches do not run in this WebGPU path.
 */
export function getResidentHeatmapWeightShader(positionFormat: 'float32x2' | 'float32x3'): string {
  return /* wgsl */ `
struct WeightAttributes {
  @builtin(vertex_index) vertexIndex: u32,
  @location(0) positions: ${positionFormat === 'float32x3' ? 'vec3<f32>' : 'vec2<f32>'},
  @location(1) weights: f32,
  @location(2) selectionMask: u32,
  @location(3) weightValidity: u32,
};
struct WeightVaryings {
  @builtin(position) position: vec4<f32>,
  @location(0) pointCoordinates: vec2<f32>,
  @location(1) weights: vec4<f32>,
};
const WEIGHT_QUAD_POSITIONS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
fn heatmapFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
@vertex fn vertexMain(attributes: WeightAttributes) -> WeightVaryings {
  var output: WeightVaryings;
  let position = ${positionFormat === 'float32x3' ? 'attributes.positions' : 'vec3<f32>(attributes.positions, 0.0)'};
  if (attributes.selectionMask == 0u || !heatmapFinite(position.x) || !heatmapFinite(position.y) || !heatmapFinite(position.z)) {
    output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let commonPosition = project_position_vec3_f64(position, vec3<f32>(0.0));
  let clipPosition = (commonPosition.xy - weight.commonBounds.xy) /
    (weight.commonBounds.zw - weight.commonBounds.xy) * 2.0 - vec2<f32>(1.0);
  let radiusTexels = project_pixel_size_float(weight.radiusPixels) * weight.textureWidth /
    (weight.commonBounds.z - weight.commonBounds.x);
  let offset = WEIGHT_QUAD_POSITIONS[attributes.vertexIndex];
  output.position = vec4<f32>(clipPosition + offset * radiusTexels * 2.0 / weight.textureWidth, 0.0, 1.0);
  output.pointCoordinates = (offset + vec2<f32>(1.0)) * 0.5;
  let sourceWeight = select(0.0, attributes.weights, attributes.weightValidity != 0u && heatmapFinite(attributes.weights));
  output.weights = vec4<f32>(sourceWeight * weight.weightsScale, 0.0, 0.0, 1.0);
  return output;
}
@fragment fn fragmentMain(varyings: WeightVaryings) -> @location(0) vec4<f32> {
  let distanceFromCenter = length(varyings.pointCoordinates - vec2<f32>(0.5));
  if (distanceFromCenter > 0.5) { discard; }
  let gaussian = exp(-pow(2.0 * distanceFromCenter, 2.0) / 0.05555) /
    (1.77245385 * 0.166666);
  return varyings.weights * gaussian;
}
`;
}

/** Batches all source chunks into one additive native density texture without source copying. */
export class HeatmapWeightRuntime {
  readonly transform: TextureTransform;
  readonly readbackByteLength = 0;

  constructor(
    device: Device,
    targetTexture: Texture,
    readonly inputs: ResidentHeatmapInputs
  ) {
    // This adapter deliberately requires the native floating-point WebGPU blend path.
    assert(device.type === 'webgpu' && targetTexture.format === 'rgba16float');
    // The native splat uses a square texture and one canonical vertex format per input.
    assert(targetTexture.width === targetTexture.height && targetTexture.width > 0);
    assert(inputs.selectionMask.format === 'uint32');
    assert(!inputs.weights || inputs.weights.format === 'float32');
    assert(!inputs.weightValidity || inputs.weightValidity.format === 'uint32');
    const positionFormat = inputs.positions.format;
    assert(positionFormat === 'float32x2' || positionFormat === 'float32x3');
    // Reject incompatible source topology before allocating transform resources.
    getGPUVectorModelBatches('resident-heatmap', {
      positions: inputs.positions,
      selectionMask: inputs.selectionMask,
      weights: inputs.weights,
      weightValidity: inputs.weightValidity
    });
    this.transform = new TextureTransform(device, {
      id: `${targetTexture.id}-resident-weights`,
      source: getResidentHeatmapWeightShader(positionFormat),
      modules: [project32, weightUniforms],
      targetTexture,
      targetTextureChannels: 4,
      targetTextureVarying: 'weights',
      topology: 'triangle-list',
      vertexCount: 6,
      isInstanced: true,
      inferDrawCount: 'instance',
      attributeVectors: {
        positions: inputs.positions,
        weights: inputs.weights ?? new GPUConstant({format: 'float32', value: Float32Array.of(1)}),
        selectionMask: inputs.selectionMask,
        weightValidity:
          inputs.weightValidity ?? new GPUConstant({format: 'uint32', value: Uint32Array.of(1)})
      },
      colorAttachmentFormats: ['rgba16float'],
      parameters: {
        depthWriteEnabled: false,
        blend: true,
        blendColorOperation: 'add',
        blendColorSrcFactor: 'one',
        blendColorDstFactor: 'one',
        blendAlphaSrcFactor: 'one',
        blendAlphaDstFactor: 'one'
      }
    });
  }

  render(settings: {
    viewport: Viewport;
    devicePixelRatio: number;
    coordinateSystem: LayerProps['coordinateSystem'];
    coordinateOrigin: LayerProps['coordinateOrigin'];
    modelMatrix: LayerProps['modelMatrix'];
    commonBounds: readonly [number, number, number, number];
    radiusPixels: number;
  }): boolean {
    const {transform} = this;
    const textureSize = transform.getTargetTexture().width;
    transform.model.shaderInputs.setProps({
      project: settings,
      weight: {
        commonBounds: settings.commonBounds,
        radiusPixels: settings.radiusPixels,
        textureWidth: textureSize,
        weightsScale: 1
      }
    });
    transform.run({
      parameters: {viewport: [0, 0, textureSize, textureSize]},
      clearColor: [0, 0, 0, 0]
    });
    return !transform.model.pipeline.isPending && !transform.model.needsRedraw();
  }

  destroy(): void {
    this.transform.destroy();
    // TextureTransform owns this sampler separately from its model/framebuffer.
    this.transform.sampler.destroy();
  }
}
