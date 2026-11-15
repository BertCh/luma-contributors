// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) Flowmap.gl contributors
// SPDX-FileCopyrightText: Copyright (c) 2018-2020 Teralytics

// Shader logic in this file is adapted from the FlowCirclesLayer GLSL of flowmap.gl
// (Apache-2.0, Copyright (c) Flowmap.gl contributors, Copyright (c) 2018-2020 Teralytics),
// ported to WGSL for the WebGPU device.

import {
  Layer,
  picking,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {type Buffer, type BufferLayout, type Device, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';
import {
  createFlowInstanceAttribute,
  packFlowPositions,
  type GPUFlowAttribute
} from './gpu-flow-lines-layer';

/** Structural copy of flowmap.gl `FlowCirclesLayerAttributes`; no flowmap.gl import is required. */
export type GPUFlowCirclesData = {
  length: number;
  attributes: {
    /** Float64 lng/lat(/z) per row. */
    getPosition: GPUFlowAttribute<Float64Array | Float32Array>;
    /** RGBA bytes, four per row. */
    getColor: GPUFlowAttribute<Uint8Array | Uint8ClampedArray>;
    /** Inner radius in pixels per row. */
    getInRadius: GPUFlowAttribute<Float32Array>;
    /** Outer radius in pixels per row. */
    getOutRadius: GPUFlowAttribute<Float32Array>;
    /** Non-zero values tint the circle red; zero when omitted. */
    getOutOfScale?: GPUFlowAttribute<Float32Array>;
  };
};

export type GPUFlowCirclesLayerProps = Omit<LayerProps, 'data'> & {
  data: GPUFlowCirclesData;
  /** RGBA 0-255 fill of the empty part of the circle, defaults to white. */
  emptyColor?: readonly [number, number, number, number];
  /** Mix of the empty color into the outline, defaults to 0.4. */
  outlineEmptyMix?: number;
};

type GPUFlowCirclesLayerState = {
  models: Model[];
  buffers: Buffer[];
  boundData: GPUFlowCirclesData | null;
};

/** Uniforms of the flow circle shader. */
export const gpuFlowCircles = {
  name: 'flowCircles',
  uniformTypes: {
    emptyColor: 'vec4<f32>',
    parameters: 'vec4<f32>'
  },
  defaultUniforms: {
    emptyColor: [1, 1, 1, 1],
    parameters: [0.4, 0, 0, 0]
  },
  source: /* wgsl */ `
struct flowCirclesUniforms {
  // Linear RGBA color of the empty part of the circle.
  emptyColor: vec4<f32>,
  // x outlineEmptyMix; yzw unused.
  parameters: vec4<f32>,
};
@group(0) @binding(auto) var<uniform> flowCircles: flowCirclesUniforms;
`
} as const satisfies ShaderModule;

/** WGSL for flow circles, derived from flowmap.gl FlowCirclesLayer. */
export const GPU_FLOW_CIRCLES_WGSL = /* wgsl */ `
const SOFT_OUTLINE: f32 = 0.05;
const EPS: f32 = 0.05;
struct FlowCircleVertex {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) unitPosition: vec2<f32>,
  @location(2) @interpolate(flat) unitRadii: vec2<f32>,
  @location(3) @interpolate(flat) outOfScale: f32,
  @location(4) @interpolate(flat) pickingColor: vec3<f32>,
};
fn encodePickingColor(rowIndex: u32) -> vec3<f32> {
  let value = rowIndex + 1u;
  return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0;
}
fn getCorner(vertexIndex: u32) -> vec2<f32> {
  // A square that minimally covers the unit circle.
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}
@vertex fn vertexMain(
  @location(0) positions: vec3<f32>,
  @location(1) positionsLow: vec3<f32>,
  @location(2) inRadii: f32,
  @location(3) outRadii: f32,
  @location(4) colors: vec4<f32>,
  @location(5) outOfScales: f32,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> FlowCircleVertex {
  let corner = getCorner(vertexIndex);
  let outerRadiusPixels = max(max(inRadii, outRadii), 0.0001);
  let offset = vec3<f32>(corner * project_pixel_size_float(outerRadiusPixels), 0.0);
  var output: FlowCircleVertex;
  output.position = project_position_to_clipspace(positions, positionsLow, offset);
  output.color = colors;
  output.unitPosition = corner;
  output.unitRadii = vec2<f32>(inRadii, outRadii) / outerRadiusPixels;
  output.outOfScale = outOfScales;
  output.pickingColor = encodePickingColor(instanceIndex);
  return output;
}
@fragment fn fragmentMain(input: FlowCircleVertex) -> @location(0) vec4<f32> {
  let distanceToCenter = length(input.unitPosition);
  if (distanceToCenter > 1.0) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let emptyColor = flowCircles.emptyColor;
  let unitInRadius = input.unitRadii.x;
  let unitOutRadius = input.unitRadii.y;
  let isInGreater = select(0.0, 1.0, unitInRadius > unitOutRadius);
  let ringColor = mix(emptyColor, input.color, isInGreater);
  let outlineColor = mix(mix(input.color, emptyColor, flowCircles.parameters.x), input.color, isInGreater);
  let innerRadius = min(unitInRadius, unitOutRadius) * (1.0 - SOFT_OUTLINE);
  let step2 = innerRadius - 2.0 * EPS;
  let step3 = innerRadius - EPS;
  let step4 = innerRadius;
  let step5 = 1.0 - 5.0 * EPS;
  let step6 = 1.0;
  var color = input.color;
  color = mix(color, emptyColor, smoothstep(step2, step3, distanceToCenter));
  color = mix(color, ringColor, smoothstep(step3, step4, distanceToCenter));
  color = mix(color, outlineColor, smoothstep(step5, step6, distanceToCenter));
  let rgb = mix(color.rgb, vec3<f32>(1.0, 0.188235, 0.188235), step(0.5, input.outOfScale));
  let alpha = input.color.a * layer.opacity * smoothstep(0.0, SOFT_OUTLINE, 1.0 - distanceToCenter);
  return vec4<f32>(rgb * alpha, alpha);
}
`;

/** Deck layer drawing flowmap.gl style location circles (inner/outer radius rings) from binary attributes on WebGPU. */
export class GPUFlowCirclesLayer extends Layer<GPUFlowCirclesLayerProps> {
  static override layerName = 'GPUFlowCirclesLayer';
  static override defaultProps = {
    emptyColor: [255, 255, 255, 255],
    outlineEmptyMix: 0.4,
    parameters: {depthCompare: 'always', depthWriteEnabled: false, cullMode: 'none'}
  };
  override getAttributeManager() {
    return null;
  }

  override initializeState(context: LayerContext): void {
    this.setState({models: [], buffers: [], boundData: null} satisfies GPUFlowCirclesLayerState);
    this.rebuildResources(context.device);
  }

  override updateState({props}: UpdateParameters<this>): void {
    if (props.data !== (this.state as GPUFlowCirclesLayerState).boundData) {
      this.rebuildResources(this.context.device);
    }
  }

  override getModels(): Model[] {
    return (this.state as GPUFlowCirclesLayerState).models;
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const [red, green, blue, alpha] = this.props.emptyColor ?? [255, 255, 255, 255];
    for (const model of (this.state as GPUFlowCirclesLayerState).models) {
      model.shaderInputs.setProps({
        flowCircles: {
          emptyColor: [red / 255, green / 255, blue / 255, alpha / 255],
          parameters: [this.props.outlineEmptyMix ?? 0.4, 0, 0, 0]
        }
      });
      model.draw(renderPass);
    }
  }

  override finalizeState(context: LayerContext): void {
    this.destroyResources();
    super.finalizeState(context);
  }

  private rebuildResources(device: Device): void {
    this.destroyResources();
    const {data} = this.props;
    const rowCount = data?.length ?? 0;
    this.setState({boundData: data ?? null});
    if (rowCount === 0) return;

    const attributes = data.attributes;
    const buffers: Buffer[] = [];
    const layouts: BufferLayout[] = [];
    const bindings: Record<string, Buffer> = {};
    const addAttribute = (
      name: string,
      array: Float32Array | Uint8Array | Uint8ClampedArray,
      format: 'float32' | 'float32x3' | 'unorm8x4'
    ) => {
      const attribute = createFlowInstanceAttribute(
        device,
        `${this.id}-${name}`,
        name,
        array,
        format
      );
      buffers.push(attribute.buffer);
      layouts.push(attribute.layout);
      bindings[name] = attribute.buffer;
    };
    const positions = packFlowPositions(
      attributes.getPosition.value,
      attributes.getPosition.size,
      rowCount
    );
    addAttribute('positions', positions.high, 'float32x3');
    addAttribute('positionsLow', positions.low, 'float32x3');
    addAttribute('inRadii', attributes.getInRadius.value, 'float32');
    addAttribute('outRadii', attributes.getOutRadius.value, 'float32');
    addAttribute('colors', attributes.getColor.value, 'unorm8x4');
    addAttribute(
      'outOfScales',
      attributes.getOutOfScale?.value ?? new Float32Array(rowCount),
      'float32'
    );

    const model = new Model(device, {
      ...this.getShaders({
        source: GPU_FLOW_CIRCLES_WGSL,
        modules: [project32, picking, gpuFlowCircles]
      }),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: rowCount,
      attributes: bindings,
      bufferLayout: layouts
    });
    this.setState({models: [model], buffers});
  }

  private destroyResources(): void {
    const state = this.state as GPUFlowCirclesLayerState;
    state.models?.forEach(model => model.destroy());
    state.buffers?.forEach(buffer => buffer.destroy());
    this.setState({models: [], buffers: []});
  }
}
