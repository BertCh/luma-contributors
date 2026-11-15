// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  picking,
  project32,
  type Color,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {type Buffer, type RenderPass} from '@luma.gl/core';
import {GPUVectorModel} from '@luma.gl/engine';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  applyGPUBrushingToShader,
  gpuBrushing,
  setGPUBrushingProps,
  type GPUBrushingProps
} from './gpu-brushing';
import {
  createGPUVectorStyleBuffer,
  getGPUVectorStyleBufferBinding,
  getGPUVectorBuffer,
  getGPUVectorLayerBatches,
  getGPUVectorPickingProvenance,
  makeGPUVectorBufferLayout,
  resizeGPUVectorStyleBuffer,
  makeGPUVectorSelectionLayout,
  validateGPUVectorPickingCapacity,
  type GPUVectorLayerSelectionProps,
  type GPUVectorLayerPickingInfo
} from './gpu-vector-layer-utils';
import {GPU_HIGHLIGHT_WGSL} from './gpu-highlight';
import {
  WGSL_SHADOW_VARYING_FIELDS,
  ensureWGSLShadowInputs,
  wgslShadow
} from '../effects/wgsl-shadow';
import {
  applyEndpointAltitudeToShader,
  GPU_ENDPOINT_POSITION_FORMATS,
  type GPUEndpointPositionVector
} from './gpu-endpoint-altitude';

/** GPUVector-native straight line props. All input vectors are borrowed. */
export type GPULineLayerProps = Omit<LayerProps, 'data'> &
  GPUVectorLayerSelectionProps &
  GPUBrushingProps & {
    /** float32x2 longitude/latitude, or float32x3 with altitude in meters. */
    getSourcePosition: GPUEndpointPositionVector;
    getTargetPosition: GPUEndpointPositionVector;
    /**
     * Optional fp64 low parts of the endpoint longitude/latitude: `getSourcePosition +
     * getSourcePositionLow` reconstructs the Float64 position, which Deck's projection adds after
     * subtracting the viewport origin in Float32. Share the endpoint vector's chunk boundaries.
     * Defaults to zero. GlobeView great-circle arc interpolation ignores the low parts.
     */
    getSourcePositionLow?: GPUVector<'float32x2'>;
    getTargetPositionLow?: GPUVector<'float32x2'>;
    /** Source RGBA color; also colors the target when getTargetColor is omitted. */
    getColor?: Color | GPUVector<'unorm8x4'>;
    /** Optional target RGBA, interpolated along the line with the source color. */
    getTargetColor?: Color | GPUVector<'unorm8x4'>;
    getWidth?: number | GPUVector<'float32'>;
    widthScale?: number;
    widthMinPixels?: number;
    widthMaxPixels?: number;
    /** Multiplies float32x3 endpoint altitude, matching Kepler's line `elevationScale`. */
    elevationScale?: number;
  };

type GPULineLayerState = {
  model: GPUVectorModel | null;
  styleBuffer: Buffer | null;
  defaults: Buffer[];
};

const STYLE_BYTE_LENGTH = 80;

function getLineShader(
  brushing: boolean,
  sourcePosition: GPUEndpointPositionVector,
  targetPosition: GPUEndpointPositionVector
): string {
  return applyGPUBrushingToShader(
    applyEndpointAltitudeToShader(
      GPU_LINE_SHADER,
      sourcePosition.format,
      targetPosition.format,
      'lineStyle.elevationScale'
    ),
    brushing,
    'brushing_arePointsInRange(sourceLongitudeLatitude, targetLongitudeLatitude)'
  );
}

const GPU_LINE_SHADER = /* wgsl */ `
${GPU_HIGHLIGHT_WGSL}
struct LineStyle {
  color: vec4<f32>,
  width: f32,
  widthScale: f32,
  widthMinPixels: f32,
  widthMaxPixels: f32,
  useColors: u32,
  useWidths: u32,
  rowIndexOffset: u32,
  hasTargetColor: u32,
  targetColor: vec4<f32>,
  useTargetColors: u32,
  elevationScale: f32,
  _padding1: u32,
  _padding2: u32,
};
@group(0) @binding(auto) var<uniform> lineStyle: LineStyle;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32>, @location(1) @interpolate(flat) pickingColor: vec3<f32>, @location(2) @interpolate(flat) selected: u32, ${WGSL_SHADOW_VARYING_FIELDS(3)} };
fn encodePickingColor(rowIndex: u32) -> vec3<f32> { let value = rowIndex + 1u; return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0; }
fn getCorner(vertexIndex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}
@vertex fn vertexMain(
  @location(0) sourcePositions: vec2<f32>,
  @location(1) targetPositions: vec2<f32>,
  @location(2) colors: vec4<f32>,
  @location(3) widths: f32,
  @location(4) selectionMask: u32,
  @location(5) targetColors: vec4<f32>,
  @location(6) sourcePositionsLow: vec2<f32>,
  @location(7) targetPositionsLow: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let corner = getCorner(vertexIndex);
  let sourceProjected = project_position_to_clipspace_and_commonspace(vec3<f32>(sourcePositions, sourceAltitude), vec3<f32>(sourcePositionsLow, 0.0), vec3<f32>(0.0));
  let targetProjected = project_position_to_clipspace_and_commonspace(vec3<f32>(targetPositions, targetAltitude), vec3<f32>(targetPositionsLow, 0.0), vec3<f32>(0.0));
  let sourceClip = sourceProjected.clipPosition;
  let targetClip = targetProjected.clipPosition;
  let delta = targetClip.xy / targetClip.w - sourceClip.xy / sourceClip.w;
  let direction = select(vec2<f32>(1.0, 0.0), normalize(delta), length(delta) > 0.000001);
  let normal = vec2<f32>(-direction.y, direction.x);
  let width = clamp(
    select(lineStyle.width, widths, lineStyle.useWidths != 0u) * lineStyle.widthScale,
    lineStyle.widthMinPixels,
    lineStyle.widthMaxPixels
  );
  let commonPosition = mix(sourceProjected.commonPosition, targetProjected.commonPosition, corner.x);
  // Like Deck, the shadow map swap happens before the pixel extrusion is added.
  var clipPosition = shadow_setVertexPosition(commonPosition, mix(sourceClip, targetClip, corner.x));
  clipPosition = vec4<f32>(clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * width * 0.5) * clipPosition.w, clipPosition.z, clipPosition.w);
  let pickingColor = encodePickingColor(instanceIndex + lineStyle.rowIndexOffset);
  geometry.worldPosition = vec3<f32>(mix(sourcePositions, targetPositions, corner.x), mix(sourceAltitude, targetAltitude, corner.x)); geometry.pickingColor = pickingColor;
  var output: VertexOutput;
  output.position = clipPosition;
  output.shadowPosition0 = shadow_getVertexPosition(0u, commonPosition);
  output.shadowPosition1 = shadow_getVertexPosition(1u, commonPosition);
  let sourceColor = select(lineStyle.color, colors, lineStyle.useColors != 0u);
  let targetColor = select(sourceColor, select(lineStyle.targetColor, targetColors, lineStyle.useTargetColors != 0u), lineStyle.hasTargetColor != 0u);
  output.color = mix(sourceColor, targetColor, corner.x);
  output.pickingColor = pickingColor;
  output.selected = selectionMask;
  return output;
}
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (input.selected == 0u) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  // Interpolate straight RGBA first, then match Deck's premultiplied-alpha blend contract.
  let color = gpuHighlight(input.color, input.pickingColor);
  let alpha = color.a * layer.opacity;
  return shadow_filterPremultiplied(vec4<f32>(color.rgb, alpha), input.position.z, input.shadowPosition0, input.shadowPosition1);
}
`;

/** Chunk-preserving GPUVector straight-line layer. */
export class GPULineLayer extends Layer<GPULineLayerProps> {
  static override layerName = 'GPULineLayer';
  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPULineLayer requires WebGPU');
    validateGPUVectorPickingCapacity(
      this.props.getSourcePosition.length,
      Boolean(this.props.pickable)
    );
    const selectionMask = this.props.selectionMask;
    const colors = isGPUVector(this.props.getColor) ? this.props.getColor : undefined;
    const targetColors = isGPUVector(this.props.getTargetColor)
      ? this.props.getTargetColor
      : undefined;
    const widths = isGPUVector(this.props.getWidth) ? this.props.getWidth : undefined;
    getGPUVectorLayerBatches(
      this.id,
      {
        sourcePositions: this.props.getSourcePosition,
        targetPositions: this.props.getTargetPosition,
        sourcePositionsLow: this.props.getSourcePositionLow,
        targetPositionsLow: this.props.getTargetPositionLow,
        colors,
        targetColors,
        widths,
        selectionMask
      },
      {
        sourcePositions: GPU_ENDPOINT_POSITION_FORMATS,
        targetPositions: GPU_ENDPOINT_POSITION_FORMATS,
        sourcePositionsLow: ['float32x2'],
        targetPositionsLow: ['float32x2'],
        colors: ['unorm8x4'],
        targetColors: ['unorm8x4'],
        widths: ['float32'],
        selectionMask: ['uint32']
      }
    );
    if (this.props.getSourcePosition.data.length === 0) {
      this.setState({model: null, styleBuffer: null, defaults: []} satisfies GPULineLayerState);
      return;
    }
    const styleBuffer = createGPUVectorStyleBuffer(
      device,
      `${this.id}-style`,
      STYLE_BYTE_LENGTH,
      this.props.getSourcePosition.data.length
    );
    const defaultColor = device.createBuffer({
      id: `${this.id}-default-color`,
      data: new Uint8Array([0, 0, 0, 255])
    });
    const defaultWidth = device.createBuffer({
      id: `${this.id}-default-width`,
      data: new Float32Array([1])
    });
    const defaultSelection = device.createBuffer({data: new Uint32Array([1])});
    const defaultPositionLow = device.createBuffer({data: new Float32Array([0, 0])});
    const sourcePositionsLow = this.props.getSourcePositionLow;
    const targetPositionsLow = this.props.getTargetPositionLow;
    const model = new GPUVectorModel(device, {
      shaderAssembler: this.context.shaderAssembler,
      ...this.getShaders({
        modules: this.props.brushingEnabled
          ? [project32, picking, gpuBrushing, wgslShadow]
          : [project32, picking, wgslShadow],
        source: getLineShader(
          Boolean(this.props.brushingEnabled),
          this.props.getSourcePosition,
          this.props.getTargetPosition
        )
      }),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      inferDrawCount: 'instance',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: 0,
      attributes: {
        sourcePositions: getGPUVectorBuffer(this.props.getSourcePosition),
        targetPositions: getGPUVectorBuffer(this.props.getTargetPosition),
        sourcePositionsLow: sourcePositionsLow
          ? getGPUVectorBuffer(sourcePositionsLow)
          : defaultPositionLow,
        targetPositionsLow: targetPositionsLow
          ? getGPUVectorBuffer(targetPositionsLow)
          : defaultPositionLow,
        colors: colors ? getGPUVectorBuffer(colors) : defaultColor,
        targetColors: targetColors ? getGPUVectorBuffer(targetColors) : defaultColor,
        widths: widths ? getGPUVectorBuffer(widths) : defaultWidth,
        selectionMask: selectionMask ? getGPUVectorBuffer(selectionMask) : defaultSelection
      },
      bufferLayout: [
        makeGPUVectorSelectionLayout(selectionMask),
        makeGPUVectorBufferLayout(this.props.getSourcePosition, 'sourcePositions'),
        makeGPUVectorBufferLayout(this.props.getTargetPosition, 'targetPositions'),
        sourcePositionsLow
          ? makeGPUVectorBufferLayout(sourcePositionsLow, 'sourcePositionsLow')
          : makeConstantLayout('sourcePositionsLow', 'float32x2'),
        targetPositionsLow
          ? makeGPUVectorBufferLayout(targetPositionsLow, 'targetPositionsLow')
          : makeConstantLayout('targetPositionsLow', 'float32x2'),
        colors
          ? makeGPUVectorBufferLayout(colors, 'colors')
          : {
              name: 'colors',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'colors', format: 'unorm8x4'}]
            },
        targetColors
          ? makeGPUVectorBufferLayout(targetColors, 'targetColors')
          : {
              name: 'targetColors',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'targetColors', format: 'unorm8x4'}]
            },
        widths
          ? makeGPUVectorBufferLayout(widths, 'widths')
          : {
              name: 'widths',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'widths', format: 'float32'}]
            }
      ],
      bindings: {lineStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH)}
    });
    model.userData['boundInputs'] = [
      this.props.getSourcePosition,
      this.props.getTargetPosition,
      colors,
      widths,
      selectionMask,
      targetColors,
      Boolean(this.props.brushingEnabled),
      sourcePositionsLow,
      targetPositionsLow
    ];
    this.setState({
      model,
      styleBuffer,
      defaults: [defaultColor, defaultWidth, defaultSelection, defaultPositionLow]
    } satisfies GPULineLayerState);
  }

  override getModels(): GPUVectorModel[] {
    const model = (this.state as GPULineLayerState).model;
    return model ? [model] : [];
  }
  override updateState({props}: UpdateParameters<this>): void {
    const boundInputs = ((this.state as GPULineLayerState).model?.userData['boundInputs'] ??
      []) as unknown[];
    const colors = isGPUVector(props.getColor) ? props.getColor : undefined;
    const targetColors = isGPUVector(props.getTargetColor) ? props.getTargetColor : undefined;
    const widths = isGPUVector(props.getWidth) ? props.getWidth : undefined;
    if (
      props.getSourcePosition !== boundInputs[0] ||
      props.getTargetPosition !== boundInputs[1] ||
      colors !== boundInputs[2] ||
      widths !== boundInputs[3] ||
      props.selectionMask !== boundInputs[4] ||
      targetColors !== boundInputs[5] ||
      Boolean(props.brushingEnabled) !== boundInputs[6] ||
      props.getSourcePositionLow !== boundInputs[7] ||
      props.getTargetPositionLow !== boundInputs[8]
    ) {
      this.destroyResources();
      this.initializeState(this.context);
    }
  }
  override draw({renderPass}: {renderPass: RenderPass}): void {
    validateGPUVectorPickingCapacity(
      this.props.getSourcePosition.length,
      Boolean(this.props.pickable)
    );
    const selectionMask = this.props.selectionMask;
    const {model} = this.state as GPULineLayerState;
    const {styleBuffer: previousStyleBuffer} = this.state as GPULineLayerState;
    if (!model || !previousStyleBuffer) return;
    if (model.shaderInputs) ensureWGSLShadowInputs([model], this.context.device);
    const styleBuffer = resizeGPUVectorStyleBuffer(
      previousStyleBuffer,
      STYLE_BYTE_LENGTH,
      this.props.getSourcePosition.data.length
    );
    if (styleBuffer !== previousStyleBuffer) {
      this.setState({styleBuffer});
      model.setBindings({
        lineStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH)
      });
    }
    const colors = isGPUVector(this.props.getColor) ? this.props.getColor : undefined;
    const targetColors = isGPUVector(this.props.getTargetColor)
      ? this.props.getTargetColor
      : undefined;
    const widths = isGPUVector(this.props.getWidth) ? this.props.getWidth : undefined;
    const [red, green, blue, alpha = 255] = isColor(this.props.getColor)
      ? this.props.getColor
      : [0, 0, 0, 255];
    const [targetRed, targetGreen, targetBlue, targetAlpha = 255] = isColor(
      this.props.getTargetColor
    )
      ? this.props.getTargetColor
      : [0, 0, 0, 255];
    const bytes = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(bytes);
    const uints = new Uint32Array(bytes);
    floats.set([
      red / 255,
      green / 255,
      blue / 255,
      alpha / 255,
      typeof this.props.getWidth === 'number' ? this.props.getWidth : 1,
      this.props.widthScale ?? 1,
      this.props.widthMinPixels ?? 0,
      this.props.widthMaxPixels ?? 1e9
    ]);
    uints.set(
      [colors ? 1 : 0, widths ? 1 : 0, 0, this.props.getTargetColor === undefined ? 0 : 1],
      8
    );
    floats.set([targetRed / 255, targetGreen / 255, targetBlue / 255, targetAlpha / 255], 12);
    uints[16] = targetColors ? 1 : 0;
    floats[17] = this.props.elevationScale ?? 1;
    setGPUBrushingProps(model, this.context, this.props);
    model.drawBatches(renderPass, {
      vectors: {
        sourcePositions: this.props.getSourcePosition,
        targetPositions: this.props.getTargetPosition,
        sourcePositionsLow: this.props.getSourcePositionLow,
        targetPositionsLow: this.props.getTargetPositionLow,
        colors,
        targetColors,
        widths,
        selectionMask
      },
      onBatch: batch => {
        uints[10] = batch.rowIndexOffset;
        const binding = getGPUVectorStyleBufferBinding(
          styleBuffer,
          STYLE_BYTE_LENGTH,
          batch.batchIndex
        );
        styleBuffer.write(new Uint8Array(bytes), binding.offset);
        model.setBindings({lineStyle: binding});
      }
    });
  }
  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    const pickingInfo = info as GPUVectorLayerPickingInfo;
    pickingInfo.gpuVector = getGPUVectorPickingProvenance(
      this.props.getSourcePosition,
      pickingInfo.index
    );
    return pickingInfo;
  }
  override finalizeState(context: LayerContext): void {
    this.destroyResources();
    super.finalizeState(context);
  }
  private destroyResources(): void {
    const state = this.state as GPULineLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.defaults.forEach(buffer => buffer.destroy());
    this.setState({model: null, styleBuffer: null, defaults: []});
  }
}

function makeConstantLayout(name: string, format: 'float32x2') {
  return {
    name,
    byteStride: 0,
    stepMode: 'instance' as const,
    attributes: [{attribute: name, format}]
  };
}

function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
function isColor(value: unknown): value is Color {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}
