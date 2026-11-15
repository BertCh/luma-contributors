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
  WGSL_SHADOW_VARYING_FIELDS,
  ensureWGSLShadowInputs,
  wgslShadow
} from '../effects/wgsl-shadow';
import {GPU_GLOBE_WGSL} from './gpu-globe';
import {GPU_HIGHLIGHT_WGSL} from './gpu-highlight';
import {
  applyEndpointAltitudeToShader,
  GPU_ENDPOINT_POSITION_FORMATS,
  type GPUEndpointPositionVector
} from './gpu-endpoint-altitude';
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

/** Native Kepler target arrow shapes. */
export type GPUArrowHeadShape = 'chevron' | 'triangle' | 'stealth';

/**
 * Screen-aligned arrowheads derived on the GPU from borrowed source and target positions.
 * One instance is drawn per connection row; the head tip sits at the target and the body is
 * oriented along the final segment of the straight line or Deck arc, with sizes in pixels.
 */
export type GPUArrowHeadLayerProps = Omit<LayerProps, 'data'> &
  GPUVectorLayerSelectionProps &
  GPUBrushingProps & {
    /** float32x2 longitude/latitude, or float32x3 with altitude in meters. */
    getSourcePosition: GPUEndpointPositionVector;
    getTargetPosition: GPUEndpointPositionVector;
    /**
     * Optional fp64 low parts of the endpoint longitude/latitude (float32x2, altitude low is zero).
     * Chunk boundaries must match the endpoint vectors. Applied to flat projections; GlobeView
     * interpolates great circles in Float32 and ignores them. Defaults to zero.
     */
    getSourcePositionLow?: GPUVector<'float32x2'>;
    getTargetPositionLow?: GPUVector<'float32x2'>;
    getColor?: Color | GPUVector<'unorm8x4'>;
    shape?: GPUArrowHeadShape;
    /** Head length in pixels; Kepler `targetArrowSize`. */
    sizePixels?: number;
    /** Follow a straight projected segment or the Deck arc profile used by GPUArcLayer. */
    mode?: 'line' | 'arc';
    /** Arc distance multiplier matching GPUArcLayer `getHeight`. */
    arcHeight?: number;
    /** Multiplies float32x3 endpoint altitude, matching Kepler's arrow `elevationScale`. */
    elevationScale?: number;
  };

type GPUArrowHeadLayerState = {
  model: GPUVectorModel | null;
  styleBuffer: Buffer | null;
  defaults: Buffer[];
};

const STYLE_BYTE_LENGTH = 64;
const ARC_SEGMENT_COUNT = 50;
const SHAPE_INDICES: Record<GPUArrowHeadShape, number> = {chevron: 0, triangle: 1, stealth: 2};

export const GPU_ARROW_HEAD_SHADER = /* wgsl */ `
${GPU_HIGHLIGHT_WGSL}
${GPU_GLOBE_WGSL}
struct ArrowStyle {
  color: vec4<f32>,
  sizePixels: f32,
  shape: u32,
  mode: u32,
  rowIndexOffset: u32,
  arcHeight: f32,
  useColors: u32,
  segmentCount: f32,
  elevationScale: f32,
  _padding1: u32,
  _padding2: u32,
  _padding3: u32,
  _padding4: u32,
};
@group(0) @binding(auto) var<uniform> arrowStyle: ArrowStyle;
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) @interpolate(flat) pickingColor: vec3<f32>,
  @location(2) @interpolate(flat) selected: u32,
  ${WGSL_SHADOW_VARYING_FIELDS(3)}
};
fn encodePickingColor(rowIndex: u32) -> vec3<f32> {
  let value = rowIndex + 1u;
  return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0;
}
// Mesh tip at the origin, body toward -y: matches Kepler's ArrowHeadLayer geometries.
fn getMeshVertex(shape: u32, vertexIndex: u32) -> vec2<f32> {
  var chevron = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(0.0, -0.55), vec2<f32>(-0.55, -1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(0.55, -1.0), vec2<f32>(0.0, -0.55)
  );
  var triangle = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(-0.45, -1.0), vec2<f32>(0.45, -1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0)
  );
  var stealth = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(0.0, -0.7), vec2<f32>(-0.22, -1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(0.22, -1.0), vec2<f32>(0.0, -0.7)
  );
  if (shape == 1u) { return triangle[vertexIndex]; }
  if (shape == 2u) { return stealth[vertexIndex]; }
  return chevron[vertexIndex];
}
fn getProjectedArcPosition(sourcePosition: vec3<f32>, targetPosition: vec3<f32>, progress: f32, height: f32) -> vec3<f32> {
  let linear = mix(sourcePosition, targetPosition, progress);
  let elevation = sqrt(max(progress * (1.0 - progress), 0.0)) * length(targetPosition.xy - sourcePosition.xy) * height;
  return vec3<f32>(linear.xy, linear.z + elevation);
}
@vertex fn vertexMain(
  @location(0) sourcePositions: vec2<f32>,
  @location(1) targetPositions: vec2<f32>,
  @location(2) colors: vec4<f32>,
  @location(3) selectionMask: u32,
  @location(4) sourcePositionsLow: vec2<f32>,
  @location(5) targetPositionsLow: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let targetProjected = project_position_to_clipspace_and_commonspace(vec3<f32>(targetPositions, targetAltitude), vec3<f32>(targetPositionsLow, 0.0), vec3<f32>(0.0));
  let targetClip = targetProjected.clipPosition;
  var tipCommon = targetProjected.commonPosition;
  var previousClip: vec4<f32>;
  var tipClip = targetClip;
  var globeHidden = gpuGlobeIsHidden(vec3<f32>(targetPositions, targetAltitude));
  if (arrowStyle.mode == 1u && gpuGlobeUsesGreatCircle()) {
    // GlobeView connections are great circles: orient to the tangent of the last arc segment.
    let previousProgress = smoothstep(0.0, 1.0, (arrowStyle.segmentCount - 1.0) / arrowStyle.segmentCount);
    let angularDistance = gpuGlobeAngularDistance(sourcePositions, targetPositions);
    var previousGlobe = gpuGlobeInterpolate(sourcePositions, targetPositions, angularDistance, previousProgress, arrowStyle.arcHeight);
    previousGlobe.z += mix(sourceAltitude, targetAltitude, previousProgress);
    previousClip = project_position_to_clipspace(previousGlobe, vec3<f32>(0.0), vec3<f32>(0.0));
    globeHidden = globeHidden || abs(previousGlobe.x - targetPositions.x) > 180.0;
  } else if (arrowStyle.mode == 1u) {
    let previousProgress = smoothstep(0.0, 1.0, (arrowStyle.segmentCount - 1.0) / arrowStyle.segmentCount);
    let sourceCommon = project_position_vec3_f64(vec3<f32>(sourcePositions, sourceAltitude), vec3<f32>(sourcePositionsLow, 0.0));
    let targetCommon = project_position_vec3_f64(vec3<f32>(targetPositions, targetAltitude), vec3<f32>(targetPositionsLow, 0.0));
    previousClip = project_common_position_to_clipspace(vec4<f32>(getProjectedArcPosition(sourceCommon, targetCommon, previousProgress, arrowStyle.arcHeight), 1.0));
    tipClip = project_common_position_to_clipspace(vec4<f32>(targetCommon, 1.0));
    tipCommon = vec4<f32>(targetCommon, 1.0);
  } else {
    previousClip = project_position_to_clipspace(vec3<f32>(sourcePositions, sourceAltitude), vec3<f32>(sourcePositionsLow, 0.0), vec3<f32>(0.0));
  }
  // Pixel-space travel direction at the tip; a degenerate segment points up like Kepler's zero angle.
  let delta = (tipClip.xy / tipClip.w - previousClip.xy / previousClip.w) * project.viewportSize;
  let direction = select(vec2<f32>(0.0, 1.0), normalize(delta), length(delta) > 0.001);
  let right = vec2<f32>(direction.y, -direction.x);
  let vertex = getMeshVertex(arrowStyle.shape, vertexIndex);
  let offsetPixels = (right * vertex.x + direction * vertex.y) * arrowStyle.sizePixels;
  // Like Deck, the shadow map swap happens on the tip before the pixel offset is added.
  var clipPosition = shadow_setVertexPosition(tipCommon, tipClip);
  clipPosition = vec4<f32>(clipPosition.xy + project_pixel_size_to_clipspace(offsetPixels) * clipPosition.w, clipPosition.z, clipPosition.w);
  if (globeHidden) { clipPosition = vec4<f32>(2.0, 2.0, 2.0, 1.0); }
  let pickingColor = encodePickingColor(instanceIndex + arrowStyle.rowIndexOffset);
  geometry.worldPosition = vec3<f32>(targetPositions, targetAltitude);
  geometry.pickingColor = pickingColor;
  var output: VertexOutput;
  output.position = clipPosition;
  output.color = select(arrowStyle.color, colors, arrowStyle.useColors != 0u);
  output.pickingColor = pickingColor;
  output.selected = selectionMask;
  output.shadowPosition0 = shadow_getVertexPosition(0u, tipCommon);
  output.shadowPosition1 = shadow_getVertexPosition(1u, tipCommon);
  return output;
}
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (input.selected == 0u) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let color = gpuHighlight(input.color, input.pickingColor);
  let alpha = color.a * layer.opacity;
  return shadow_filterPremultiplied(vec4<f32>(color.rgb, alpha), input.position.z, input.shadowPosition0, input.shadowPosition1);
}
`;

/** Chunk-preserving GPUVector target arrowhead layer. */
export class GPUArrowHeadLayer extends Layer<GPUArrowHeadLayerProps> {
  static override layerName = 'GPUArrowHeadLayer';
  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPUArrowHeadLayer requires WebGPU');
    validateGPUVectorPickingCapacity(
      this.props.getSourcePosition.length,
      Boolean(this.props.pickable)
    );
    const selectionMask = this.props.selectionMask;
    const colors = isGPUVector(this.props.getColor) ? this.props.getColor : undefined;
    getGPUVectorLayerBatches(
      this.id,
      {
        sourcePositions: this.props.getSourcePosition,
        targetPositions: this.props.getTargetPosition,
        sourcePositionsLow: this.props.getSourcePositionLow,
        targetPositionsLow: this.props.getTargetPositionLow,
        colors,
        selectionMask
      },
      {
        sourcePositions: GPU_ENDPOINT_POSITION_FORMATS,
        targetPositions: GPU_ENDPOINT_POSITION_FORMATS,
        sourcePositionsLow: ['float32x2'],
        targetPositionsLow: ['float32x2'],
        colors: ['unorm8x4'],
        selectionMask: ['uint32']
      }
    );
    if (this.props.getSourcePosition.data.length === 0) {
      this.setState({
        model: null,
        styleBuffer: null,
        defaults: []
      } satisfies GPUArrowHeadLayerState);
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
    const defaultSelection = device.createBuffer({data: new Uint32Array([1])});
    const defaultPositionLow = device.createBuffer({data: new Float32Array([0, 0])});
    const sourcePositionsLow = this.props.getSourcePositionLow;
    const targetPositionsLow = this.props.getTargetPositionLow;
    const model = new GPUVectorModel(device, {
      ...this.getShaders({
        modules: this.props.brushingEnabled
          ? [project32, picking, gpuBrushing, wgslShadow]
          : [project32, picking, wgslShadow],
        // Native heads brush with 'source_target': either endpoint inside the radius shows the row.
        source: applyGPUBrushingToShader(
          applyEndpointAltitudeToShader(
            GPU_ARROW_HEAD_SHADER,
            this.props.getSourcePosition.format,
            this.props.getTargetPosition.format,
            'arrowStyle.elevationScale'
          ),
          Boolean(this.props.brushingEnabled),
          'brushing_arePointsInRange(sourceLongitudeLatitude, targetLongitudeLatitude)'
        )
      }),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      inferDrawCount: 'instance',
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
        selectionMask: selectionMask ? getGPUVectorBuffer(selectionMask) : defaultSelection
      },
      bufferLayout: [
        makeGPUVectorSelectionLayout(selectionMask),
        makeGPUVectorBufferLayout(this.props.getSourcePosition, 'sourcePositions'),
        makeGPUVectorBufferLayout(this.props.getTargetPosition, 'targetPositions'),
        sourcePositionsLow
          ? makeGPUVectorBufferLayout(sourcePositionsLow, 'sourcePositionsLow')
          : makeConstantPositionLowLayout('sourcePositionsLow'),
        targetPositionsLow
          ? makeGPUVectorBufferLayout(targetPositionsLow, 'targetPositionsLow')
          : makeConstantPositionLowLayout('targetPositionsLow'),
        colors
          ? makeGPUVectorBufferLayout(colors, 'colors')
          : {
              name: 'colors',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'colors', format: 'unorm8x4'}]
            }
      ],
      bindings: {arrowStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH)}
    });
    model.userData['boundInputs'] = [
      this.props.getSourcePosition,
      this.props.getTargetPosition,
      colors,
      selectionMask,
      Boolean(this.props.brushingEnabled),
      this.props.getSourcePositionLow,
      this.props.getTargetPositionLow
    ];
    this.setState({
      model,
      styleBuffer,
      defaults: [defaultColor, defaultSelection, defaultPositionLow]
    } satisfies GPUArrowHeadLayerState);
  }

  override getModels(): GPUVectorModel[] {
    const model = (this.state as GPUArrowHeadLayerState).model;
    return model ? [model] : [];
  }
  override updateState({props}: UpdateParameters<this>): void {
    const boundInputs = ((this.state as GPUArrowHeadLayerState).model?.userData['boundInputs'] ??
      []) as unknown[];
    const colors = isGPUVector(props.getColor) ? props.getColor : undefined;
    if (
      props.getSourcePosition !== boundInputs[0] ||
      props.getTargetPosition !== boundInputs[1] ||
      colors !== boundInputs[2] ||
      props.selectionMask !== boundInputs[3] ||
      Boolean(props.brushingEnabled) !== boundInputs[4] ||
      props.getSourcePositionLow !== boundInputs[5] ||
      props.getTargetPositionLow !== boundInputs[6]
    ) {
      this.destroyResources();
      this.initializeState(this.context);
    }
  }
  override draw({renderPass}: {renderPass: RenderPass}): void {
    const selectionMask = this.props.selectionMask;
    const {model, styleBuffer: previousStyleBuffer} = this.state as GPUArrowHeadLayerState;
    if (!model || !previousStyleBuffer) return;
    ensureWGSLShadowInputs([model], model.device);
    const styleBuffer = resizeGPUVectorStyleBuffer(
      previousStyleBuffer,
      STYLE_BYTE_LENGTH,
      this.props.getSourcePosition.data.length
    );
    if (styleBuffer !== previousStyleBuffer) {
      this.setState({styleBuffer});
      model.setBindings({
        arrowStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH)
      });
    }
    const colors = isGPUVector(this.props.getColor) ? this.props.getColor : undefined;
    const [red, green, blue, alpha = 255] = isColor(this.props.getColor)
      ? this.props.getColor
      : [0, 0, 0, 255];
    const bytes = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(bytes);
    const uints = new Uint32Array(bytes);
    floats.set([red / 255, green / 255, blue / 255, alpha / 255, this.props.sizePixels ?? 16]);
    uints[5] = SHAPE_INDICES[this.props.shape ?? 'chevron'];
    uints[6] = this.props.mode === 'arc' ? 1 : 0;
    floats[8] = this.props.arcHeight ?? 1;
    uints[9] = colors ? 1 : 0;
    floats[10] = ARC_SEGMENT_COUNT;
    floats[11] = this.props.elevationScale ?? 1;
    setGPUBrushingProps(model, this.context, this.props);
    model.drawBatches(renderPass, {
      vectors: {
        sourcePositions: this.props.getSourcePosition,
        targetPositions: this.props.getTargetPosition,
        sourcePositionsLow: this.props.getSourcePositionLow,
        targetPositionsLow: this.props.getTargetPositionLow,
        colors,
        selectionMask
      },
      onBatch: batch => {
        uints[7] = batch.rowIndexOffset;
        const binding = getGPUVectorStyleBufferBinding(
          styleBuffer,
          STYLE_BYTE_LENGTH,
          batch.batchIndex
        );
        styleBuffer.write(new Uint8Array(bytes), binding.offset);
        model.setBindings({arrowStyle: binding});
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
    const state = this.state as GPUArrowHeadLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.defaults.forEach(buffer => buffer.destroy());
    this.setState({model: null, styleBuffer: null, defaults: []});
  }
}

function makeConstantPositionLowLayout(name: string) {
  return {
    name,
    byteStride: 0,
    stepMode: 'instance' as const,
    attributes: [{attribute: name, format: 'float32x2' as const}]
  };
}

function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
function isColor(value: unknown): value is Color {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}
