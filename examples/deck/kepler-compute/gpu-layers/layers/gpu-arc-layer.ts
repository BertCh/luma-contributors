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
import {GPU_GLOBE_WGSL} from './gpu-globe';
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

/** GPUVector-native curved-arc props. Input vectors are borrowed. */
export type GPUArcLayerProps = Omit<LayerProps, 'data'> &
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
    getSourceColor?: Color | GPUVector<'unorm8x4'>;
    getTargetColor?: Color | GPUVector<'unorm8x4'>;
    getWidth?: number | GPUVector<'float32'>;
    getHeight?: number | GPUVector<'float32'>;
    widthScale?: number;
    widthMinPixels?: number;
    widthMaxPixels?: number;
    numSegments?: number;
    /** Default preserves absolute-height parabola. Deck mode uses a projected circular profile
     * with getHeight as a distance multiplier and numSegments as the number of sampled points. */
    arcMode?: 'parabola' | 'deck';
  };

type GPUArcLayerState = {
  model: GPUVectorModel | null;
  styleBuffer: Buffer | null;
  defaults: Buffer[];
};

function getArcShader(
  brushing: boolean,
  sourcePosition: GPUEndpointPositionVector,
  targetPosition: GPUEndpointPositionVector
): string {
  return applyGPUBrushingToShader(
    applyEndpointAltitudeToShader(GPU_ARC_SHADER, sourcePosition.format, targetPosition.format),
    brushing,
    'brushing_arePointsInRange(sourceLongitudeLatitude, targetLongitudeLatitude)'
  );
}

const GPU_ARC_SHADER = /* wgsl */ `
${GPU_HIGHLIGHT_WGSL}
${GPU_GLOBE_WGSL}
struct ArcStyle {
  sourceColor: vec4<f32>,
  targetColor: vec4<f32>,
  width: f32,
  height: f32,
  widthScale: f32,
  widthMinPixels: f32,
  widthMaxPixels: f32,
  numSegments: u32,
  useSourceColors: u32,
  useTargetColors: u32,
  useWidths: u32,
  useHeights: u32,
  rowIndexOffset: u32,
  arcMode: u32,
};
@group(0) @binding(auto) var<uniform> arcStyle: ArcStyle;

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

fn getCorner(vertexIndex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex % 6u];
}

fn getArcPosition(sourcePosition: vec3<f32>, targetPosition: vec3<f32>, progress: f32, height: f32) -> vec3<f32> {
  let linear = mix(sourcePosition, targetPosition, progress);
  return vec3<f32>(linear.xy, linear.z + 4.0 * progress * (1.0 - progress) * height);
}

fn getProjectedArcPosition(sourcePosition: vec3<f32>, targetPosition: vec3<f32>, progress: f32, height: f32) -> vec3<f32> {
  let linear = mix(sourcePosition, targetPosition, progress);
  let elevation = sqrt(max(progress * (1.0 - progress), 0.0)) * length(targetPosition.xy - sourcePosition.xy) * height;
  return vec3<f32>(linear.xy, linear.z + elevation);
}

@vertex fn vertexMain(
  @location(0) sourcePositions: vec2<f32>,
  @location(1) targetPositions: vec2<f32>,
  @location(2) sourceColors: vec4<f32>,
  @location(3) targetColors: vec4<f32>,
  @location(4) widths: f32,
  @location(5) heights: f32,
  @location(6) selectionMask: u32,
  @location(7) sourcePositionsLow: vec2<f32>,
  @location(8) targetPositionsLow: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let segmentIndex = vertexIndex / 6u;
  let corner = getCorner(vertexIndex);
  var startProgress = f32(segmentIndex) / f32(arcStyle.numSegments);
  var endProgress = f32(segmentIndex + 1u) / f32(arcStyle.numSegments);
  let height = select(arcStyle.height, heights, arcStyle.useHeights != 0u);
  var startWorld = getArcPosition(vec3<f32>(sourcePositions, sourceAltitude), vec3<f32>(targetPositions, targetAltitude), startProgress, height);
  var endWorld = getArcPosition(vec3<f32>(sourcePositions, sourceAltitude), vec3<f32>(targetPositions, targetAltitude), endProgress, height);
  var startClip: vec4<f32>;
  var endClip: vec4<f32>;
  var startCommonPosition: vec4<f32>;
  var endCommonPosition: vec4<f32>;
  var globeHidden = false;
  if (arcStyle.arcMode != 0u) {
    startProgress = smoothstep(0.0, 1.0, startProgress);
    endProgress = smoothstep(0.0, 1.0, endProgress);
    let sourceCommon = project_position_vec3_f64(vec3<f32>(sourcePositions, sourceAltitude), vec3<f32>(sourcePositionsLow, 0.0));
    let targetCommon = project_position_vec3_f64(vec3<f32>(targetPositions, targetAltitude), vec3<f32>(targetPositionsLow, 0.0));
    let startCommon = getProjectedArcPosition(sourceCommon, targetCommon, startProgress, height);
    let endCommon = getProjectedArcPosition(sourceCommon, targetCommon, endProgress, height);
    startCommonPosition = vec4<f32>(startCommon, 1.0);
    endCommonPosition = vec4<f32>(endCommon, 1.0);
    startClip = project_common_position_to_clipspace(startCommonPosition);
    endClip = project_common_position_to_clipspace(endCommonPosition);
    startWorld = vec3<f32>(mix(sourcePositions, targetPositions, startProgress), mix(sourceAltitude, targetAltitude, startProgress));
    endWorld = vec3<f32>(mix(sourcePositions, targetPositions, endProgress), mix(sourceAltitude, targetAltitude, endProgress));
    if (gpuGlobeUsesGreatCircle()) {
      // GlobeView arcs follow the great circle; segments across the antimeridian or horizon are culled.
      let angularDistance = gpuGlobeAngularDistance(sourcePositions, targetPositions);
      // Great-circle height is relative to the end-point altitude ramp so elevated endpoints stay attached.
      var startGlobe = gpuGlobeInterpolate(sourcePositions, targetPositions, angularDistance, startProgress, height);
      var endGlobe = gpuGlobeInterpolate(sourcePositions, targetPositions, angularDistance, endProgress, height);
      startGlobe.z += mix(sourceAltitude, targetAltitude, startProgress);
      endGlobe.z += mix(sourceAltitude, targetAltitude, endProgress);
      globeHidden = abs(startGlobe.x - endGlobe.x) > 180.0 || gpuGlobeIsLngLatHidden(startGlobe) || gpuGlobeIsLngLatHidden(endGlobe);
      let startGlobeProjected = project_position_to_clipspace_and_commonspace(startGlobe, vec3<f32>(0.0), vec3<f32>(0.0));
      let endGlobeProjected = project_position_to_clipspace_and_commonspace(endGlobe, vec3<f32>(0.0), vec3<f32>(0.0));
      startClip = startGlobeProjected.clipPosition;
      endClip = endGlobeProjected.clipPosition;
      startCommonPosition = startGlobeProjected.commonPosition;
      endCommonPosition = endGlobeProjected.commonPosition;
      startWorld = startGlobe;
      endWorld = endGlobe;
    } else if (gpuGlobeIsActive()) {
      // Meter offsets and geocentric data stay in projected common space; hide by horizon.
      globeHidden = gpuGlobeIsCommonHidden(startCommon) || gpuGlobeIsCommonHidden(endCommon);
    }
  } else {
    // The low parts follow the linear interpolation; the parabola only adds altitude.
    let startProjected = project_position_to_clipspace_and_commonspace(startWorld, vec3<f32>(mix(sourcePositionsLow, targetPositionsLow, startProgress), 0.0), vec3<f32>(0.0));
    let endProjected = project_position_to_clipspace_and_commonspace(endWorld, vec3<f32>(mix(sourcePositionsLow, targetPositionsLow, endProgress), 0.0), vec3<f32>(0.0));
    startClip = startProjected.clipPosition;
    endClip = endProjected.clipPosition;
    startCommonPosition = startProjected.commonPosition;
    endCommonPosition = endProjected.commonPosition;
  }
  let delta = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let direction = select(vec2<f32>(1.0, 0.0), normalize(delta), length(delta) > 0.000001);
  let normal = vec2<f32>(-direction.y, direction.x);
  let width = clamp(select(arcStyle.width, widths, arcStyle.useWidths != 0u) * arcStyle.widthScale, arcStyle.widthMinPixels, arcStyle.widthMaxPixels);
  let commonPosition = mix(startCommonPosition, endCommonPosition, corner.x);
  // Like Deck, the shadow map swap happens before the pixel extrusion is added.
  var clipPosition = shadow_setVertexPosition(commonPosition, mix(startClip, endClip, corner.x));
  // Width semantics. 'parabola' (default) keeps the original contract of the first release: a
  // constant on-screen pixel width, so the extrusion is scaled by w. 'deck' matches Deck's ArcLayer,
  // which adds project_pixel_size_to_clipspace(offset) without the w factor (width follows the
  // perspective divide, focal-distance scaled). The two are identical when w is 1 (non-perspective).
  // Deliberate fix shared by both modes: the extrusion direction is computed in pixel space
  // (NDC delta times viewport size, like Deck) instead of raw NDC, which skewed it on non-square viewports.
  let clipScale = select(clipPosition.w, 1.0, arcStyle.arcMode != 0u);
  clipPosition = vec4<f32>(clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * width * 0.5) * clipScale, clipPosition.z, clipPosition.w);
  if (globeHidden) { clipPosition = vec4<f32>(2.0, 2.0, 2.0, 1.0); }
  let progress = mix(startProgress, endProgress, corner.x);
  let pickingColor = encodePickingColor(instanceIndex + arcStyle.rowIndexOffset);
  geometry.worldPosition = mix(startWorld, endWorld, corner.x);
  geometry.pickingColor = pickingColor;
  var output: VertexOutput;
  output.position = clipPosition;
  output.shadowPosition0 = shadow_getVertexPosition(0u, commonPosition);
  output.shadowPosition1 = shadow_getVertexPosition(1u, commonPosition);
  output.color = mix(
    select(arcStyle.sourceColor, sourceColors, arcStyle.useSourceColors != 0u),
    select(arcStyle.targetColor, targetColors, arcStyle.useTargetColors != 0u),
    progress
  );
  output.pickingColor = pickingColor;
  output.selected = selectionMask;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (input.selected == 0u) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let color = gpuHighlight(input.color, input.pickingColor);
  let outputColor = vec4<f32>(color.rgb, color.a * layer.opacity);
  // Deck blends with premultiplied alpha; the filter also premultiplies when no shadow is active.
  return shadow_filterPremultiplied(outputColor, input.position.z, input.shadowPosition0, input.shadowPosition1);
}`;

/** Chunk-preserving GPUVector curved-arc layer. */
export class GPUArcLayer extends Layer<GPUArcLayerProps> {
  static override layerName = 'GPUArcLayer';

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPUArcLayer requires WebGPU');
    validateGPUVectorPickingCapacity(
      this.props.getSourcePosition.length,
      Boolean(this.props.pickable)
    );
    const selectionMask = this.props.selectionMask;
    const sourceColors = isGPUVector(this.props.getSourceColor)
      ? this.props.getSourceColor
      : undefined;
    const targetColors = isGPUVector(this.props.getTargetColor)
      ? this.props.getTargetColor
      : undefined;
    const widths = isGPUVector(this.props.getWidth) ? this.props.getWidth : undefined;
    const heights = isGPUVector(this.props.getHeight) ? this.props.getHeight : undefined;
    getGPUVectorLayerBatches(
      this.id,
      {
        sourcePositions: this.props.getSourcePosition,
        targetPositions: this.props.getTargetPosition,
        sourceColors,
        targetColors,
        widths,
        heights,
        selectionMask,
        sourcePositionsLow: this.props.getSourcePositionLow,
        targetPositionsLow: this.props.getTargetPositionLow
      },
      {
        sourcePositions: GPU_ENDPOINT_POSITION_FORMATS,
        targetPositions: GPU_ENDPOINT_POSITION_FORMATS,
        sourcePositionsLow: ['float32x2'],
        targetPositionsLow: ['float32x2'],
        sourceColors: ['unorm8x4'],
        targetColors: ['unorm8x4'],
        widths: ['float32'],
        heights: ['float32'],
        selectionMask: ['uint32']
      }
    );
    if (this.props.getSourcePosition.data.length === 0) {
      this.setState({model: null, styleBuffer: null, defaults: []} satisfies GPUArcLayerState);
      return;
    }
    const sourcePositionsLow = this.props.getSourcePositionLow;
    const targetPositionsLow = this.props.getTargetPositionLow;
    // Absent optional inputs share one zero-stride buffer: WebGPU allows only 8 vertex buffers.
    const constantInputs = [
      {name: 'sourceColors', format: 'unorm8x4', present: Boolean(sourceColors)},
      {name: 'targetColors', format: 'unorm8x4', present: Boolean(targetColors)},
      {name: 'widths', format: 'float32', present: Boolean(widths)},
      {name: 'heights', format: 'float32', present: Boolean(heights)},
      {name: 'sourcePositionsLow', format: 'float32x2', present: Boolean(sourcePositionsLow)},
      {name: 'targetPositionsLow', format: 'float32x2', present: Boolean(targetPositionsLow)}
    ] as const;
    const vertexBufferCount =
      3 +
      constantInputs.filter(input => input.present).length +
      (constantInputs.some(input => !input.present) ? 1 : 0);
    if (vertexBufferCount > device.limits.maxVertexBuffers) {
      throw new Error(
        `${this.id}: the arc layer needs ${vertexBufferCount} vertex buffers but the device allows ` +
          `${device.limits.maxVertexBuffers}; omit optional per-arc inputs`
      );
    }
    const styleBuffer = createGPUVectorStyleBuffer(
      device,
      `${this.id}-style`,
      80,
      this.props.getSourcePosition.data.length
    );
    const constants = makeConstantInputs(constantInputs.filter(input => !input.present));
    const defaults = [
      device.createBuffer({data: new Uint32Array([1])}),
      ...(constants.bytes.byteLength ? [device.createBuffer({data: constants.bytes})] : [])
    ];
    const defaultSelection = defaults[0]!;
    let model: GPUVectorModel;
    try {
      model = new GPUVectorModel(device, {
        shaderAssembler: this.context.shaderAssembler,
        ...this.getShaders({
          modules: this.props.brushingEnabled
            ? [project32, picking, gpuBrushing, wgslShadow]
            : [project32, picking, wgslShadow],
          source: getArcShader(
            Boolean(this.props.brushingEnabled),
            this.props.getSourcePosition,
            this.props.getTargetPosition
          )
        }),
        id: `${this.id}-model`,
        topology: 'triangle-list',
        inferDrawCount: 'instance',
        isInstanced: true,
        vertexCount: getSegmentCount(this.props) * 6,
        instanceCount: 0,
        attributes: {
          sourcePositions: getGPUVectorBuffer(this.props.getSourcePosition),
          targetPositions: getGPUVectorBuffer(this.props.getTargetPosition),
          ...(sourceColors && {sourceColors: getGPUVectorBuffer(sourceColors)}),
          ...(targetColors && {targetColors: getGPUVectorBuffer(targetColors)}),
          ...(widths && {widths: getGPUVectorBuffer(widths)}),
          ...(heights && {heights: getGPUVectorBuffer(heights)}),
          selectionMask: selectionMask ? getGPUVectorBuffer(selectionMask) : defaultSelection,
          ...(sourcePositionsLow && {sourcePositionsLow: getGPUVectorBuffer(sourcePositionsLow)}),
          ...(targetPositionsLow && {targetPositionsLow: getGPUVectorBuffer(targetPositionsLow)}),
          ...(constants.layout && {constants: defaults[1]!})
        },
        bufferLayout: [
          makeGPUVectorSelectionLayout(selectionMask),
          makeGPUVectorBufferLayout(this.props.getSourcePosition, 'sourcePositions'),
          makeGPUVectorBufferLayout(this.props.getTargetPosition, 'targetPositions'),
          ...(sourceColors ? [makeGPUVectorBufferLayout(sourceColors, 'sourceColors')] : []),
          ...(targetColors ? [makeGPUVectorBufferLayout(targetColors, 'targetColors')] : []),
          ...(widths ? [makeGPUVectorBufferLayout(widths, 'widths')] : []),
          ...(heights ? [makeGPUVectorBufferLayout(heights, 'heights')] : []),
          ...(sourcePositionsLow
            ? [makeGPUVectorBufferLayout(sourcePositionsLow, 'sourcePositionsLow')]
            : []),
          ...(targetPositionsLow
            ? [makeGPUVectorBufferLayout(targetPositionsLow, 'targetPositionsLow')]
            : []),
          ...(constants.layout ? [constants.layout] : [])
        ],
        bindings: {arcStyle: getGPUVectorStyleBufferBinding(styleBuffer, 80)}
      });
    } catch (error) {
      styleBuffer.destroy();
      for (const buffer of defaults) buffer.destroy();
      throw error;
    }
    model.userData['boundInputs'] = [
      this.props.getSourcePosition,
      this.props.getTargetPosition,
      sourceColors,
      targetColors,
      widths,
      heights,
      this.props.numSegments,
      selectionMask,
      this.props.arcMode,
      Boolean(this.props.brushingEnabled),
      sourcePositionsLow,
      targetPositionsLow
    ];
    this.setState({model, styleBuffer, defaults} satisfies GPUArcLayerState);
  }

  override getModels(): GPUVectorModel[] {
    const model = (this.state as GPUArcLayerState).model;
    return model ? [model] : [];
  }

  override updateState({props}: UpdateParameters<this>): void {
    const boundInputs = ((this.state as GPUArcLayerState).model?.userData['boundInputs'] ??
      []) as unknown[];
    const sourceColors = isGPUVector(props.getSourceColor) ? props.getSourceColor : undefined;
    const targetColors = isGPUVector(props.getTargetColor) ? props.getTargetColor : undefined;
    const widths = isGPUVector(props.getWidth) ? props.getWidth : undefined;
    const heights = isGPUVector(props.getHeight) ? props.getHeight : undefined;
    if (
      props.getSourcePosition !== boundInputs[0] ||
      props.getTargetPosition !== boundInputs[1] ||
      sourceColors !== boundInputs[2] ||
      targetColors !== boundInputs[3] ||
      widths !== boundInputs[4] ||
      heights !== boundInputs[5] ||
      props.numSegments !== boundInputs[6] ||
      props.selectionMask !== boundInputs[7] ||
      props.arcMode !== boundInputs[8] ||
      Boolean(props.brushingEnabled) !== boundInputs[9] ||
      props.getSourcePositionLow !== boundInputs[10] ||
      props.getTargetPositionLow !== boundInputs[11]
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
    const {model} = this.state as GPUArcLayerState;
    const {styleBuffer: previousStyleBuffer} = this.state as GPUArcLayerState;
    if (!model || !previousStyleBuffer) return;
    if (model.shaderInputs) ensureWGSLShadowInputs([model], this.context.device);
    const styleBuffer = resizeGPUVectorStyleBuffer(
      previousStyleBuffer,
      80,
      this.props.getSourcePosition.data.length
    );
    if (styleBuffer !== previousStyleBuffer) {
      this.setState({styleBuffer});
      model.setBindings({arcStyle: getGPUVectorStyleBufferBinding(styleBuffer, 80)});
    }
    const sourceColors = isGPUVector(this.props.getSourceColor)
      ? this.props.getSourceColor
      : undefined;
    const targetColors = isGPUVector(this.props.getTargetColor)
      ? this.props.getTargetColor
      : undefined;
    const widths = isGPUVector(this.props.getWidth) ? this.props.getWidth : undefined;
    const heights = isGPUVector(this.props.getHeight) ? this.props.getHeight : undefined;
    const defaultColor: Color = [0, 0, 0, 255];
    const sourceColor = isColor(this.props.getSourceColor)
      ? this.props.getSourceColor
      : defaultColor;
    const targetColor = isColor(this.props.getTargetColor)
      ? this.props.getTargetColor
      : sourceColor;
    const bytes = new ArrayBuffer(80);
    const floats = new Float32Array(bytes);
    const uints = new Uint32Array(bytes);
    floats.set(normalizeColor(sourceColor));
    floats.set(normalizeColor(targetColor), 4);
    floats.set(
      [
        typeof this.props.getWidth === 'number' ? this.props.getWidth : 1,
        typeof this.props.getHeight === 'number' ? this.props.getHeight : 1,
        this.props.widthScale ?? 1,
        this.props.widthMinPixels ?? 0,
        this.props.widthMaxPixels ?? 1e9
      ],
      8
    );
    uints.set(
      [
        getSegmentCount(this.props),
        sourceColors ? 1 : 0,
        targetColors ? 1 : 0,
        widths ? 1 : 0,
        heights ? 1 : 0,
        0,
        this.props.arcMode === 'deck' ? 1 : 0
      ],
      13
    );
    setGPUBrushingProps(model, this.context, this.props);
    model.drawBatches(renderPass, {
      vectors: {
        sourcePositions: this.props.getSourcePosition,
        targetPositions: this.props.getTargetPosition,
        sourceColors,
        targetColors,
        widths,
        heights,
        selectionMask,
        sourcePositionsLow: this.props.getSourcePositionLow,
        targetPositionsLow: this.props.getTargetPositionLow
      },
      onBatch: batch => {
        uints[18] = batch.rowIndexOffset;
        const binding = getGPUVectorStyleBufferBinding(styleBuffer, 80, batch.batchIndex);
        styleBuffer.write(new Uint8Array(bytes), binding.offset);
        model.setBindings({arcStyle: binding});
      }
    });
  }

  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    const result = info as GPUVectorLayerPickingInfo;
    result.gpuVector = getGPUVectorPickingProvenance(this.props.getSourcePosition, result.index);
    return result;
  }

  override finalizeState(context: LayerContext): void {
    this.destroyResources();
    super.finalizeState(context);
  }

  private destroyResources(): void {
    const state = this.state as GPUArcLayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.defaults.forEach(buffer => buffer.destroy());
    this.setState({model: null, styleBuffer: null, defaults: []});
  }
}

function getSegmentCount(props: GPUArcLayerProps): number {
  const count = Math.floor(props.numSegments ?? (props.arcMode === 'deck' ? 50 : 32));
  return Math.max(1, props.arcMode === 'deck' ? count - 1 : count);
}

/** Packs the default value of each absent input into one zero-stride, instance-stepped buffer. */
function makeConstantInputs(
  inputs: readonly {name: string; format: 'unorm8x4' | 'float32' | 'float32x2'}[]
) {
  const defaultBytes = {
    unorm8x4: new Uint8Array([0, 0, 0, 255]),
    float32: new Uint8Array(new Float32Array([1]).buffer),
    float32x2: new Uint8Array(new Float32Array([0, 0]).buffer)
  } as const;
  const attributes: {
    attribute: string;
    format: 'unorm8x4' | 'float32' | 'float32x2';
    byteOffset: number;
  }[] = [];
  const chunks: Uint8Array[] = [];
  let byteOffset = 0;
  for (const input of inputs) {
    attributes.push({attribute: input.name, format: input.format, byteOffset});
    chunks.push(defaultBytes[input.format]);
    byteOffset += defaultBytes[input.format].byteLength;
  }
  const bytes = new Uint8Array(byteOffset);
  let cursor = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return {
    bytes,
    layout: attributes.length
      ? {name: 'constants', byteStride: 0, stepMode: 'instance' as const, attributes}
      : null
  };
}

function normalizeColor(color: Color): [number, number, number, number] {
  return [color[0] / 255, color[1] / 255, color[2] / 255, (color[3] ?? 255) / 255];
}

function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}

function isColor(value: unknown): value is Color {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}
