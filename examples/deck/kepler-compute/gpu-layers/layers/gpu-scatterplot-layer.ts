// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  UNIT,
  picking,
  project32,
  type Color,
  type GetPickingInfoParams,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {assert, type Buffer, type RenderPass} from '@luma.gl/core';
import {GPUVectorModel} from '@luma.gl/engine';
import type {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
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
  makeGPUDataBufferLayout,
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
import {
  WGSL_SHADOW_VARYING_FIELDS,
  ensureWGSLShadowInputs,
  wgslShadow
} from '../effects/wgsl-shadow';
import {
  GPU_HIGHLIGHT_RANGE_WGSL,
  gpuHighlightRange,
  setGPUHighlightRangeProps,
  type GPUHighlightRangeProps
} from './gpu-highlight';

/** GPUVector-native scatterplot props. All input vectors are borrowed. */
export type GPUScatterplotLayerProps = Omit<LayerProps, 'data'> &
  GPUVectorLayerSelectionProps &
  GPUHighlightRangeProps &
  GPUBrushingProps & {
    /** XY uses altitude zero; XYZ preserves the caller's coordinate-system altitude. */
    getPosition: GPUVector<'float32x2' | 'float32x3'>;
    /**
     * Optional fp64 low part: `getPosition + getPositionLow` reconstructs the Float64 position, which
     * Deck's `project_position_to_clipspace` adds after subtracting the viewport origin in Float32.
     * Must have the component count of `getPosition` and share its chunk boundaries. Defaults to zero.
     */
    getPositionLow?: GPUVector<'float32x2' | 'float32x3'>;
    getRadius?: number | GPUVector<'float32'>;
    getFillColor?: Color | GPUVector<'unorm8x4'>;
    getLineColor?: Color | GPUVector<'unorm8x4'>;
    getLineWidth?: number | GPUVector<'float32'>;
    /** Center the outline on the radius. Defaults to false. */
    stroked?: boolean;
    /** Defaults to true. A hollow circle's interior is also excluded from picking. */
    filled?: boolean;
    lineWidthScale?: number;
    lineWidthMinPixels?: number;
    lineWidthMaxPixels?: number;
    /** Defaults to meters, matching Deck's outline contract. */
    lineWidthUnits?: 'pixels' | 'meters' | 'common';
    antialiasing?: boolean;
    radiusScale?: number;
    radiusMinPixels?: number;
    radiusMaxPixels?: number;
    /** Defaults to pixels for compatibility with the original resident layer. */
    radiusUnits?: 'pixels' | 'meters' | 'common';
    /** Defaults to true; false draws circles in the projected ground plane. */
    billboard?: boolean;
  };

type GPUScatterplotLayerState = {
  model: GPUVectorModel | null;
  styleBuffer: Buffer | null;
  /** Inputs seen while the position vector was empty, so updates do not re-initialize. */
  emptyBoundInputs?: unknown[];
};

/** Input identities baked into the model; a change rebuilds it. */
function getScatterplotBoundInputs(
  props: GPUScatterplotLayerProps,
  iconGeometry: GPUData<'float32x3'> | undefined
): unknown[] {
  return [
    props.getPosition,
    isGPUVector(props.getRadius) ? props.getRadius : undefined,
    isGPUVector(props.getFillColor) ? props.getFillColor : undefined,
    props.selectionMask,
    isGPUVector(props.getLineWidth) ? props.getLineWidth : undefined,
    isGPUVector(props.getLineColor) ? props.getLineColor : undefined,
    iconGeometry,
    Boolean(props.brushingEnabled),
    props.getPositionLow
  ];
}

const STYLE_BYTE_LENGTH = 128;

function getScatterplotShader(
  positionFormat: 'float32x2' | 'float32x3',
  mesh = false,
  brushing = false
): string {
  return applyGPUBrushingToShader(
    getScatterplotShaderSource(positionFormat, mesh),
    brushing,
    'brushing_isPointInRange(position.xy)'
  );
}

function getScatterplotShaderSource(
  positionFormat: 'float32x2' | 'float32x3',
  mesh: boolean
): string {
  const positionComponents = positionFormat === 'float32x3' ? 3 : 2;
  return /* wgsl */ `
${GPU_HIGHLIGHT_RANGE_WGSL}
${GPU_GLOBE_WGSL}
struct ScatterStyle {
  color: vec4<f32>,
  radius: f32,
  radiusScale: f32,
  radiusMinPixels: f32,
  radiusMaxPixels: f32,
  useRadii: u32,
  useColors: u32,
  rowIndexOffset: u32,
  radiusUnits: i32,
  billboard: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
  lineColor: vec4<f32>,
  lineWidth: f32,
  lineWidthScale: f32,
  lineWidthMinPixels: f32,
  lineWidthMaxPixels: f32,
  useLineWidths: u32,
  useLineColors: u32,
  stroked: u32,
  filled: u32,
  lineWidthUnits: i32,
  antialiasing: u32,
  _padding3: u32,
  _padding4: u32,
};
@group(0) @binding(auto) var<uniform> scatterStyle: ScatterStyle;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) @interpolate(flat) pickingColor: vec3<f32>,
  @location(3) @interpolate(flat) selected: u32,
  @location(4) lineColor: vec4<f32>,
  @location(5) @interpolate(flat) outerRadiusPixels: f32,
  @location(6) @interpolate(flat) innerRadiusPixels: f32,
  ${WGSL_SHADOW_VARYING_FIELDS(7)}
};

fn encodePickingColor(rowIndex: u32) -> vec3<f32> {
  let value = rowIndex + 1u;
  return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0;
}

fn getCorner(vertexIndex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}

@vertex fn vertexMain(
  @location(0) positions: vec${positionComponents}<f32>,
  @location(1) radii: f32,
  @location(2) fillColors: vec4<f32>,
  @location(3) selectionMask: u32,
  @location(4) lineWidths: f32,
  @location(5) lineColors: vec4<f32>,
  ${mesh ? '@location(6) meshPositions: vec3<f32>,' : ''}
  @location(7) positionsLow: vec${positionComponents}<f32>,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let corner = ${mesh ? 'meshPositions.xy' : 'getCorner(vertexIndex)'};
  let pickingColor = encodePickingColor(instanceIndex + scatterStyle.rowIndexOffset);
  let position = ${positionComponents === 3 ? 'positions' : 'vec3<f32>(positions, 0.0)'};
  let positionLow = ${positionComponents === 3 ? 'positionsLow' : 'vec3<f32>(positionsLow, 0.0)'};
  geometry.worldPosition = position;
  geometry.pickingColor = pickingColor;
  let radius = clamp(project_unit_size_to_pixel(select(scatterStyle.radius, radii, scatterStyle.useRadii != 0u) * scatterStyle.radiusScale, scatterStyle.radiusUnits), scatterStyle.radiusMinPixels, scatterStyle.radiusMaxPixels);
  let lineWidthPixels = clamp(project_unit_size_to_pixel(select(scatterStyle.lineWidth, lineWidths, scatterStyle.useLineWidths != 0u) * scatterStyle.lineWidthScale, scatterStyle.lineWidthUnits), scatterStyle.lineWidthMinPixels, scatterStyle.lineWidthMaxPixels);
  let strokeWidth = select(0.0, max(0.0, lineWidthPixels), scatterStyle.stroked != 0u);
  let outerRadiusPixels = max(0.0, radius) + strokeWidth * 0.5;
  let paddedRadius = outerRadiusPixels + select(0.0, 0.5, scatterStyle.antialiasing != 0u);
  var clipPosition: vec4<f32>;
  var commonPosition: vec4<f32>;
  if (scatterStyle.billboard != 0u) {
    let projected = project_position_to_clipspace_and_commonspace(position, positionLow, vec3<f32>(0.0));
    commonPosition = projected.commonPosition;
    // Like Deck, the shadow map swap happens on the center before the pixel offset is added.
    clipPosition = shadow_setVertexPosition(projected.commonPosition, project_globe_billboard_clipspace(projected.clipPosition, projected.commonPosition.xyz));
    // Preserve the original screen-facing pixel mode. Meter/common radii follow Deck projection.
    // GlobeView follows Deck's billboard offset (focal-distance scaled), which varies in w across the sphere.
    let clipScale = select(1.0, clipPosition.w, scatterStyle.radiusUnits == UNIT_PIXELS && !gpuGlobeIsActive());
    clipPosition = vec4<f32>(clipPosition.xy + project_pixel_size_to_clipspace(corner * paddedRadius) * clipScale, clipPosition.z, clipPosition.w);
  } else {
    let projected = project_position_to_clipspace_and_commonspace(position, positionLow, ${mesh ? 'meshPositions * project_pixel_size_float(paddedRadius)' : 'vec3<f32>(corner * project_pixel_size_float(paddedRadius), 0.0)'});
    commonPosition = projected.commonPosition;
    clipPosition = shadow_setVertexPosition(projected.commonPosition, projected.clipPosition);
  }
  // GlobeView has no occluder: cull points on the far hemisphere.
  if (gpuGlobeIsHidden(position)) { clipPosition = vec4<f32>(2.0, 2.0, 2.0, 1.0); }
  var output: VertexOutput;
  output.position = clipPosition;
  output.corner = corner * paddedRadius;
  output.color = select(scatterStyle.color, fillColors, scatterStyle.useColors != 0u);
  output.pickingColor = pickingColor;
  output.selected = selectionMask;
  output.lineColor = select(scatterStyle.lineColor, lineColors, scatterStyle.useLineColors != 0u);
  output.outerRadiusPixels = outerRadiusPixels;
  output.innerRadiusPixels = outerRadiusPixels - strokeWidth;
  output.shadowPosition0 = shadow_getVertexPosition(0u, commonPosition);
  output.shadowPosition1 = shadow_getVertexPosition(1u, commonPosition);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (input.selected == 0u || input.outerRadiusPixels <= 0.0 || (scatterStyle.filled == 0u && scatterStyle.stroked == 0u)) { discard; }
  ${
    mesh
      ? 'var color = input.color;\n  let outerCoverage = 1.0;'
      : `  let distancePixels = length(input.corner);
  let outerCoverage = select(step(distancePixels, input.outerRadiusPixels), 1.0 - smoothstep(input.outerRadiusPixels - 0.5, input.outerRadiusPixels + 0.5, distancePixels), scatterStyle.antialiasing != 0u);
  if (outerCoverage == 0.0) { discard; }
  var color = input.color;
  if (scatterStyle.stroked != 0u) {
    let lineCoverage = select(step(input.innerRadiusPixels, distancePixels), smoothstep(input.innerRadiusPixels - 0.5, input.innerRadiusPixels + 0.5, distancePixels), scatterStyle.antialiasing != 0u);
    if (scatterStyle.filled != 0u) {
      color = mix(input.color, input.lineColor, lineCoverage);
    } else {
      if (lineCoverage == 0.0) { discard; }
      color = vec4<f32>(input.lineColor.rgb, input.lineColor.a * lineCoverage);
    }
  }
`
  }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  color = gpuHighlightRows(color, input.pickingColor);
  let alpha = color.a * layer.opacity * outerCoverage;
  // Deck uses premultiplied alpha blending, including antialiased outline coverage.
  return shadow_filterPremultiplied(vec4<f32>(color.rgb, alpha), input.position.z, input.shadowPosition0, input.shadowPosition1);
}`;
}

/** Chunk-preserving GPUVector scatterplot layer. */
export class GPUScatterplotLayer<
  PropsT extends GPUScatterplotLayerProps = GPUScatterplotLayerProps
> extends Layer<PropsT> {
  static override layerName = 'GPUScatterplotLayer';

  /** Optional borrowed triangle geometry for mesh specializations. Circle behavior is unchanged. */
  protected getIconGeometry(): GPUData<'float32x3'> | undefined {
    return undefined;
  }

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPUScatterplotLayer requires WebGPU');
    validateGPUVectorPickingCapacity(this.props.getPosition.length, Boolean(this.props.pickable));
    const selectionMask = this.props.selectionMask;
    const radii = isGPUVector(this.props.getRadius) ? this.props.getRadius : undefined;
    const fillColors = isGPUVector(this.props.getFillColor) ? this.props.getFillColor : undefined;
    const lineColors = isGPUVector(this.props.getLineColor) ? this.props.getLineColor : undefined;
    const lineWidths = isGPUVector(this.props.getLineWidth) ? this.props.getLineWidth : undefined;
    getGPUVectorLayerBatches(
      this.id,
      {
        positions: this.props.getPosition,
        positionsLow: this.props.getPositionLow,
        radii,
        fillColors,
        selectionMask,
        lineWidths,
        lineColors
      },
      {
        positions: ['float32x2', 'float32x3'],
        positionsLow: ['float32x2', 'float32x3'],
        radii: ['float32'],
        fillColors: ['unorm8x4'],
        lineColors: ['unorm8x4'],
        lineWidths: ['float32'],
        selectionMask: ['uint32']
      }
    );
    const iconGeometry = this.getIconGeometry();
    const positionFormat = this.props.getPosition.format;
    // Batch validation above requires an explicit XY or XYZ format.
    assert(positionFormat);
    if (this.props.getPosition.data.length === 0) {
      this.setState({
        model: null,
        styleBuffer: null,
        emptyBoundInputs: getScatterplotBoundInputs(this.props, iconGeometry)
      } satisfies GPUScatterplotLayerState);
      return;
    }
    const styleBuffer = createGPUVectorStyleBuffer(
      device,
      `${this.id}-style`,
      STYLE_BYTE_LENGTH,
      this.props.getPosition.data.length
    );
    const defaultRadius = device.createBuffer({
      id: `${this.id}-default-radius`,
      data: new Float32Array([1])
    });
    const defaultColor = device.createBuffer({
      id: `${this.id}-default-color`,
      data: new Uint8Array([0, 0, 0, 255])
    });
    const defaultSelection = device.createBuffer({data: new Uint32Array([1])});
    const defaultPositionLow = device.createBuffer({data: new Float32Array([0, 0, 0])});
    const positionsLow = this.props.getPositionLow;
    // The low part must have getPosition's component count; the shader reads both with one type.
    assert(!positionsLow || positionsLow.format === positionFormat);
    const model = new GPUVectorModel(device, {
      shaderAssembler: this.context.shaderAssembler,
      ...this.getShaders({
        modules: this.props.brushingEnabled
          ? [project32, picking, gpuBrushing, gpuHighlightRange, wgslShadow]
          : [project32, picking, gpuHighlightRange, wgslShadow],
        source: getScatterplotShader(
          positionFormat,
          Boolean(iconGeometry),
          Boolean(this.props.brushingEnabled)
        )
      }),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      inferDrawCount: 'instance',
      isInstanced: true,
      vertexCount: iconGeometry?.length ?? 6,
      instanceCount: 0,
      attributes: {
        ...(iconGeometry ? {meshPositions: iconGeometry.buffer} : {}),
        positions: getGPUVectorBuffer(this.props.getPosition),
        positionsLow: positionsLow ? getGPUVectorBuffer(positionsLow) : defaultPositionLow,
        radii: radii ? getGPUVectorBuffer(radii) : defaultRadius,
        fillColors: fillColors ? getGPUVectorBuffer(fillColors) : defaultColor,
        lineColors: lineColors ? getGPUVectorBuffer(lineColors) : defaultColor,
        lineWidths: lineWidths ? getGPUVectorBuffer(lineWidths) : defaultRadius,
        selectionMask: selectionMask ? getGPUVectorBuffer(selectionMask) : defaultSelection
      },
      bufferLayout: [
        ...(iconGeometry
          ? [
              {
                ...makeGPUDataBufferLayout(iconGeometry, 'meshPositions'),
                stepMode: 'vertex' as const,
                attributes: [{attribute: 'meshPositions', format: 'float32x3' as const}]
              }
            ]
          : []),
        makeGPUVectorSelectionLayout(selectionMask),
        makeGPUVectorBufferLayout(this.props.getPosition, 'positions'),
        positionsLow
          ? makeGPUVectorBufferLayout(positionsLow, 'positionsLow')
          : {
              name: 'positionsLow',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'positionsLow', format: positionFormat}]
            },
        radii
          ? makeGPUVectorBufferLayout(radii, 'radii')
          : {
              name: 'radii',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'radii', format: 'float32'}]
            },
        lineWidths
          ? makeGPUVectorBufferLayout(lineWidths, 'lineWidths')
          : {
              name: 'lineWidths',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'lineWidths', format: 'float32'}]
            },
        lineColors
          ? makeGPUVectorBufferLayout(lineColors, 'lineColors')
          : {
              name: 'lineColors',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'lineColors', format: 'unorm8x4'}]
            },
        fillColors
          ? makeGPUVectorBufferLayout(fillColors, 'fillColors')
          : {
              name: 'fillColors',
              byteStride: 0,
              stepMode: 'instance',
              attributes: [{attribute: 'fillColors', format: 'unorm8x4'}]
            }
      ],
      bindings: {scatterStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH)}
    });
    if (iconGeometry) {
      model.setAttributes(
        {meshPositions: iconGeometry.buffer},
        {byteOffsets: {meshPositions: iconGeometry.byteOffset}}
      );
    }
    model.userData['ownedDefaultBuffers'] = [
      defaultRadius,
      defaultColor,
      defaultSelection,
      defaultPositionLow
    ];
    model.userData['boundInputs'] = getScatterplotBoundInputs(this.props, iconGeometry);
    this.setState({model, styleBuffer} satisfies GPUScatterplotLayerState);
  }

  override getModels(): GPUVectorModel[] {
    const model = (this.state as GPUScatterplotLayerState).model;
    return model ? [model] : [];
  }

  override updateState({props}: UpdateParameters<this>): void {
    const {model, emptyBoundInputs} = this.state as GPUScatterplotLayerState;
    // An empty stream re-initializes only once chunks arrive or an input changes.
    const boundInputs = (model?.userData['boundInputs'] ?? emptyBoundInputs ?? []) as unknown[];
    const nextInputs = getScatterplotBoundInputs(props, this.getIconGeometry());
    if (
      nextInputs.some((input, index) => input !== boundInputs[index]) ||
      (!model && props.getPosition.data.length > 0)
    ) {
      this.destroyResources();
      this.initializeState(this.context);
    }
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    validateGPUVectorPickingCapacity(this.props.getPosition.length, Boolean(this.props.pickable));
    const selectionMask = this.props.selectionMask;
    const {model} = this.state as GPUScatterplotLayerState;
    const {styleBuffer: previousStyleBuffer} = this.state as GPUScatterplotLayerState;
    if (!model || !previousStyleBuffer) return;
    // Icon meshes are two-sided: Deck's globe projection reverses their screen winding, and Deck
    // reapplies its own cull mode before every draw.
    if (this.getIconGeometry() && model.parameters.cullMode !== 'none')
      model.setParameters({...model.parameters, cullMode: 'none'});
    if (model.shaderInputs) ensureWGSLShadowInputs([model], this.context.device);
    setGPUHighlightRangeProps([model], this.props);
    const styleBuffer = resizeGPUVectorStyleBuffer(
      previousStyleBuffer,
      STYLE_BYTE_LENGTH,
      this.props.getPosition.data.length
    );
    if (styleBuffer !== previousStyleBuffer) {
      this.setState({styleBuffer});
      model.setBindings({
        scatterStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH)
      });
    }
    const radii = isGPUVector(this.props.getRadius) ? this.props.getRadius : undefined;
    const fillColors = isGPUVector(this.props.getFillColor) ? this.props.getFillColor : undefined;
    const lineColors = isGPUVector(this.props.getLineColor) ? this.props.getLineColor : undefined;
    const lineWidths = isGPUVector(this.props.getLineWidth) ? this.props.getLineWidth : undefined;
    const [red, green, blue, alpha = 255] = isColor(this.props.getFillColor)
      ? this.props.getFillColor
      : [0, 0, 0, 255];
    const [lineRed, lineGreen, lineBlue, lineAlpha = 255] = isColor(this.props.getLineColor)
      ? this.props.getLineColor
      : [0, 0, 0, 255];
    const bytes = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(bytes);
    const uints = new Uint32Array(bytes);
    floats.set([
      red / 255,
      green / 255,
      blue / 255,
      alpha / 255,
      typeof this.props.getRadius === 'number' ? this.props.getRadius : 1,
      this.props.radiusScale ?? 1,
      this.props.radiusMinPixels ?? 0,
      this.props.radiusMaxPixels ?? 1e9
    ]);
    uints.set(
      [
        radii ? 1 : 0,
        fillColors ? 1 : 0,
        0,
        UNIT[this.props.radiusUnits ?? 'pixels'],
        this.props.billboard === false ? 0 : 1
      ],
      8
    );
    floats.set(
      [
        lineRed / 255,
        lineGreen / 255,
        lineBlue / 255,
        lineAlpha / 255,
        typeof this.props.getLineWidth === 'number' ? this.props.getLineWidth : 1,
        this.props.lineWidthScale ?? 1,
        this.props.lineWidthMinPixels ?? 0,
        this.props.lineWidthMaxPixels ?? 1e9
      ],
      16
    );
    uints.set(
      [
        lineWidths ? 1 : 0,
        lineColors ? 1 : 0,
        this.props.stroked ? 1 : 0,
        this.props.filled === false ? 0 : 1,
        UNIT[this.props.lineWidthUnits ?? 'meters'],
        this.props.antialiasing === false ? 0 : 1
      ],
      24
    );
    setGPUBrushingProps(model, this.context, this.props);
    model.drawBatches(renderPass, {
      vectors: {
        positions: this.props.getPosition,
        positionsLow: this.props.getPositionLow,
        radii,
        fillColors,
        selectionMask,
        lineWidths,
        lineColors
      },
      onBatch: batch => {
        uints[10] = batch.rowIndexOffset;
        const binding = getGPUVectorStyleBufferBinding(
          styleBuffer,
          STYLE_BYTE_LENGTH,
          batch.batchIndex
        );
        styleBuffer.write(new Uint8Array(bytes), binding.offset);
        model.setBindings({scatterStyle: binding});
      }
    });
  }

  override getPickingInfo(params: GetPickingInfoParams): PickingInfo {
    const info = params.info as GPUVectorLayerPickingInfo;
    info.gpuVector = getGPUVectorPickingProvenance(this.props.getPosition, info.index);
    return info;
  }

  override finalizeState(context: LayerContext): void {
    this.destroyResources();
    super.finalizeState(context);
  }

  private destroyResources(): void {
    const state = this.state as GPUScatterplotLayerState;
    const owned = state.model?.userData['ownedDefaultBuffers'] as Buffer[] | undefined;
    state.model?.destroy();
    owned?.forEach(buffer => buffer.destroy());
    state.styleBuffer?.destroy();
    this.setState({model: null, styleBuffer: null});
  }
}

function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
function isColor(value: unknown): value is Color {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}
