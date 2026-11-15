// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  UNIT,
  project32,
  picking,
  gouraudMaterial,
  phongMaterial,
  type Color,
  type LayerProps,
  type LayerContext,
  type Material,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {assert, Buffer, type RenderPass} from '@luma.gl/core';
import {GPUVectorModel} from '@luma.gl/engine';
import type {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  gpuBrushing,
  replaceGPUBrushingMarker,
  setGPUBrushingProps,
  type GPUBrushingProps
} from './gpu-brushing';
import {
  createGPUVectorStyleBuffer,
  resizeGPUVectorStyleBuffer,
  getGPUVectorStyleBufferBinding,
  getGPUVectorBuffer,
  getGPUVectorLayerBatches,
  makeGPUVectorBufferLayout,
  makeGPUVectorSelectionLayout,
  validateGPUVectorPickingCapacity,
  getGPUVectorPickingProvenance,
  type GPUVectorLayerPickingInfo,
  type GPUVectorLayerSelectionProps
} from './gpu-vector-layer-utils';
import {GPU_HIGHLIGHT_WGSL} from './gpu-highlight';
import {applyWGSLShadow, ensureWGSLShadowInputs, wgslShadow} from '../effects/wgsl-shadow';

/** Borrowed custom meter footprint; intended for the native H3 low-precision column branch. */
export type GPUMeshColumnLayerProps = Omit<LayerProps, 'data'> &
  GPUVectorLayerSelectionProps &
  GPUBrushingProps & {
    /** Open, counterclockwise convex perimeter in meters. STORAGE usage; no upload or readback here. */
    vertices: GPUData<'float32x2'>;
    /** Native ColumnLayer's cos(pi / vertexCount) times mean vertex distance from the origin. */
    edgeDistance: number;
    getPosition: GPUVector<'float32x2'>;
    /**
     * Optional fp64 low part: `getPosition + getPositionLow` reconstructs the Float64 longitude/latitude.
     * Deck's project_position_vec3_f64 adds it after subtracting the viewport origin in Float32.
     * Must share getPosition's chunk boundaries.
     */
    getPositionLow?: GPUVector<'float32x2'>;
    getFillColor?: Color | GPUVector<'unorm8x4'>;
    getLineColor?: Color | GPUVector<'unorm8x4'>;
    getCoverage?: number | GPUVector<'float32'>;
    getElevation?: number | GPUVector<'float32'>;
    getLineWidth?: number | GPUVector<'float32'>;
    coverage?: number;
    elevationScale?: number;
    lineWidthScale?: number;
    lineWidthMinPixels?: number;
    lineWidthMaxPixels?: number;
    lineWidthUnits?: 'pixels' | 'meters' | 'common';
    filled?: boolean;
    stroked?: boolean;
    extruded?: boolean;
    /** Kepler EnhancedColumnLayer applies this only to the flat outline pass. */
    strokeOpacity?: number;
    flatShading?: boolean;
    material?: Material;
    /** Wireframe rendering is not part of this meter-footprint primitive. */
    wireframe?: false;
  };

type LayerState = {model: GPUVectorModel | null; styleBuffer: Buffer | null; defaults: Buffer[]};
const STYLE_BYTE_LENGTH = 128;

/** Native Column/EnhancedColumn geometry, projection and lighting over resident source chunks. */
export class GPUMeshColumnLayer extends Layer<GPUMeshColumnLayerProps> {
  static override layerName = 'GPUMeshColumnLayer';
  static override defaultProps = {material: true};

  override getAttributeManager() {
    return null;
  }
  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPUMeshColumnLayer requires WebGPU');
    // Caller prepares a convex CCW meter perimeter and its native edge-distance metadata.
    assert(
      this.props.vertices.format === 'float32x2' &&
        this.props.vertices.length >= 3 &&
        this.props.vertices.byteOffset % 4 === 0 &&
        this.props.vertices.byteStride % 4 === 0 &&
        !this.props.vertices.nullBitmap &&
        Number.isFinite(this.props.edgeDistance) &&
        this.props.edgeDistance > 0
    );
    // Wireframe/globe require the host's separate native-compatible rendering branch.
    assert(!this.props.wireframe);
    validateGPUVectorPickingCapacity(this.props.getPosition.length, Boolean(this.props.pickable));
    const vectors = getVectors(this.props);
    getGPUVectorLayerBatches(this.id, vectors, {
      positions: ['float32x2'],
      positionsLow: ['float32x2'],
      colors: ['unorm8x4'],
      lineColors: ['unorm8x4'],
      elevations: ['float32'],
      coverages: ['float32'],
      widths: ['float32'],
      selectionMask: ['uint32']
    });
    if (!this.props.getPosition.data.length) {
      this.setState({model: null, styleBuffer: null, defaults: []} satisfies LayerState);
      return;
    }
    const styleBuffer = createGPUVectorStyleBuffer(
      device,
      `${this.id}-style`,
      STYLE_BYTE_LENGTH,
      this.props.getPosition.data.length * 2
    );
    const defaults = [
      device.createBuffer({data: new Float32Array([1])}),
      device.createBuffer({data: new Uint8Array([0, 0, 0, 255])}),
      device.createBuffer({data: new Uint32Array([1])}),
      device.createBuffer({data: new Float32Array([0, 0])})
    ];
    try {
      const model = new GPUVectorModel(device, {
        ...this.getShaders({
          modules: [
            project32,
            this.props.flatShading ? phongMaterial : gouraudMaterial,
            picking,
            wgslShadow,
            ...(this.props.brushingEnabled ? [gpuBrushing] : [])
          ],
          // Native H3 columns brush the column center ('source' target), per instance.
          source: replaceGPUBrushingMarker(
            applyWGSLShadow(getShader(Boolean(this.props.flatShading)), {
              varyingAnchor: '@location(4) @interpolate(flat) selected: u32,',
              firstLocation: 5,
              positionAssignment: 'output.position = projected.clipPosition;',
              commonPosition: 'projected.commonPosition',
              clipPosition: 'projected.clipPosition',
              finalReturn: 'return vec4<f32>(highlighted.rgb * highlighted.a, highlighted.a);',
              finalColorExpression: 'highlighted'
            }),
            Boolean(this.props.brushingEnabled),
            'output.selected = select(0u, 1u, renderable);',
            'output.selected = select(0u, 1u, renderable && brushing_isPointInRange(positions));'
          )
        }),
        id: `${this.id}-model`,
        // Preserve native strips: expanding them changes derivative normals at raster helper quads.
        topology: 'triangle-strip',
        isInstanced: true,
        inferDrawCount: 'instance',
        vertexCount: 0,
        instanceCount: 0,
        attributes: {
          positions: getGPUVectorBuffer(vectors.positions),
          positionsLow: vectors.positionsLow
            ? getGPUVectorBuffer(vectors.positionsLow)
            : defaults[3],
          colors: vectors.colors ? getGPUVectorBuffer(vectors.colors) : defaults[1],
          lineColors: vectors.lineColors ? getGPUVectorBuffer(vectors.lineColors) : defaults[1],
          elevations: vectors.elevations ? getGPUVectorBuffer(vectors.elevations) : defaults[0],
          coverages: vectors.coverages ? getGPUVectorBuffer(vectors.coverages) : defaults[0],
          widths: vectors.widths ? getGPUVectorBuffer(vectors.widths) : defaults[0],
          selectionMask: vectors.selectionMask
            ? getGPUVectorBuffer(vectors.selectionMask)
            : defaults[2]
        },
        bufferLayout: [
          makeGPUVectorBufferLayout(vectors.positions, 'positions'),
          vectors.positionsLow
            ? makeGPUVectorBufferLayout(vectors.positionsLow, 'positionsLow')
            : {
                name: 'positionsLow',
                byteStride: 0,
                stepMode: 'instance' as const,
                attributes: [{attribute: 'positionsLow', format: 'float32x2' as const}]
              },
          ...(['colors', 'lineColors', 'elevations', 'coverages', 'widths'] as const).map(name =>
            vectors[name]
              ? makeGPUVectorBufferLayout(vectors[name], name)
              : {
                  name,
                  byteStride: 0,
                  stepMode: 'instance' as const,
                  attributes: [
                    {
                      attribute: name,
                      format:
                        name === 'colors' || name === 'lineColors'
                          ? ('unorm8x4' as const)
                          : ('float32' as const)
                    }
                  ]
                }
          ),
          makeGPUVectorSelectionLayout(vectors.selectionMask)
        ],
        bindings: {
          columnStyle: getGPUVectorStyleBufferBinding(styleBuffer, STYLE_BYTE_LENGTH),
          footprint: this.props.vertices.buffer
        }
      });
      model.userData['boundInputs'] = [
        ...Object.values(vectors),
        this.props.vertices,
        this.props.flatShading,
        Boolean(this.props.brushingEnabled)
      ];
      this.setState({model, styleBuffer, defaults} satisfies LayerState);
    } catch (error) {
      defaults.forEach(buffer => buffer.destroy());
      styleBuffer.destroy();
      throw error;
    }
  }
  override getModels(): GPUVectorModel[] {
    const model = (this.state as LayerState).model;
    return model ? [model] : [];
  }
  override updateState({props}: UpdateParameters<this>): void {
    const previous = (this.state as LayerState).model?.userData['boundInputs'] as
      | unknown[]
      | undefined;
    const next = [
      ...Object.values(getVectors(props)),
      props.vertices,
      props.flatShading,
      Boolean(props.brushingEnabled)
    ];
    if (!previous || next.some((value, index) => value !== previous[index])) {
      this.destroyResources();
      this.initializeState(this.context);
    }
  }
  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer: previousBuffer} = this.state as LayerState;
    if (!model || !previousBuffer) return;
    ensureWGSLShadowInputs([model], this.context.device);
    validateGPUVectorPickingCapacity(this.props.getPosition.length, Boolean(this.props.pickable));
    const batches = this.props.getPosition.data.length;
    const styleBuffer = resizeGPUVectorStyleBuffer(previousBuffer, STYLE_BYTE_LENGTH, batches * 2);
    if (styleBuffer !== previousBuffer) this.setState({styleBuffer});
    const vectors = getVectors(this.props);
    const bytes = new ArrayBuffer(STYLE_BYTE_LENGTH);
    const floats = new Float32Array(bytes);
    const uints = new Uint32Array(bytes);
    floats.set(
      normalizeColor(isColor(this.props.getFillColor) ? this.props.getFillColor : [0, 0, 0, 255]),
      0
    );
    floats.set(
      normalizeColor(isColor(this.props.getLineColor) ? this.props.getLineColor : [0, 0, 0, 255]),
      4
    );
    floats.set(
      [
        typeof this.props.getElevation === 'number' ? this.props.getElevation : 1000,
        typeof this.props.getCoverage === 'number' ? this.props.getCoverage : 1,
        typeof this.props.getLineWidth === 'number' ? this.props.getLineWidth : 1,
        this.props.coverage ?? 1,
        this.props.elevationScale ?? 1,
        this.props.lineWidthScale ?? 1,
        this.props.lineWidthMinPixels ?? 0,
        this.props.lineWidthMaxPixels ?? Number.MAX_SAFE_INTEGER
      ],
      8
    );
    uints.set(
      [
        vectors.colors ? 1 : 0,
        vectors.lineColors ? 1 : 0,
        vectors.elevations ? 1 : 0,
        vectors.coverages ? 1 : 0,
        vectors.widths ? 1 : 0,
        this.props.vertices.length,
        this.props.vertices.byteOffset / 4,
        this.props.vertices.byteStride / 4,
        0,
        this.props.extruded === false ? 0 : 1,
        this.props.stroked ? 1 : 0,
        0
      ],
      16
    );
    floats[28] = this.props.edgeDistance;
    floats[29] = Math.pow(this.props.strokeOpacity ?? this.props.opacity ?? 1, 1 / 2.2);
    new Int32Array(bytes)[30] = UNIT[this.props.lineWidthUnits ?? 'meters'];
    const extruded = this.props.extruded !== false;
    const sideVertices = (this.props.vertices.length + 1) * 2;
    const fillVertices =
      extruded || this.props.stroked
        ? (this.props.vertices.length + 1) * 3 + 1
        : this.props.vertices.length;
    setGPUBrushingProps(model, this.context, this.props);
    for (const stroke of [false, true]) {
      if (stroke ? extruded || !this.props.stroked : this.props.filled === false) continue;
      uints[27] = stroke ? 1 : 0;
      model.setVertexCount(stroke ? sideVertices : fillVertices);
      model.drawBatches(renderPass, {
        vectors,
        onBatch: batch => {
          uints[24] = batch.rowIndexOffset;
          // Separate regions prevent queued stroke writes from changing a prior fill command.
          const binding = getGPUVectorStyleBufferBinding(
            styleBuffer,
            STYLE_BYTE_LENGTH,
            batch.batchIndex + (stroke ? batches : 0)
          );
          styleBuffer.write(new Uint8Array(bytes), binding.offset);
          model.setBindings({columnStyle: binding});
        }
      });
    }
  }
  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    const result = info as GPUVectorLayerPickingInfo;
    result.gpuVector = getGPUVectorPickingProvenance(this.props.getPosition, result.index);
    return result;
  }
  override finalizeState(context: LayerContext): void {
    this.destroyResources();
    super.finalizeState(context);
  }
  private destroyResources(): void {
    const state = this.state as LayerState;
    state.model?.destroy();
    state.styleBuffer?.destroy();
    state.defaults?.forEach(buffer => buffer.destroy());
    this.setState({model: null, styleBuffer: null, defaults: []});
  }
}

function getVectors(props: GPUMeshColumnLayerProps) {
  return {
    positions: props.getPosition,
    positionsLow: props.getPositionLow,
    colors: isGPUVector(props.getFillColor) ? props.getFillColor : undefined,
    lineColors: isGPUVector(props.getLineColor) ? props.getLineColor : undefined,
    elevations: isGPUVector(props.getElevation) ? props.getElevation : undefined,
    coverages: isGPUVector(props.getCoverage) ? props.getCoverage : undefined,
    widths: isGPUVector(props.getLineWidth) ? props.getLineWidth : undefined,
    selectionMask: props.selectionMask
  };
}
function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
function isColor(value: unknown): value is Color {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}
function normalizeColor(value: Color): number[] {
  return [value[0] / 255, value[1] / 255, value[2] / 255, (value[3] ?? 255) / 255];
}

// Follows pinned Deck ColumnLayer and Kepler EnhancedColumnLayer's vertex contracts. Native
// zigzag cap triangulation and radial side normals matter for matching interpolated lighting.
function getShader(flatShading: boolean): string {
  return /* wgsl */ `
${GPU_HIGHLIGHT_WGSL}
struct ColumnStyle {
  fillColor: vec4<f32>, lineColor: vec4<f32>,
  elevation: f32, instanceCoverage: f32, lineWidth: f32, coverage: f32,
  elevationScale: f32, widthScale: f32, widthMin: f32, widthMax: f32,
  useColors: u32, useLineColors: u32, useElevations: u32, useCoverages: u32,
  useWidths: u32, vertexCount: u32, vertexOffset: u32, vertexStride: u32,
  rowIndexOffset: u32, extruded: u32, stroked: u32, isStroke: u32,
  edgeDistance: f32, strokeOpacity: f32, widthUnits: i32, _padding: u32,
};
@group(0) @binding(auto) var<uniform> columnStyle: ColumnStyle;
@group(0) @binding(auto) var<storage, read> footprint: array<f32>;
struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) cameraPosition: vec3<f32>,
  @location(2) commonPosition: vec4<f32>,
  @location(3) @interpolate(flat) pickingColor: vec3<f32>,
  @location(4) @interpolate(flat) selected: u32,
};
fn getFootprint(index: u32) -> vec2<f32> {
  let offset = columnStyle.vertexOffset + (index % columnStyle.vertexCount) * columnStyle.vertexStride;
  return vec2<f32>(footprint[offset], footprint[offset + 1u]);
}
fn capIndex(index: u32) -> u32 {
  let half = index / 2u;
  return select(half, (columnStyle.vertexCount - half) % columnStyle.vertexCount, index % 2u != 0u);
}
@vertex fn vertexMain(
  @location(0) positions: vec2<f32>, @location(1) colors: vec4<f32>,
  @location(2) lineColors: vec4<f32>, @location(3) elevations: f32,
  @location(4) coverages: f32, @location(5) widths: f32,
  @location(6) selectionMask: u32, @location(7) positionsLow: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32
) -> Varyings {
  let sideCount = (columnStyle.vertexCount + 1u) * 2u;
  let hasSides = columnStyle.extruded != 0u || columnStyle.stroked != 0u;
  var corner: vec2<f32>; var height = 1.0; var normal = vec3<f32>(0.0, 0.0, 1.0);
  if (hasSides && vertexIndex < sideCount) {
    corner = getFootprint(vertexIndex / 2u);
    height = select(1.0, -1.0, vertexIndex % 2u != 0u);
    normal = vec3<f32>(corner, 0.0);
  } else if (hasSides && vertexIndex == sideCount) {
    corner = getFootprint(0u);
    height = -1.0;
    normal = vec3<f32>(0.0);
  } else {
    let capVertex = select(vertexIndex + 1u, vertexIndex - sideCount - 1u, hasSides);
    corner = getFootprint(capIndex(capVertex));
  }
  let instanceElevation = select(columnStyle.elevation, elevations, columnStyle.useElevations != 0u);
  let instanceCoverage = select(columnStyle.instanceCoverage, coverages, columnStyle.useCoverages != 0u);
  let fillColor = select(columnStyle.fillColor, colors, columnStyle.useColors != 0u);
  let lineColor = select(columnStyle.lineColor, lineColors, columnStyle.useLineColors != 0u);
  let baseColor = select(fillColor, lineColor, columnStyle.isStroke != 0u);
  geometry.worldPosition = vec3<f32>(positions, 0.0);
  let encoded = instanceIndex + columnStyle.rowIndexOffset + 1u;
  let pickingColor = vec3<f32>(f32(encoded & 255u), f32((encoded >> 8u) & 255u), f32((encoded >> 16u) & 255u)) / 255.0;
  geometry.pickingColor = pickingColor;
  var elevation = 0.0; var strokeOffsetRatio = 1.0;
  if (columnStyle.extruded != 0u) {
    elevation = instanceElevation * (height + 1.0) / 2.0 * columnStyle.elevationScale;
  } else if (columnStyle.stroked != 0u) {
    let width = select(columnStyle.lineWidth, widths, columnStyle.useWidths != 0u);
    let widthPixels = clamp(project_unit_size_to_pixel(width * columnStyle.widthScale, columnStyle.widthUnits), columnStyle.widthMin, columnStyle.widthMax) / 2.0;
    // EnhancedColumn intentionally excludes instanceCoverage from this denominator.
    let halfOffset = project_pixel_size_float(widthPixels) / project_size_float(columnStyle.edgeDistance * columnStyle.coverage);
    strokeOffsetRatio -= select(halfOffset, sign(height) * halfOffset, columnStyle.isStroke != 0u);
  }
  let renderable = selectionMask != 0u && baseColor.a > 0.0 && instanceElevation >= 0.0 && columnStyle.coverage != 0.0 && instanceCoverage != 0.0;
  let dotRadius = columnStyle.coverage * instanceCoverage * select(0.0, 1.0, renderable);
  let offset = vec3<f32>(project_size_vec2(corner * strokeOffsetRatio * dotRadius), 0.0);
  let projected = project_position_to_clipspace_and_commonspace(vec3<f32>(positions, elevation), vec3<f32>(positionsLow, 0.0), offset);
  geometry.position = projected.commonPosition;
  geometry.normal = project_normal(normal);
  var rgb = baseColor.rgb;
  ${
    flatShading
      ? ''
      : `if (columnStyle.extruded != 0u && columnStyle.isStroke == 0u) {
    rgb = lighting_getLightColor2(baseColor.rgb, project.cameraPosition, geometry.position.xyz, geometry.normal);
  }`
  }
  var output: Varyings;
  output.position = projected.clipPosition;
  output.color = vec4<f32>(rgb, baseColor.a * select(layer.opacity, columnStyle.strokeOpacity, columnStyle.isStroke != 0u));
  output.commonPosition = projected.commonPosition;
  output.cameraPosition = project.cameraPosition;
  output.pickingColor = pickingColor;
  output.selected = select(0u, 1u, renderable);
  return output;
}
@fragment fn fragmentMain(input: Varyings) -> @location(0) vec4<f32> {
  var color = input.color;
  ${
    flatShading
      ? `if (columnStyle.extruded != 0u && columnStyle.isStroke == 0u) {
    let normal = normalize(cross(dpdy(input.commonPosition.xyz), dpdx(input.commonPosition.xyz)));
    color = vec4<f32>(lighting_getLightColor2(color.rgb, input.cameraPosition, input.commonPosition.xyz, normal), color.a);
  }`
      : ''
  }
  // Derivatives above must run before nonuniform selection/picking exits, including helper lanes.
  if (input.selected == 0u) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let highlighted = gpuHighlight(color, input.pickingColor);
  return vec4<f32>(highlighted.rgb * highlighted.a, highlighted.a);
}`;
}
