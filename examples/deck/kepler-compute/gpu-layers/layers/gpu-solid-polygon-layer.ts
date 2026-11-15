// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  picking,
  project32,
  gouraudMaterial,
  type Material,
  type Color,
  type LayerContext,
  type LayerProps,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {assert, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import {GPUConstant, type GPUData, type GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {PolygonAttributeModel} from '@luma.gl/engine/models';
import {
  getGPUVectorPickingProvenance,
  validateGPUVectorPickingCapacity,
  type GPUVectorLayerPickingInfo
} from './gpu-vector-layer-utils';
import {GPU_GLOBE_WGSL} from './gpu-globe';
import {applyWGSLShadow, ensureWGSLShadowInputs, wgslShadow} from '../effects/wgsl-shadow';
import {
  applyGPUBrushingVisibility,
  gpuBrushing,
  setGPUBrushingProps,
  type GPUBrushingProps
} from './gpu-brushing';
import {
  GPU_HIGHLIGHT_RANGE_WGSL,
  gpuHighlightRange,
  setGPUHighlightRangeProps,
  type GPUHighlightRangeProps
} from './gpu-highlight';

import {applyGPUPolygonLow, GPUPolygonLowBinding, GPU_POLYGON_LOW_MARKER} from './gpu-polygon-low';

import {
  GPUPolygonExtrusionBinding,
  GPU_POLYGON_EXTRUSION_SHADER,
  type GPUPolygonExtrusionRings
} from './gpu-polygon-extrusion';

import {
  borrowGPUGeometryChunk,
  GPUGeometrySelectionBinding,
  GPU_GEOMETRY_SELECTION_SHADER,
  validateGPUGeometrySelection,
  validateGPUGeometryChunks
} from './gpu-geometry-selection';

/** GPUVector-native tessellated filled-polygon props. Input vectors are borrowed. */
export type GPUSolidPolygonLayerProps = Omit<LayerProps, 'data'> &
  GPUHighlightRangeProps &
  GPUBrushingProps & {
    /** One nonzero flag per source feature, aligned with positions' logical rows. */
    selectionMask?: GPUVector<'uint32'>;
    positions: GPUVector<VertexList<'float32x4'>>;
    /**
     * Optional fp64 low part of XY, vertex-aligned with `positions` (float32x2, tightly packed).
     * Ring lows for extruded walls go in `extrusionRings.positionsLow`. Omit for the unchanged
     * Float32 path; the low part only affects Deck's LNGLAT offset mode at high zoom.
     */
    positionsLow?: GPUVector<VertexList<'float32x2'>>;
    /** Tessellation row indices must be zero-based global source feature indices. */
    rowIndices: GPUVector<VertexList<'uint32'>>;
    indices: GPUVector<VertexList<'uint32'>>;
    /** Feature colors or top-vertex colors. Walls use each feature's first top-vertex color. */
    getFillColor?: Color | GPUVector<'unorm8x4' | VertexList<'unorm8x4'>>;
    extruded?: boolean;
    filled?: boolean;
    wireframe?: boolean;
    getLineColor?: Color | GPUVector<'unorm8x4' | VertexList<'unorm8x4'>>;
    /** Source-aligned unscaled offset, applied to input Z even without extrusion. */
    getElevationOffset?: number | GPUVector<'float32'>;
    /** Source-aligned elevation, added to input Z. Defaults to Deck's 1000 world units. */
    getElevation?: number | GPUVector<'float32'>;
    elevationScale?: number;
    material?: Material;
    /** Required for extrusion. Closed rings preserve source batch and global feature mappings. */
    extrusionRings?: GPUPolygonExtrusionRings;
  };

type GPUSolidPolygonLayerBatch = {
  model?: PolygonAttributeModel;
  selection: GPUGeometrySelectionBinding;
  extrusion: GPUPolygonExtrusionBinding;
  low?: GPUPolygonLowBinding;
  sideModel?: Model;
  wireframeModel?: Model;
  chunkIndex: number;
  rowIndexBase: number;
};

type GPUSolidPolygonLayerState = {
  batches: GPUSolidPolygonLayerBatch[];
  boundInputs: unknown[];
  retirement: {bytes: number};
};

const GPU_SOLID_POLYGON_SHADER = /* wgsl */ `
${GPU_HIGHLIGHT_RANGE_WGSL}
${GPU_GLOBE_WGSL}
${GPU_GEOMETRY_SELECTION_SHADER}
${GPU_POLYGON_EXTRUSION_SHADER}
${GPU_POLYGON_LOW_MARKER}
/*BRUSHING_VISIBILITY*/
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) @interpolate(flat) pickingColor: vec3<f32>,
  @location(2) @interpolate(flat) selected: u32,
  @location(3) brushed: f32,
  @location(4) globePosition: vec3<f32>,
};

fn encodePickingColor(rowIndex: u32) -> vec3<f32> {
  let value = rowIndex + 1u;
  return vec3<f32>(f32(value % 256u), f32((value / 256u) % 256u), f32((value / 65536u) % 256u)) / 255.0;
}

@vertex fn vertexMain(
  @location(0) positions: vec4<f32>,
  @location(1) colors: vec4<f32>,
  @location(2) rowIndices: u32,
  @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
  let pickingColor = encodePickingColor(rowIndices);
  geometry.worldPosition = positions.xyz;
  geometry.pickingColor = pickingColor;
  var output: VertexOutput;
  let worldPosition = positions.xyz + vec3<f32>(0.0, 0.0, getPolygonBaseElevation(rowIndices) + getPolygonElevation(rowIndices));
  var projected = project_position_to_clipspace_and_commonspace(worldPosition, getPolygonTopLow(vertexIndex), vec3<f32>(0.0));
  if (gpuGlobeUsesMercatorCommon()) {
    projected.commonPosition = vec4<f32>(gpuGlobeFromMercatorCommon(worldPosition), 1.0);
    projected.clipPosition = project_common_position_to_clipspace(projected.commonPosition);
  }
  geometry.position = projected.commonPosition;
  geometry.normal = project_normal(vec3<f32>(0.0, 0.0, 1.0));
  output.position = projected.clipPosition;
  output.color = shadePolygon(getPolygonColor(rowIndices, colors, false), geometry.normal, geometry.position);
  output.pickingColor = pickingColor;
  output.selected = select(0u, 1u, isPolygonSourceRow(rowIndices) && isGeometryRowSelected(rowIndices));
  output.globePosition = geometry.position.xyz;
  output.brushed = brushingVisibility(positions.xy);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (input.selected == 0u || input.brushed < 0.5 || gpuGlobeIsCommonHidden(input.globePosition)) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let color = gpuHighlightRows(input.color, input.pickingColor);
  let alpha = color.a * layer.opacity;
  return vec4<f32>(color.rgb * alpha, alpha);
}`;

/** Fill vertices never read ring storage; omitting those bindings keeps the stage within 8 storage buffers. */
export const GPU_POLYGON_FILL_ONLY_BINDINGS = [
  'polygonRingPositions',
  'polygonRingSources',
  'polygonOrientations',
  'polygonRingLowPositions'
] as const;

const GPU_POLYGON_FILL_SHADER = GPU_SOLID_POLYGON_SHADER.replace(
  /^@group\(0\) @binding\(auto\) var<storage, read> polygon(RingPositions|RingSources|Orientations): .*\n/gm,
  ''
).replace(
  /fn getPolygonRingPosition[\s\S]*?\n}\n/,
  'fn getPolygonRingPosition(index: u32) -> vec3<f32> { return vec3<f32>(0.0); }\n'
);

const GPU_POLYGON_SIDE_SHADER = GPU_SOLID_POLYGON_SHADER.replace(
  /@vertex fn vertexMain\([\s\S]*?\n}\n\n@fragment/,
  /* wgsl */ `@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  let record = getPolygonEdge(instanceIndex);
  let source = getPolygonRingPosition(record.x);
  let targetPosition = getPolygonRingPosition(record.y);
  let rowIndex = polygonRingSources[polygonExtrusion.ringSourceOffset + record.z * polygonExtrusion.ringSourceStride];
  let corners = array<vec2<f32>, 6>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(0.0,1.0),vec2<f32>(0.0,1.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0));
  let corner = corners[vertexIndex];
  let position = mix(source, targetPosition, corner.x);
  let worldPosition = position + vec3<f32>(0.0, 0.0, getPolygonBaseElevation(rowIndex) + getPolygonElevation(rowIndex) * corner.y);
  var output: VertexOutput;
  output.selected = 0u;
  if (length(targetPosition.xy - source.xy) == 0.0 || !isPolygonSourceRow(rowIndex)) { return output; }
  let sideLow = mix(getPolygonRingLow(record.x), getPolygonRingLow(record.y), corner.x);
  let projected = project_position_to_clipspace_and_commonspace(worldPosition, sideLow, vec3<f32>(0.0));
  var normalSign = polygonExtrusion.normalSign;
  if (polygonExtrusion.useOrientations != 0u) { normalSign = polygonOrientations[polygonExtrusion.orientationOffset + record.z * polygonExtrusion.orientationStride]; }
  geometry.worldPosition = position;
  geometry.position = projected.commonPosition;
  geometry.normal = projectPolygonNormal(vec3<f32>(targetPosition.y-source.y, source.x-targetPosition.x, 0.0) * normalSign);
  geometry.pickingColor = encodePickingColor(rowIndex);
  output.position = projected.clipPosition;
  output.color = shadePolygon(getPolygonColor(rowIndex, vec4<f32>(0.0), true), geometry.normal, geometry.position);
  output.pickingColor = geometry.pickingColor;
  output.selected = select(0u, 1u, isGeometryRowSelected(rowIndex));
  output.globePosition = geometry.position.xyz;
  output.brushed = brushingVisibility(position.xy);
  return output;
}

@fragment`
);

const GPU_POLYGON_WIREFRAME_SHADER = GPU_POLYGON_SIDE_SHADER.replace(
  'let corners = array<vec2<f32>, 6>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(0.0,1.0),vec2<f32>(0.0,1.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0));',
  'let corners = array<vec2<f32>, 4>(vec2<f32>(1.0,0.0),vec2<f32>(0.0,0.0),vec2<f32>(0.0,1.0),vec2<f32>(1.0,1.0));'
);

/** Adds WGSL shadow map casting and receiving to a polygon shader variant. */
function applyPolygonShadow(source: string): string {
  return applyWGSLShadow(source, {
    varyingAnchor: '@location(4) globePosition: vec3<f32>,',
    firstLocation: 5,
    positionAssignment: 'output.position = projected.clipPosition;',
    commonPosition: 'projected.commonPosition',
    clipPosition: 'projected.clipPosition',
    finalReturn: 'return vec4<f32>(color.rgb * alpha, alpha);',
    finalColorExpression: 'vec4<f32>(color.rgb, alpha)'
  });
}

/** Deck host for already tessellated polygon GPUVectors. Tessellation belongs in adapters. */
export class GPUSolidPolygonLayer extends Layer<GPUSolidPolygonLayerProps> {
  static override layerName = 'GPUSolidPolygonLayer';
  static override defaultProps = {material: true};

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error('GPUSolidPolygonLayer requires WebGPU');
    this.setState({batches: [], boundInputs: [], retirement: {bytes: 0}});
    this.createModel();
  }

  override updateState({props}: UpdateParameters<this>): void {
    validateGPUVectorPickingCapacity(props.positions.length, Boolean(props.pickable));
    validateGPUGeometrySelection(this.context.device, props.positions, props.selectionMask);
    const {boundInputs} = this.state as GPUSolidPolygonLayerState;
    if (
      props.positions !== boundInputs[0] ||
      props.rowIndices !== boundInputs[1] ||
      props.indices !== boundInputs[2] ||
      props.getFillColor !== boundInputs[3] ||
      props.extrusionRings !== boundInputs[4] ||
      props.positionsLow !== boundInputs[9] ||
      props.extruded !== boundInputs[5] ||
      props.filled !== boundInputs[6] ||
      props.wireframe !== boundInputs[7] ||
      Boolean(props.brushingEnabled) !== boundInputs[8]
    ) {
      this.destroyModel();
      this.createModel();
    }
  }

  override getModels(): Model[] {
    return (
      (this.state as GPUSolidPolygonLayerState | undefined)?.batches.flatMap(batch => [
        ...(batch.model ? [batch.model] : []),
        ...(batch.sideModel ? [batch.sideModel] : []),
        ...(batch.wireframeModel ? [batch.wireframeModel] : [])
      ]) ?? []
    );
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    if (this.props.filled === false && !this.props.wireframe) return;
    ensureWGSLShadowInputs(this.getModels(), this.context.device);
    setGPUHighlightRangeProps(this.getModels(), this.props);
    for (const model of this.getModels()) setGPUBrushingProps(model, this.context, this.props);
    validateGPUVectorPickingCapacity(this.props.positions.length, Boolean(this.props.pickable));
    validateGPUGeometrySelection(
      this.context.device,
      this.props.positions,
      this.props.selectionMask
    );
    if (isGPUVector(this.props.getElevation))
      validateGPUGeometryChunks(this.props.positions, [this.props.getElevation]);
    if (isGPUVector(this.props.getElevationOffset))
      validateGPUGeometryChunks(this.props.positions, [this.props.getElevationOffset]);
    if (isGPUVector(this.props.getLineColor))
      validateGPUGeometryChunks(this.props.positions, [this.props.getLineColor]);
    for (const batch of (this.state as GPUSolidPolygonLayerState).batches) {
      if (batch.wireframeModel && this.props.wireframe) {
        batch.wireframeModel.setBindings({
          ...batch.selection.update(this.props.selectionMask, batch.chunkIndex, batch.rowIndexBase),
          ...batch.extrusion.update(
            this.getExtrusionProps(batch.chunkIndex, batch.rowIndexBase, true),
            true
          )
        });
        batch.wireframeModel.draw(renderPass);
      }
      if (this.props.filled === false) continue;
      const bindings = {
        ...batch.selection.update(this.props.selectionMask, batch.chunkIndex, batch.rowIndexBase),
        ...batch.extrusion.update(this.getExtrusionProps(batch.chunkIndex, batch.rowIndexBase))
      };
      batch.model?.setBindings(
        Object.fromEntries(
          Object.entries(bindings).filter(
            ([name]) => !(GPU_POLYGON_FILL_ONLY_BINDINGS as readonly string[]).includes(name)
          )
        )
      );
      batch.sideModel?.setBindings(bindings);
      batch.sideModel?.draw(renderPass);
      batch.model?.draw(renderPass);
    }
  }

  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    const result = info as GPUVectorLayerPickingInfo;
    result.gpuVector = getGPUVectorPickingProvenance(this.props.positions, result.index);
    return result;
  }

  override finalizeState(context: LayerContext): void {
    this.destroyModel();
    super.finalizeState(context);
  }

  private createModel(): void {
    const brushing = Boolean(this.props.brushingEnabled);
    const modules = [project32, picking, gouraudMaterial, gpuHighlightRange, wgslShadow];
    if (brushing) modules.push(gpuBrushing);
    // Walls already bind 8 storage buffers; ring lows (a 9th) need a device that allows more.
    const wallRingLowEnabled = this.context.device.limits.maxStorageBuffersPerShaderStage >= 9;
    const lowEnabled = Boolean(this.props.positionsLow || this.props.extrusionRings?.positionsLow);
    const shaderProps = this.getShaders({
      modules,
      source: applyGPUBrushingVisibility(
        applyGPUPolygonLow(applyPolygonShadow(GPU_POLYGON_FILL_SHADER), lowEnabled, false),
        brushing
      )
    });
    const vertexColors =
      isGPUVector(this.props.getFillColor) &&
      this.props.getFillColor.format === 'vertex-list<unorm8x4>'
        ? (this.props.getFillColor as GPUVector<VertexList<'unorm8x4'>>)
        : undefined;
    const color = new GPUConstant({format: 'unorm8x4', value: new Uint8Array([0, 0, 0, 255])});
    const {device} = this.context;
    validateGPUVectorPickingCapacity(this.props.positions.length, Boolean(this.props.pickable));
    validateGPUGeometrySelection(device, this.props.positions, this.props.selectionMask);
    validateGPUGeometryChunks(this.props.positions, [
      this.props.rowIndices,
      this.props.indices,
      isGPUVector(this.props.getElevationOffset) ? this.props.getElevationOffset : undefined,
      isGPUVector(this.props.getLineColor) ? this.props.getLineColor : undefined,
      isGPUVector(this.props.getFillColor) ? this.props.getFillColor : undefined
    ]);
    if (isGPUVector(this.props.getElevation))
      validateGPUGeometryChunks(this.props.positions, [this.props.getElevation]);
    if (this.props.positionsLow)
      validateGPUGeometryChunks(this.props.positions, [this.props.positionsLow]);
    const rings = this.props.extrusionRings;
    // Extrusion requires explicit rings; the renderer must never silently omit side walls.
    assert(!this.props.extruded || rings);
    if (rings) {
      assert(rings.positions.data.length === this.props.positions.data.length);
      validateGPUGeometryChunks(rings.positions, [
        rings.sourceRowIds,
        rings.sideOrientations,
        rings.positionsLow
      ]);
    }
    const sideShaderProps = this.getShaders({
      modules,
      source: applyGPUBrushingVisibility(
        applyGPUPolygonLow(
          applyPolygonShadow(GPU_POLYGON_SIDE_SHADER),
          lowEnabled,
          wallRingLowEnabled,
          false
        ),
        brushing
      )
    });
    const wireframeShaderProps = this.getShaders({
      modules,
      source: applyGPUBrushingVisibility(
        applyGPUPolygonLow(
          applyPolygonShadow(GPU_POLYGON_WIREFRAME_SHADER),
          lowEnabled,
          wallRingLowEnabled,
          false
        ),
        brushing
      )
    });
    const batches: GPUSolidPolygonLayerBatch[] = [];
    let rowIndexBase = 0;
    try {
      for (const [chunkIndex, chunk] of this.props.positions.data.entries()) {
        if (
          chunk.length > 0 &&
          (chunk.valueLength > 0 || (rings?.positions.data[chunkIndex].valueLength ?? 0) > 0)
        ) {
          const selection = new GPUGeometrySelectionBinding(device, `${this.id}-${chunkIndex}`);
          let extrusion: GPUPolygonExtrusionBinding | undefined;
          let low: GPUPolygonLowBinding | undefined;
          let model: PolygonAttributeModel | undefined;
          let sideModel: Model | undefined;
          let wireframeModel: Model | undefined;
          try {
            extrusion = new GPUPolygonExtrusionBinding(
              device,
              `${this.id}-${chunkIndex}`,
              chunk,
              rings?.positions.data[chunkIndex],
              rings?.sourceRowIds.data[chunkIndex],
              rings?.sideOrientations?.data[chunkIndex],
              Boolean(this.props.wireframe)
            );
            if (lowEnabled)
              low = new GPUPolygonLowBinding(
                device,
                `${this.id}-${chunkIndex}`,
                this.props.positionsLow?.data[chunkIndex],
                rings?.positionsLow?.data[chunkIndex]
              );
            const bindings = {
              ...selection.update(this.props.selectionMask, chunkIndex, rowIndexBase),
              ...extrusion.update(this.getExtrusionProps(chunkIndex, rowIndexBase)),
              ...low?.getBindings()
            };
            if (this.props.filled !== false && chunk.valueLength > 0)
              model = new PolygonAttributeModel(device, {
                ...shaderProps,
                id: `${this.id}-model-${chunkIndex}`,
                positions: borrowGPUGeometryChunk(this.props.positions, chunkIndex),
                colors: vertexColors ? borrowGPUGeometryChunk(vertexColors, chunkIndex) : color,
                rowIndices: borrowGPUGeometryChunk(this.props.rowIndices, chunkIndex),
                indices: borrowGPUGeometryChunk(this.props.indices, chunkIndex),
                bindings: Object.fromEntries(
                  Object.entries(bindings).filter(
                    ([name]) =>
                      !(GPU_POLYGON_FILL_ONLY_BINDINGS as readonly string[]).includes(name)
                  )
                ),
                shaderInputs: shaderProps.shaderInputs as never
              });
            if (this.props.extruded && extrusion.edgeCount) {
              sideModel = new Model(device, {
                ...sideShaderProps,
                id: `${this.id}-sides-${chunkIndex}`,
                topology: 'triangle-list',
                vertexCount: 6,
                instanceCount: extrusion.edgeCount,
                isInstanced: true,
                bindings
              });
            }
            if (this.props.extruded && this.props.wireframe && extrusion.edgeCount) {
              wireframeModel = new Model(device, {
                ...wireframeShaderProps,
                id: `${this.id}-wireframe-${chunkIndex}`,
                topology: 'line-strip',
                vertexCount: 4,
                instanceCount: extrusion.edgeCount,
                isInstanced: true,
                bindings: {
                  ...selection.update(this.props.selectionMask, chunkIndex, rowIndexBase),
                  ...extrusion.update(this.getExtrusionProps(chunkIndex, rowIndexBase, true), true),
                  ...low?.getBindings()
                }
              });
            }
            batches.push({
              model,
              sideModel,
              wireframeModel,
              selection,
              extrusion,
              low,
              chunkIndex,
              rowIndexBase
            });
          } catch (error) {
            model?.destroy();
            sideModel?.destroy();
            wireframeModel?.destroy();
            extrusion?.destroy();
            low?.destroy();
            selection.destroy();
            throw error;
          }
        }
        rowIndexBase += chunk.length;
      }
    } catch (error) {
      for (const batch of batches) {
        batch.model?.destroy();
        batch.sideModel?.destroy();
        batch.wireframeModel?.destroy();
        batch.extrusion.destroy();
        batch.low?.destroy();
        batch.selection.destroy();
      }
      throw error;
    }
    this.setState({
      batches,
      retirement: (this.state as GPUSolidPolygonLayerState).retirement,
      boundInputs: [
        this.props.positions,
        this.props.rowIndices,
        this.props.indices,
        this.props.getFillColor,
        this.props.extrusionRings,
        this.props.extruded,
        this.props.filled,
        this.props.wireframe,
        Boolean(this.props.brushingEnabled),
        this.props.positionsLow
      ]
    } satisfies GPUSolidPolygonLayerState);
  }

  /** Owned extrusion topology/control bytes, including allocations awaiting GPU retirement. */
  get extrusionByteLength(): number {
    return (
      ((this.state as GPUSolidPolygonLayerState | undefined)?.retirement.bytes ?? 0) +
      ((this.state as GPUSolidPolygonLayerState | undefined)?.batches.reduce(
        (sum, batch) => sum + batch.extrusion.ownedByteLength,
        0
      ) ?? 0)
    );
  }

  private getExtrusionProps(chunkIndex: number, rowIndexBase: number, wireframe = false) {
    const elevation = isGPUVector(this.props.getElevation)
      ? this.props.getElevation.data[chunkIndex]
      : (this.props.getElevation ?? 1000);
    const colorInput = wireframe ? this.props.getLineColor : this.props.getFillColor;
    const color = isGPUVector(colorInput)
      ? colorInput.data[chunkIndex]
      : (colorInput ?? ([0, 0, 0, 255] as Color));
    const elevationOffset = isGPUVector(this.props.getElevationOffset)
      ? this.props.getElevationOffset.data[chunkIndex]
      : (this.props.getElevationOffset ?? 0);
    return {
      rowIndexBase,
      extruded: Boolean(this.props.extruded),
      elevation: elevation as number | GPUData<'float32'>,
      elevationScale: this.props.elevationScale ?? 1,
      elevationOffset: elevationOffset as number | GPUData<'float32'>,
      color: color as Color | GPUData<'unorm8x4' | VertexList<'unorm8x4'>>,
      windingOrder: this.props.extrusionRings?.windingOrder
    };
  }

  private destroyModel(): void {
    const state = this.state as GPUSolidPolygonLayerState | undefined;
    const batches = state?.batches ?? [];
    if (state) this.setState({batches: [], boundInputs: []});
    if (!batches.length) return;
    const retiredBytes = batches.reduce((sum, batch) => sum + batch.extrusion.ownedByteLength, 0);
    const retirement = state!.retirement;
    retirement.bytes += retiredBytes;
    const fence = this.context.device.createFence();
    const release = () => {
      for (const batch of batches) {
        batch.model?.destroy();
        batch.sideModel?.destroy();
        batch.wireframeModel?.destroy();
        batch.extrusion.destroy();
        batch.low?.destroy();
        batch.selection.destroy();
      }
      retirement.bytes -= retiredBytes;
      fence.destroy();
    };
    void fence.signaled.then(release, release);
  }
}

function isGPUVector(value: unknown): value is GPUVector {
  return Boolean(value && typeof value === 'object' && 'data' in value && 'format' in value);
}
