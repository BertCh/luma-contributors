// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getH3HighlightProps, getH3OutlineWidthScale} from './h3-highlight';
import {PathLayer} from '@deck.gl/layers';
import {isCellLayerType, isPolygonFootprintLayerType} from './cell-layer-types';
import type {ResidentFlowBinding} from './flow-runtime';
import {getBrushingRadiusMeters} from './native-brushing-support';
import {FlowmapGPULayer, FLOWMAP_DATA_ACCESSORS} from './flowmap-gpu-layer';
import {getFlowLayerMeta, hasFlowLayerMetaChanged} from './flow-layer-meta';
import {createResidentFlowFieldLayerClass} from './flow-field-resident';
import {
  getConnectionArrowSize,
  getConnectionElevationScale,
  getConnectionLimitReason
} from './resident-arc-line-plan';
import type {FlowFieldResolution} from './flow-field-bindings';
import {getResidentHoverIdentity, ResidentHoverIdentity} from './resident-hover-identity';
import {renderResidentScenegraph} from './scenegraph-resident-layer';
import {
  NEIGHBOR_HOVER_DIMMED_OPACITY,
  makePointNeighborHoverLayer,
  readPointNeighborPositions,
  type PointNeighborSource
} from './point-neighbor-hover';

import {LayerClasses, type LayerClassesType, type Layer as KeplerLayer} from '@kepler.gl/layers';
import {
  COORDINATE_SYSTEM,
  type Color,
  type Layer,
  type LayerProps,
  type PickingInfo
} from '@deck.gl/core';
import {
  GPUScatterplotLayer,
  GPUMeshIconLayer,
  GPUMeshColumnLayer,
  GPULineLayer,
  GPUArcLayer,
  GPUArrowHeadLayer,
  GPUPathLayer,
  GPUTripMarkerLayer,
  GPUSolidPolygonLayer,
  type GPUPathLayerProps,
  type GPUSolidPolygonLayerProps
} from '@deck.gl-community/gpu-layers';
import type {
  GPUData,
  GPUVector,
  GPUVectorFormat,
  VertexList
} from '@luma-contributors/graph/gpu-data';
import type {NativeStylePlan} from './native-style-plan';
import {planKeplerTripAnimation} from '@luma-contributors/kepler';

// Native LineLayer narrows its return type to its own Deck subclasses. Its plugin override
// may return any Deck layer; widen only that method at this adapter boundary.
interface ResidentLineInstance extends Omit<InstanceType<LayerClassesType['line']>, 'renderLayer'> {
  hasLayerData(
    layerData: Parameters<KeplerLayer['hasLayerData']>[0] | {readonly resident: true}
  ): boolean;
  formatLayerData(...parameters: Parameters<KeplerLayer['formatLayerData']>): object;
  updateLayerMeta(
    ...parameters: Parameters<InstanceType<LayerClassesType['line']>['updateLayerMeta']>
  ): void;
  getHoverData(
    ...parameters: Parameters<KeplerLayer['getHoverData']>
  ): ReturnType<KeplerLayer['getHoverData']>;
  renderLayer(
    ...parameters: Parameters<KeplerLayer['renderLayer']>
  ): ReturnType<KeplerLayer['renderLayer']>;
}
export type ResidentKeplerLayer =
  | InstanceType<
      LayerClassesType[
        | 'point'
        | 'arc'
        | 'geojson'
        | 'trip'
        | 'flow'
        | 'icon'
        | 'hexagonId'
        | 's2'
        | 'a5'
        | 'geohash']
    >
  | ResidentLineInstance;
export type ResidentPickingInfo = PickingInfo & {datasetId: string; sourceRowIndex: number};

/** Multipart primitives have their own mask and picking index space. */
export type ResidentGeometryPrimitiveBinding = {
  selectionMask?: GPUVector<'uint32'>;
  resolveSourceRow?: (primitiveIndex: number) => number;
  /** Inclusive primitive row range owned by one source feature, for whole-feature hover highlight. */
  findPrimitiveRange?: (sourceRow: number) => readonly [number, number] | null;
};
export type ResidentGeometryPointBinding = ResidentGeometryPrimitiveBinding & {
  positions: GPUVector<'float32x2'>;
  /** Retained source XYZ, preferred over XY when the source includes altitude. */
  positions3D?: GPUVector<'float32x3'>;
  colors?: GPUVector<'unorm8x4'>;
  sizes?: GPUVector<'float32'>;
  strokeColors?: GPUVector<'unorm8x4'>;
  lineWidths?: GPUVector<'float32'>;
};
export type ResidentGeometryPathBinding = ResidentGeometryPrimitiveBinding & {
  getPath: GPUPathLayerProps['getPath'];
  /** fp64 low part of XY aligned with `getPath`; present only for sources Float32 cannot hold. */
  getPathLow?: GPUPathLayerProps['getPathLow'];
  getColor?: GPUPathLayerProps['getColor'];
  getWidth?: GPUPathLayerProps['getWidth'];
};
export type ResidentGeometryTripBinding = ResidentGeometryPathBinding & {
  column: string;
  /** Prepared trips are XYZ; scenegraph markers read three words per vertex. */
  getPath: GPUVector<VertexList<'float32x3'>>;
  selectionMask: GPUVector<'uint32'>;
  getTimestamps: NonNullable<GPUPathLayerProps['getTimestamps']>;
  resolveHoverSourceRow?(primitiveIndex: number, currentTimeMilliseconds: number): number;
  /** Grouped tabular trips use native table-mode column roles instead of a GeoJSON column. */
  grouping?: {
    timestampColumn: string;
    longitudeColumn: string;
    latitudeColumn: string;
    altitudeColumn?: string;
  };
  /** Grouped tabular trips: per-vertex marker channels in the path's flattened vertex order. */
  markerChannels?: {
    vertexSelection: GPUVector<'uint32'>;
    colors?: GPUVector<'unorm8x4'>;
    roll?: GPUVector<'float32'>;
    pitch?: GPUVector<'float32'>;
    yaw?: GPUVector<'float32'>;
  };
  /** Why a configured roll, pitch or yaw field has no resident per-vertex values. */
  markerChannelReason?: string;
  originMilliseconds: number;
  maximumTimeErrorMilliseconds: number;
  animationDomainMilliseconds: readonly [number, number] | null;
};
export type ResidentGeometryPolygonBinding = ResidentGeometryPrimitiveBinding &
  Pick<
    GPUSolidPolygonLayerProps,
    | 'positions'
    | 'positionsLow'
    | 'indices'
    | 'rowIndices'
    | 'getFillColor'
    | 'getLineColor'
    | 'getElevation'
    | 'getElevationOffset'
    | 'extrusionRings'
  >;

export type ResidentH3Binding = {
  column: string;
  /** Closed lng/lat boundary of a source row for the native hovered-outline layer. */
  getCellBoundary?: (sourceRow: number) => [number, number][] | undefined;
  /** A current bounded topology summary chooses the native rendering branch. */
  branch: 'empty' | 'polygons' | 'columns';
  polygons?: ResidentGeometryPolygonBinding;
  polygonOutlines?: ResidentGeometryPathBinding;
  columns?: ResidentGeometryPrimitiveBinding & {
    positions: GPUVector<'float32x2'>;
    /** fp64 low residual of `positions` for cells that collapse in Float32. */
    positionsLow?: GPUVector<'float32x2'>;
    vertices: GPUData<'float32x2'>;
    edgeDistance: number;
    colors?: GPUVector<'unorm8x4'>;
    strokeColors?: GPUVector<'unorm8x4'>;
    coverages?: GPUVector<'float32'>;
    elevations?: GPUVector<'float32'>;
  };
};

export type ResidentIconBinding = {
  columns: {lat: string; lng: string; icon: string; altitude?: string};
  groups: readonly {
    iconId: string;
    iconGeometry: GPUData<'float32x3'>;
    positions: GPUVector<'float32x2' | 'float32x3'>;
    selectionMask: GPUVector<'uint32'>;
    colors?: GPUVector<'unorm8x4'>;
    radii?: GPUVector<'float32'>;
    resolveSourceRow: (primitiveIndex: number) => number;
  }[];
};

/** Geometry is uploaded once; explicit primitive maps preserve original source identity. */
export type ResidentDatasetBinding = {
  /** Native `token` column of an S2, A5 or geohash layer; footprints live in `geojson.column`. */
  cellColumn?: string;
  h3?: ResidentH3Binding;
  icon?: ResidentIconBinding;
  trip?: ResidentGeometryTripBinding;
  flow?: ResidentFlowBinding;
  point?: {
    positions: GPUVector<'float32x2' | 'float32x3'>;
    columns: {lat: string; lng: string; altitude?: string};
  };
  connections?: {
    sourcePositions: GPUVector<'float32x2'> | GPUVector<'float32x3'>;
    targetPositions: GPUVector<'float32x2'> | GPUVector<'float32x3'>;
    columns: {
      lat0: string;
      lng0: string;
      lat1: string;
      lng1: string;
      /** Altitude columns folded into float32x3 endpoints (native Line only). */
      alt0?: string;
      alt1?: string;
    };
  };
  geojson?: {
    column: string;
    points?: ResidentGeometryPointBinding;
    paths?: GPUPathLayerProps['getPath'] | ResidentGeometryPathBinding;
    polygons?: ResidentGeometryPolygonBinding;
    polygonOutlines?: GPUPathLayerProps['getPath'] | ResidentGeometryPathBinding;
    fixedRadius?: boolean;
    /** One representative label anchor (longitude, latitude) per source row; absent when unsupported. */
    labelAnchors?: {positions: GPUVector<'float32x2'>; validity: GPUVector<'uint32'>};
  };
  /** Feature types of the source while `geojson` is absent (planar copy released under a globe view). */
  geojsonFeatureTypes?: {line: boolean; polygon: boolean; point: boolean};
  selectionMask: GPUVector<'uint32'>;
  enabled?: boolean;
  unsupportedReason?: string;
  styleRender?: NativeStylePlan['render'];
  bounds?: [number, number, number, number];
  /** Explicit analysis display override, independent of native color-channel configuration. */
  colorOverride?: GPUVector<'unorm8x4'>;
  /** Resolver supplies these only when they represent this layer's current native channels. */
  channels?: {
    colors?: GPUVector<'unorm8x4'>;
    targetColors?: GPUVector<'unorm8x4'>;
    sizes?: GPUVector<'float32'>;
    pointSizes?: GPUVector<'float32'>;
    polygonColors?: GPUSolidPolygonLayerProps['getFillColor'];
    strokeColors?: GPUVector<'unorm8x4'>;
  };
  /** Identity mapping is the default; this callback never scans or materializes source rows. */
  resolveSourceRow?: (featureIndex: number) => number;
  onHover?: (information: ResidentPickingInfo) => void;
  onClick?: (information: ResidentPickingInfo) => void;
};
export type ResidentPointLabelContext = {
  radius: GPUVector<'float32'> | number;
  radiusScale: number;
  radiusMaxPixels: number;
  visible: boolean;
};
export type ResidentLayerOptions = {
  /** Distinguishes not-yet-uploaded resident data from ordinary Kepler datasets. */
  isResidentDataset: (datasetId: string) => boolean;
  resolveDataset: (
    datasetId: string,
    layer: ResidentKeplerLayer
  ) => ResidentDatasetBinding | undefined;
  /** Flow layers in H3 mode resolve a derived center-coordinate runtime for the retained table. */
  resolveFlowH3?: (layer: KeplerLayer) => import('./flow-h3-source').FlowH3Resolution | undefined;
  /** Flow Field layers resolve their GPU particle runtime from the retained source table. */
  resolveFlowField?: (
    layer: KeplerLayer,
    selectionMask?: GPUVector<'uint32'>
  ) => FlowFieldResolution | undefined;
  /** Point labels reuse the already resolved point radius and source selection. */
  resolvePointLabels?: (
    layer:
      | InstanceType<LayerClassesType['point']>
      | InstanceType<LayerClassesType['icon']>
      | InstanceType<LayerClassesType['geojson']>,
    binding: ResidentDatasetBinding,
    renderOptions: ResidentRenderOptions,
    context: ResidentPointLabelContext
  ) => {layers: Layer[]; reason?: string};
  /** Retained source table of a point layer's dataset, for native hover-neighbor display. */
  resolvePointNeighborSource?: (layer: KeplerLayer) => PointNeighborSource | undefined;
  /**
   * Native Trips scenegraph markers: resolves the layer's model, scale and base rotation. While
   * loading the trails still draw; failures are reported. Without it, scenegraph trips are refused.
   */
  resolveTripMarker?: (layer: InstanceType<LayerClassesType['trip']>) => ResidentTripMarkerModel;
  /** null clears a previous diagnostic after the configuration becomes supported. */
  onUnsupported: (layerId: string, reason: string | null) => void;
  /** Physical renderer IDs differ from native IDs; route failures to the captured native layer. */
  onRenderError?: (layer: ResidentKeplerLayer, error: Error) => void;
};
export type ResidentTripMarkerModel =
  | {status: 'loading'}
  | {status: 'failed'; reason: string}
  | {
      status: 'ready';
      /** Post-processed glTF drawn by Deck's scenegraph shaders (textures, PBR, animations). */
      scenegraph: unknown;
      /** Native `model.scale * 2^sizeScale`. */
      sizeScale: number;
      /** Column-major 3x3 native adjust and model rotation. */
      baseRotation: readonly number[];
    };
export type ResidentRenderOptions = {
  idx?: number;
  visible?: boolean;
  layerCallbacks?: {onRedrawNeeded?: InstanceType<LayerClassesType['icon']>['onRedrawNeeded']};
  animationConfig?: {domain?: number[] | null; currentTime?: number | null};
  mapState?: {
    zoom: number;
    zoomOffset?: number;
    longitude?: number;
    latitude?: number;
    scale?: number;
    globe?: {enabled?: boolean};
    /** Present on the live Kepler map state; label collision avoidance needs the viewport size. */
    width?: number;
    height?: number;
    pitch?: number;
    bearing?: number;
    dragRotate?: boolean;
    layerParameters?: LayerProps['parameters'];
  };
  /** Native hover state; the picked layer's source-row resolver identifies the hovered feature. */
  objectHovered?: {index?: number; layer?: {props?: unknown}} | null;
  interactionConfig?: {
    brush?: {enabled?: boolean; config?: {size?: number}};
    tooltip?: {enabled?: boolean};
  };
};
/**
 * Globe under Deck's shared project module: every resident primitive projects through the globe
 * branch and horizon-culls itself. Icons, heatmaps, point labels and aggregation layers keep
 * explicit reasons.
 */
const GLOBE_UNSUPPORTED_REASONS: Record<string, string | undefined> = {};
const RESIDENT_LAYER_DATA = Object.freeze({resident: true as const});
/** Counted across layers so a replaced dataset never reuses a version Deck already saw. */
let residentFlowDataVersion = 0;
type FlowVisConfig = {
  opacity: number;
  darkBaseMapEnabled: boolean;
  colorRange?: {colors: string[]};
  flowFadeAmount: number;
  flowFadeEnabled: boolean;
  flowLocationTotalsEnabled: boolean;
  flowLinesRenderingMode?: 'straight' | 'curved' | 'animated-straight';
  flowLineThicknessScale: number;
  flowLineCurviness: number;
  flowClusteringEnabled: boolean;
  maxTopFlowsDisplayNum: number;
  flowAdaptiveScalesEnabled: boolean;
  /** Beyond native Kepler (which keeps labels off): flowmap.gl location labels and scale lock. */
  flowLocationLabelsEnabled?: boolean;
  flowScaleLockEnabled?: boolean;
};
// Pinned Kepler 3.3 uses this multiplier for pixel-sized line/arc strokes.
const PROJECTED_PIXEL_SIZE_MULTIPLIER = 2 / 3;

/** Native Kepler controls select these classes; their data path never invokes CPU accessors. */
export function createResidentLayerClasses(options: ResidentLayerOptions): LayerClassesType {
  const diagnostics = new Map<string, string | null>();
  function report(layer: KeplerLayer, reason: string | null): void {
    if (diagnostics.get(layer.id) === reason) return;
    diagnostics.set(layer.id, reason);
    options.onUnsupported(layer.id, reason);
  }
  function isResident(layer: KeplerLayer): boolean {
    return Boolean(layer.config.dataId && options.isResidentDataset(layer.config.dataId));
  }
  function resolve(layer: ResidentKeplerLayer): ResidentDatasetBinding | undefined {
    return layer.config.dataId ? options.resolveDataset(layer.config.dataId, layer) : undefined;
  }
  function updateMetadata(layer: ResidentKeplerLayer): void {
    const binding = resolve(layer);
    if (binding?.bounds) layer.meta = {...layer.meta, bounds: binding.bounds};
    // H3 flows publish metadata from the derived center runtime once it is ready.
    const flowBinding =
      layer.type === 'flow' && (layer.config as {columnMode?: string}).columnMode === 'H3'
        ? options.resolveFlowH3?.(layer)?.binding
        : binding?.flow;
    if (flowBinding)
      layer.meta = {...layer.meta, ...getFlowLayerMeta(flowBinding, binding?.bounds)};
    if (binding?.geojsonFeatureTypes && !binding.geojson)
      layer.meta = {...layer.meta, featureTypes: binding.geojsonFeatureTypes};
    if (binding?.geojson)
      layer.meta = {
        ...layer.meta,
        featureTypes: {
          line: Boolean(binding.geojson.paths),
          polygon: Boolean(binding.geojson.polygons),
          point: Boolean(binding.geojson.points)
        }
      };
  }
  function prepare(
    layer: ResidentKeplerLayer,
    renderOptions: ResidentRenderOptions
  ): ResidentDatasetBinding | undefined {
    const binding = resolve(layer);
    if (binding?.unsupportedReason) {
      report(layer, binding.unsupportedReason);
      return undefined;
    }
    if (!binding || binding.enabled === false || !layer.config.isVisible) return undefined;
    const config = layer.config;
    const reason =
      renderOptions.mapState?.globe?.enabled && GLOBE_UNSUPPORTED_REASONS[layer.type]
        ? GLOBE_UNSUPPORTED_REASONS[layer.type]
        : config.textLabel?.some(label => label.field) &&
            ((layer.type !== 'point' && layer.type !== 'icon' && layer.type !== 'geojson') ||
              !options.resolvePointLabels)
          ? 'Text labels require a resident text binding.'
          : !isPolygonFootprintLayerType(layer.type) &&
              layer.type !== 'trip' &&
              layer.type !== 'icon' &&
              layer.type !== 'hexagonId' &&
              'colorField' in config &&
              config.colorField &&
              (layer.type !== 'point' ||
                ('filled' in config.visConfig && config.visConfig.filled)) &&
              !binding.channels?.colors &&
              !binding.channels?.polygonColors &&
              !binding.colorOverride
            ? 'The selected color channel has no resident GPU result.'
            : !isPolygonFootprintLayerType(layer.type) &&
                layer.type !== 'trip' &&
                layer.type !== 'icon' &&
                layer.type !== 'hexagonId' &&
                'sizeField' in config &&
                config.sizeField &&
                !binding.channels?.sizes
              ? 'The selected size channel has no resident GPU result.'
              : null;
    if (reason) {
      report(layer, reason);
      return undefined;
    }
    return binding;
  }
  /** Native brush size is kilometers; render-only like Deck's BrushingExtension. */
  function getBrushingProps(renderOptions: ResidentRenderOptions) {
    const brush = renderOptions.interactionConfig?.brush;
    return {
      brushingEnabled: Boolean(brush?.enabled),
      brushingRadius: getBrushingRadiusMeters(brush)
    };
  }
  /** Source row of the native hover target when it was picked from this layer's primitives. */
  function getHoveredSourceRow(
    layer: ResidentKeplerLayer,
    renderOptions: ResidentRenderOptions
  ): number | undefined {
    const hovered = renderOptions.objectHovered;
    const properties = hovered?.layer?.props;
    if (
      !hovered ||
      typeof hovered.index !== 'number' ||
      hovered.index < 0 ||
      !properties ||
      typeof properties !== 'object' ||
      !('nativeLayerId' in properties) ||
      properties.nativeLayerId !== layer.id
    )
      return undefined;
    const resolve = 'nativeSourceRowIndex' in properties ? properties.nativeSourceRowIndex : null;
    return typeof resolve === 'function' ? resolve(hovered.index) : hovered.index;
  }
  function getHighlightedRange(
    layer: ResidentKeplerLayer,
    renderOptions: ResidentRenderOptions,
    primitive?: ResidentGeometryPrimitiveBinding
  ): readonly [number, number] | null {
    if (!primitive?.findPrimitiveRange || renderOptions.interactionConfig?.brush?.enabled)
      return null;
    const sourceRow = getHoveredSourceRow(layer, renderOptions);
    return sourceRow === undefined ? null : primitive.findPrimitiveRange(sourceRow);
  }
  function baseProps(
    layer: ResidentKeplerLayer,
    binding: ResidentDatasetBinding,
    renderOptions: ResidentRenderOptions,
    primitive?: ResidentGeometryPrimitiveBinding
  ): LayerProps & {
    idx?: number;
    nativeLayerId: string;
    nativeSourceRowIndex?: (primitiveIndex: number) => number;
    highlightedPrimitiveRange: readonly [number, number] | null;
  } {
    const notify =
      (kind: 'onHover' | 'onClick') =>
      (information: PickingInfo): boolean => {
        const sourceRowIndex =
          information.index < 0
            ? -1
            : ((primitive?.resolveSourceRow ?? binding.resolveSourceRow)?.(information.index) ??
              information.index);
        binding[kind]?.({...information, datasetId: layer.config.dataId!, sourceRowIndex});
        // Native hover state carries identity only; its popover performs the bounded read.
        return false;
      };
    return {
      // Deck transfers state by ID even across layer constructors. Keep native type changes
      // from transferring a scatterplot model into a line/arc renderer.
      id: `${layer.id}-resident-${layer.type}`,
      idx: renderOptions.idx,
      nativeLayerId: layer.id,
      nativeSourceRowIndex: primitive?.resolveSourceRow ?? binding.resolveSourceRow,
      visible: layer.config.isVisible && renderOptions.visible !== false,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      parameters: {
        depthCompare: 'less-equal',
        depthWriteEnabled: Boolean(renderOptions.mapState?.dragRotate),
        ...renderOptions.mapState?.layerParameters
      },
      opacity: layer.config.visConfig.opacity,
      // Native layers highlight the hovered object unless brushing owns the pointer.
      autoHighlight: !renderOptions.interactionConfig?.brush?.enabled,
      highlightColor: layer.config.highlightColor,
      // Features with several primitives highlight together; others keep the single picked id.
      highlightedPrimitiveRange: getHighlightedRange(layer, renderOptions, primitive),
      pickable:
        renderOptions.interactionConfig?.tooltip?.enabled !== false &&
        (!('allowHover' in layer.config.visConfig) || layer.config.visConfig.allowHover !== false),
      onHover: notify('onHover'),
      onClick: notify('onClick'),
      onError: error => {
        report(layer, error.message);
        options.onRenderError?.(layer, error);
        return Boolean(options.onRenderError);
      }
    };
  }
  function point(
    layer: InstanceType<LayerClassesType['point']>,
    renderOptions: ResidentRenderOptions
  ): Layer[] {
    const binding = prepare(layer, renderOptions);
    if (!binding) return [];
    const config = layer.config;
    const geometry = binding.point;
    const reason = !geometry
      ? 'No resident point geometry is bound to this dataset.'
      : !matchesColumns(config.columns, geometry.columns)
        ? 'The selected point columns do not match uploaded geometry.'
        : config.columnMode !== 'points'
          ? 'This point column mode has no resident binding.'
          : config.columns.altitude?.value &&
              (!geometry.columns.altitude || geometry.positions.format !== 'float32x3')
            ? 'Point altitude requires matching resident XYZ positions.'
            : config.visConfig.outline && config.strokeColorField && !binding.channels?.strokeColors
              ? 'The selected point stroke channel has no resident GPU result.'
              : null;
    report(layer, reason);
    if (reason || !geometry) return [];
    const radius = binding.channels?.sizes ?? 1;
    const radiusScale =
      binding.styleRender?.radiusScale ??
      (config.visConfig.fixedRadius && config.sizeField
        ? 1
        : (config.sizeField ? 1 : config.visConfig.radius) *
          layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14}));
    const radiusMaxPixels =
      binding.styleRender?.radiusMaxPixels ??
      (config.visConfig.fixedRadius ? Number.MAX_SAFE_INTEGER : 500);
    // Resolve empty configurations too, so the owner retires removed label resources.
    const labels = options.resolvePointLabels?.(layer, binding, renderOptions, {
      radius,
      radiusScale,
      radiusMaxPixels,
      visible: config.isVisible && renderOptions.visible !== false
    });
    if (labels?.reason) {
      report(layer, labels.reason);
      return [];
    }
    const renderProperties = baseProps(layer, binding, renderOptions);
    const pointVisible = config.visConfig.filled || config.visConfig.outline;
    const hoveredRow =
      config.visConfig.showNeighborOnHover &&
      renderProperties.pickable &&
      !renderOptions.interactionConfig?.brush?.enabled
        ? getHoveredSourceRow(layer, renderOptions)
        : undefined;
    let neighborHover: Layer[] = [];
    if (hoveredRow !== undefined && hoveredRow >= 0) {
      const source = options.resolvePointNeighborSource?.(layer);
      let positions: [number, number, number][] = [];
      try {
        positions = source
          ? readPointNeighborPositions(source, hoveredRow, {
              ...geometry.columns,
              neighbors: config.columns.neighbors?.value
            })
          : [];
      } catch {
        // A lazily materialized source may not hold the hovered row yet; native shows no overlay.
      }
      if (positions.length)
        neighborHover = [
          makePointNeighborHoverLayer({
            id: renderProperties.id,
            positions,
            highlightColor: layer.config.highlightColor as Color,
            showHighlightColor: Boolean(config.visConfig.showHighlightColor),
            layerColor: config.color as Color,
            strokeColor: config.visConfig.strokeColor as Color | undefined,
            filled: config.visConfig.filled,
            stroked: config.visConfig.outline,
            thickness: config.visConfig.thickness,
            billboard: config.visConfig.billboard,
            radiusScale,
            radiusMaxPixels,
            visible: Boolean(renderProperties.visible)
          })
        ];
    }
    const points: Layer[] = pointVisible
      ? [
          new GPUScatterplotLayer({
            ...renderProperties,
            ...(neighborHover.length ? {opacity: NEIGHBOR_HOVER_DIMMED_OPACITY} : {}),
            ...getBrushingProps(renderOptions),
            // Native points enable depth testing only for an assigned altitude column.
            parameters: {
              depthCompare: config.columns.altitude?.value ? 'less-equal' : 'always',
              depthWriteEnabled: Boolean(config.columns.altitude?.value),
              ...renderOptions.mapState?.layerParameters
            },
            selectionMask: binding.selectionMask,
            getPosition: geometry.positions,
            getFillColor: normalizeResidentColor(
              binding.colorOverride ?? binding.channels?.colors ?? config.color
            ),
            getRadius: radius,
            getLineColor: normalizeResidentColor(
              binding.channels?.strokeColors ?? config.visConfig.strokeColor ?? config.color
            ),
            getLineWidth: 1,
            lineWidthUnits: 'pixels',
            lineWidthScale: config.visConfig.thickness,
            stroked: config.visConfig.outline,
            filled: config.visConfig.filled,
            radiusUnits: 'meters',
            billboard: config.visConfig.billboard,
            radiusScale,
            radiusMaxPixels
          })
        ]
      : [];
    return [
      ...points,
      ...neighborHover,
      ...(labels?.layers ?? []).map(label =>
        label.clone({
          visible: renderProperties.visible,
          pickable: false,
          onError: renderProperties.onError,
          // Native passes the same brush to its text-label layers, hiding labels with their anchor.
          ...getBrushingProps(renderOptions),
          ...{nativeLayerId: layer.id}
        })
      )
    ];
  }
  function icon(
    layer: InstanceType<LayerClassesType['icon']>,
    renderOptions: ResidentRenderOptions
  ): Layer[] {
    const binding = prepare(layer, renderOptions);
    if (!binding) return [];
    const config = layer.config;
    const geometry = binding.icon;
    const hasColorField = 'colorField' in config && Boolean(config.colorField);
    const hasSizeField = 'sizeField' in config && Boolean(config.sizeField);
    const reason = !geometry
      ? 'No resident icon geometry is bound to this dataset.'
      : !matchesColumns(config.columns, geometry.columns)
        ? 'The selected icon columns do not match uploaded geometry.'
        : config.columns.altitude?.value &&
            (!geometry.columns.altitude ||
              geometry.groups.some(group => group.positions.format !== 'float32x3'))
          ? 'Icon altitude requires matching resident XYZ positions.'
          : (hasColorField || binding.colorOverride) && geometry.groups.some(group => !group.colors)
            ? 'The selected icon color channel has no gathered GPU result.'
            : hasSizeField && geometry.groups.some(group => !group.radii)
              ? 'The selected icon radius channel has no gathered GPU result.'
              : null;
    report(layer, reason);
    if (reason || !geometry) return [];
    // Icon labels anchor on source-aligned positions with the icon layer's source selection.
    const iconRadiusScale =
      binding.styleRender?.radiusScale ??
      (config.visConfig.fixedRadius
        ? 1
        : (hasSizeField ? 1 : config.visConfig.radius) *
          layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14}));
    // Native pads icon labels by the per-row radius channel (getTextOffsetByRadius(getRadius)).
    const iconLabels = options.resolvePointLabels?.(layer, binding, renderOptions, {
      radius: hasSizeField ? (binding.channels?.sizes ?? 1) : 1,
      radiusScale: iconRadiusScale,
      radiusMaxPixels: config.visConfig.fixedRadius ? Number.MAX_SAFE_INTEGER : 500,
      visible: config.isVisible && renderOptions.visible !== false
    });
    if (iconLabels?.reason) {
      report(layer, iconLabels.reason);
      return [];
    }
    const iconErrorHandler = baseProps(layer, binding, renderOptions).onError;
    const iconLabelLayers = (iconLabels?.layers ?? []).map(label =>
      label.clone({
        visible: layer.config.isVisible && renderOptions.visible !== false,
        pickable: false,
        onError: iconErrorHandler,
        ...getBrushingProps(renderOptions),
        ...{nativeLayerId: layer.id}
      })
    );
    const iconMeshLayers = geometry.groups.map(group => {
      const properties = baseProps(layer, binding, renderOptions, group);
      return new GPUMeshIconLayer({
        ...properties,
        ...getBrushingProps(renderOptions),
        id: `${properties.id}-${encodeURIComponent(group.iconId)}`,
        iconGeometry: group.iconGeometry,
        getPosition: group.positions,
        selectionMask: group.selectionMask,
        getFillColor: normalizeResidentColor(group.colors ?? config.color),
        getRadius: group.radii ?? 1,
        radiusUnits: 'meters',
        billboard: config.visConfig.billboard,
        radiusScale:
          binding.styleRender?.radiusScale ??
          (config.visConfig.fixedRadius
            ? 1
            : (hasSizeField ? 1 : config.visConfig.radius) *
              layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14})),
        radiusMaxPixels:
          binding.styleRender?.radiusMaxPixels ??
          (config.visConfig.fixedRadius ? Number.MAX_SAFE_INTEGER : 500)
      });
    });
    return [...iconMeshLayers, ...iconLabelLayers];
  }
  function hexagonId(
    layer: InstanceType<LayerClassesType['hexagonId']>,
    renderOptions: ResidentRenderOptions
  ): Layer[] {
    const binding = prepare(layer, renderOptions);
    if (!binding) return [];
    const config = layer.config;
    const geometry = binding.h3;
    // Pinned native runtime exposes strokeColor although its declaration omits it.
    const visConfig: typeof config.visConfig & {strokeColor?: Color} = config.visConfig;
    const extruded = binding.styleRender?.extruded ?? Boolean(config.visConfig.enable3d);
    const filled = binding.styleRender?.filled ?? config.visConfig.filled !== false;
    const stroked =
      !extruded && (binding.styleRender?.stroked ?? Boolean(config.visConfig.outline));
    const reason = !geometry
      ? 'No resident H3 geometry is bound to this dataset.'
      : config.columns.hex_id?.value !== geometry.column
        ? 'The selected H3 column does not match uploaded geometry.'
        : geometry.branch === 'columns' && !geometry.columns
          ? 'The current H3 column footprint is not ready.'
          : geometry.branch === 'polygons' && !geometry.polygons
            ? 'The current H3 polygon geometry is not ready.'
            : extruded && geometry.branch === 'polygons' && !geometry.polygons?.extrusionRings
              ? 'H3 extrusion requires resident boundary rings.'
              : null;
    report(layer, reason);
    if (reason || !geometry || geometry.branch === 'empty' || (!filled && !stroked)) return [];
    const elevationScale =
      binding.styleRender?.elevationScale ??
      (config.visConfig.elevationScale ?? 5) *
        (config.visConfig.fixedHeight || config.visConfig.enableElevationZoomFactor === false
          ? 1
          : 2 **
            Math.max(
              8 - (renderOptions.mapState?.zoom ?? 14) + (renderOptions.mapState?.zoomOffset ?? 0),
              0
            ));
    const widthScale = binding.styleRender?.widthScale ?? config.visConfig.thickness ?? 2;
    const parameters = {
      depthCompare: 'less-equal' as const,
      depthWriteEnabled: Boolean(renderOptions.mapState?.dragRotate || extruded),
      ...renderOptions.mapState?.layerParameters
    };
    // Native: a GeoJsonLayer outline of the hovered cell (highlight color) unless a size field is set.
    const hovered = renderOptions.objectHovered as
      | {index?: number; picked?: boolean; layer?: {props?: unknown}}
      | null
      | undefined;
    const hoveredProps = hovered?.layer?.props as
      | {nativeLayerId?: string; nativeSourceRowIndex?: (index: number) => number}
      | undefined;
    const hoveredBoundary =
      hovered?.picked !== false &&
      hoveredProps?.nativeLayerId === layer.id &&
      typeof hovered?.index === 'number' &&
      hovered.index >= 0
        ? geometry.getCellBoundary?.(
            hoveredProps.nativeSourceRowIndex?.(hovered.index) ?? hovered.index
          )
        : undefined;
    const hoveredOutline =
      hoveredBoundary && !config.sizeField
        ? new PathLayer({
            id: `${layer.id}-hovered`,
            data: [{path: hoveredBoundary}],
            getPath: (item: {path: [number, number][]}) => item.path,
            getColor: config.highlightColor as [number, number, number],
            getWidth: 1,
            widthUnits: 'meters',
            widthScale: getH3OutlineWidthScale(
              layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14})
            ),
            pickable: false,
            wrapLongitude: false,
            visible: layer.config.isVisible && renderOptions.visible !== false,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT
          })
        : undefined;
    if (geometry.branch === 'columns' && geometry.columns) {
      const columns = geometry.columns;
      const properties = baseProps(layer, binding, renderOptions, columns);
      return [
        new GPUMeshColumnLayer({
          ...properties,
          ...getBrushingProps(renderOptions),
          ...getH3HighlightProps(
            extruded,
            Boolean(renderOptions.interactionConfig?.brush?.enabled)
          ),
          id: `${properties.id}-columns`,
          parameters,
          flatShading: true,
          vertices: columns.vertices,
          edgeDistance: columns.edgeDistance,
          getPosition: columns.positions,
          getPositionLow: columns.positionsLow,
          selectionMask: columns.selectionMask,
          getFillColor: normalizeResidentColor(columns.colors ?? config.color),
          getLineColor: normalizeResidentColor(
            columns.strokeColors ?? visConfig.strokeColor ?? config.color
          ),
          getCoverage: columns.coverages ?? 1,
          coverage:
            binding.styleRender?.coverage ??
            (config.coverageField ? 1 : (config.visConfig.coverage ?? 1)),
          getElevation: columns.elevations ?? 500,
          elevationScale,
          getLineWidth: 1,
          lineWidthUnits: 'meters',
          lineWidthScale: widthScale,
          strokeOpacity:
            binding.styleRender?.strokeOpacity ?? config.visConfig.strokeOpacity ?? 0.8,
          filled,
          stroked,
          extruded
        }),
        ...(hoveredOutline ? [hoveredOutline] : [])
      ];
    }
    const rendered: Layer[] = [];
    if (filled && geometry.polygons) {
      const polygons = geometry.polygons;
      const properties = baseProps(layer, binding, renderOptions, polygons);
      rendered.push(
        new GPUSolidPolygonLayer({
          ...properties,
          ...polygons,
          // Native polygon sublayers brush per vertex with the same shared radius.
          ...getBrushingProps(renderOptions),
          ...getH3HighlightProps(
            extruded,
            Boolean(renderOptions.interactionConfig?.brush?.enabled)
          ),
          id: `${properties.id}-polygons`,
          parameters,
          getFillColor: normalizeResidentColor(polygons.getFillColor ?? config.color),
          getElevation: polygons.getElevation ?? 500,
          elevationScale,
          extruded
        })
      );
    }
    if (stroked && geometry.polygonOutlines) {
      const outlines = geometry.polygonOutlines;
      const properties = baseProps(layer, binding, renderOptions, outlines);
      rendered.push(
        new GPUPathLayer({
          ...properties,
          ...outlines,
          ...getBrushingProps(renderOptions),
          id: `${properties.id}-polygon-outlines`,
          getColor: normalizeResidentColor(
            outlines.getColor ?? visConfig.strokeColor ?? config.color
          ),
          getWidth: 1,
          widthUnits: 'meters',
          widthScale,
          // Native high-precision H3 uses PolygonLayer's fill opacity for its outlines.
          opacity: binding.styleRender?.opacity ?? config.visConfig.opacity
        })
      );
    }
    if (hoveredOutline) rendered.push(hoveredOutline);
    return rendered;
  }
  function connection(
    layer: ResidentLineInstance | InstanceType<LayerClassesType['arc']>,
    renderOptions: ResidentRenderOptions
  ): Layer[] {
    const binding = prepare(layer, renderOptions);
    if (!binding) return [];
    const config = layer.config;
    const geometry = binding.connections;
    const line = layer.type === 'line';
    const reason = !geometry
      ? 'No resident source/target geometry is bound to this dataset.'
      : (getConnectionLimitReason(line ? 'line' : 'arc', config as never, {
          alt0: geometry.columns.alt0,
          alt1: geometry.columns.alt1
        }) ??
        // Unsupported column modes (neighbors) name their specific reason before a column mismatch.
        (!matchesColumns(config.columns, geometry.columns)
          ? 'The selected connection columns do not match uploaded geometry.'
          : null));
    report(layer, reason);
    if (reason || !geometry) return [];
    const globe = Boolean(renderOptions.mapState?.globe?.enabled);
    // Native lines scale altitude by elevationScale on maps; globe lines ignore it.
    const elevationScale = getConnectionElevationScale(
      (config.visConfig as {elevationScale?: unknown}).elevationScale,
      globe
    );
    const props = {
      ...baseProps(layer, binding, renderOptions),
      ...getBrushingProps(renderOptions),
      selectionMask: binding.selectionMask,
      getSourcePosition: geometry.sourcePositions,
      getTargetPosition: geometry.targetPositions,
      getWidth: binding.channels?.sizes ?? 1,
      widthScale:
        binding.styleRender?.widthScale ??
        config.visConfig.thickness * PROJECTED_PIXEL_SIZE_MULTIPLIER
    };
    const color = normalizeResidentColor(
      binding.colorOverride ?? binding.channels?.colors ?? config.color
    );
    const targetColor = normalizeResidentColor(
      binding.colorOverride ??
        binding.channels?.targetColors ??
        binding.channels?.colors ??
        config.visConfig.targetColor ??
        config.color
    );
    // Native lines on a globe are zero-height great-circle arcs.
    const stroke =
      line && !globe
        ? new GPULineLayer({...props, getColor: color, getTargetColor: targetColor, elevationScale})
        : new GPUArcLayer({
            ...props,
            arcMode: 'deck',
            getHeight: line ? 0 : 1,
            getSourceColor: color,
            getTargetColor: targetColor
          });
    if (!config.visConfig.targetArrowEnabled) return [stroke];
    // Native heads use the target color, are sized in pixels and orient to the final segment.
    const {targetArrowShape, targetArrowSize} = config.visConfig;
    return [
      stroke,
      new GPUArrowHeadLayer({
        ...baseProps(layer, binding, renderOptions),
        ...getBrushingProps(renderOptions),
        id: `${layer.id}-resident-${layer.type}-arrow-head`,
        parameters: {...props.parameters, depthWriteEnabled: false, cullMode: 'none'},
        selectionMask: binding.selectionMask,
        getSourcePosition: geometry.sourcePositions,
        getTargetPosition: geometry.targetPositions,
        getColor: targetColor,
        shape:
          targetArrowShape === 'triangle' || targetArrowShape === 'stealth'
            ? targetArrowShape
            : 'chevron',
        sizePixels: getConnectionArrowSize(targetArrowSize),
        // Globe lines are zero-height great circles, so their heads follow the arc profile.
        mode: line && !globe ? 'line' : 'arc',
        arcHeight: line ? 0 : 1,
        elevationScale
      })
    ];
  }
  function trip(
    layer: InstanceType<LayerClassesType['trip']>,
    renderOptions: ResidentRenderOptions
  ): Layer[] {
    const binding = prepare(layer, renderOptions);
    if (!binding) return [];
    const geometry = binding.trip;
    const config = layer.config;
    const grouping = geometry?.grouping;
    const tableColumns: Record<string, {value?: string | null} | undefined> = config.columns;
    const columnReason =
      (config.columnMode === 'table') !== Boolean(grouping)
        ? 'The trip column mode does not match the prepared source.'
        : grouping
          ? tableColumns['id']?.value !== geometry?.column ||
            tableColumns['timestamp']?.value !== grouping.timestampColumn ||
            tableColumns['lng']?.value !== grouping.longitudeColumn ||
            tableColumns['lat']?.value !== grouping.latitudeColumn ||
            (tableColumns['altitude']?.value ?? undefined) !== grouping.altitudeColumn
            ? 'The selected trip table columns do not match the prepared source.'
            : null
          : config.columns.geojson?.value !== geometry?.column
            ? 'The selected trip column does not match the prepared source.'
            : null;
    const reason = !geometry
      ? 'No prepared resident timed trip geometry is bound to this dataset.'
      : (columnReason ??
        ((config.colorField || binding.colorOverride) && !geometry.getColor
          ? 'Trip colors require a result expanded to valid trip primitives.'
          : 'sizeField' in config && config.sizeField && !geometry.getWidth
            ? 'Trip widths require a result expanded to valid trip primitives.'
            : null));
    report(layer, reason);
    if (reason || !geometry) return [];
    const animation = renderOptions.animationConfig;
    if (
      !animation ||
      !Array.isArray(animation.domain) ||
      animation.domain.length !== 2 ||
      typeof animation.currentTime !== 'number'
    )
      return [];
    const planned = planKeplerTripAnimation(geometry, {
      currentTimeMilliseconds: animation.currentTime,
      domainMilliseconds: [animation.domain[0], animation.domain[1]],
      trailLengthSeconds: config.visConfig.trailLength,
      fadeTrail: config.visConfig.fadeTrail,
      scenegraphEnabled: config.visConfig.scenegraphEnabled && !options.resolveTripMarker,
      columnMode: grouping ? 'table' : 'geojson'
    });
    if (!planned.supported) {
      report(layer, planned.reason);
      return [];
    }
    const props = baseProps(layer, binding, renderOptions, geometry);
    const trailLayer = new GPUPathLayer({
      ...props,
      parameters: {...props.parameters, depthWriteEnabled: false},
      getPath: geometry.getPath,
      getTimestamps: geometry.getTimestamps,
      timestampMode: 'trips',
      selectionMask: geometry.selectionMask,
      getColor: normalizeResidentColor(geometry.getColor ?? config.color),
      getWidth: geometry.getWidth ?? 1,
      widthUnits: 'meters',
      billboard: config.visConfig.billboard,
      capRounded: true,
      jointRounded: true,
      widthScale:
        binding.styleRender?.widthScale ??
        config.visConfig.thickness *
          layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14}) *
          8 *
          (config.visConfig.billboard ? PROJECTED_PIXEL_SIZE_MULTIPLIER : 1),
      currentTime: planned.currentTime,
      trailLength: planned.trailLength,
      fadeTrail: planned.fadeTrail
    });
    if (!config.visConfig.scenegraphEnabled) return [trailLayer];
    const marker = options.resolveTripMarker!(layer);
    const channels = geometry.markerChannels;
    // Native datums exist only for table-mode vertices; GeoJSON trips have none, so their
    // roll and pitch stay zero and the heading follows the speed vector.
    const fields = grouping
      ? ([
          ['roll', config.rollField],
          ['pitch', config.pitchField],
          ['yaw', config.yawField]
        ] as const)
      : [];
    const missingField = fields.find(([name, field]) => field && !channels?.[name]);
    const trailColorPerVertex = Boolean(
      grouping && config.colorField && config.visConfig.scenegraphUseTrailColor
    );
    const markerReason = missingField
      ? (geometry.markerChannelReason ??
        `Trip ${missingField[0]} field ${missingField[1]?.name} has no resident Float32 column.`)
      : trailColorPerVertex && !channels?.colors
        ? 'Table-mode scenegraph trail colors need the resident per-vertex color result.'
        : marker.status === 'failed'
          ? marker.reason
          : null;
    report(layer, markerReason);
    if (markerReason) return [trailLayer];
    if (marker.status !== 'ready') return [trailLayer];
    const {visConfig} = config;
    const tripColors =
      geometry.getColor && 'data' in geometry.getColor && geometry.getColor.format === 'unorm8x4'
        ? (geometry.getColor as GPUVector<'unorm8x4'>)
        : undefined;
    const tint = visConfig.scenegraphColorEnabled
      ? visConfig.scenegraphUseTrailColor
        ? (tripColors ?? config.color)
        : visConfig.scenegraphColor
      : undefined;
    return [
      trailLayer,
      new GPUTripMarkerLayer({
        ...props,
        // Native scenegraphs draw opaque with depth testing.
        parameters: {...props.parameters, depthCompare: 'less-equal', depthWriteEnabled: true},
        id: `${props.id}-scenegraph`,
        opacity: 1,
        getPath: geometry.getPath,
        getTimestamps: geometry.getTimestamps,
        selectionMask: geometry.selectionMask,
        currentTime: planned.currentTime,
        scenegraph: marker.scenegraph,
        sizeScale: marker.sizeScale,
        baseRotation: marker.baseRotation,
        getColor: tint ? normalizeResidentColor(tint) : undefined,
        getVertexColor: trailColorPerVertex ? channels?.colors : undefined,
        vertexSelection: channels?.vertexSelection,
        getVertexRoll: grouping && config.rollField ? channels?.roll : undefined,
        getVertexPitch: grouping && config.pitchField ? channels?.pitch : undefined,
        getVertexYaw: grouping && config.yawField ? channels?.yaw : undefined,
        invertRoll: config.visConfig.invertRoll,
        invertPitch: config.visConfig.invertPitch,
        invertYaw: config.visConfig.invertYaw
        // Lighting and animations keep the layer defaults: native 'pbr' and speed 5 for all.
      })
    ];
  }
  function geojson(
    layer: InstanceType<LayerClassesType['geojson']>,
    renderOptions: ResidentRenderOptions
  ): Layer[] {
    const binding = prepare(layer, renderOptions);
    if (!binding) return [];
    const config = layer.config;
    const geometry = binding.geojson;
    const offsetActive =
      binding.styleRender?.elevationOffsetActive ??
      Boolean(config.elevationOffsetField || config.visConfig.elevationOffset > 0);
    const extruded =
      binding.styleRender?.extruded ?? Boolean(config.visConfig.enable3d || offsetActive);
    const polygonStroked = config.visConfig.stroked && !extruded;
    const pointStroked = config.visConfig.stroked && !offsetActive;
    const wireframe = Boolean(config.visConfig.wireframe && extruded);
    const normalizePath = (
      path: GPUPathLayerProps['getPath'] | ResidentGeometryPathBinding | undefined
    ): ResidentGeometryPathBinding | undefined => {
      if (!path) return undefined;
      if ('getPath' in path) return path;
      // Legacy bindings have one path per source feature; multipart descriptors must
      // provide gathered channels explicitly instead of borrowing source-row styles.
      return {
        getPath: path,
        getColor: binding.channels?.strokeColors,
        getWidth: binding.channels?.sizes
      };
    };
    const paths = normalizePath(geometry?.paths);
    const outlines = normalizePath(geometry?.polygonOutlines);
    const polygonColors = geometry?.polygons?.getFillColor ?? binding.channels?.polygonColors;
    const drawnPaths = [paths, polygonStroked ? outlines : undefined].filter(
      (path): path is ResidentGeometryPathBinding => Boolean(path)
    );
    const reason = !geometry
      ? 'No resident GeoJSON geometry is bound to this dataset.'
      : (
            isCellLayerType(layer.type)
              ? (config.columns as Record<string, {value: string | null} | undefined>)['token']
                  ?.value !== binding.cellColumn
              : config.columns.geojson?.value !== geometry.column
          )
        ? 'The selected GeoJSON column does not match uploaded geometry.'
        : extruded && geometry.polygons && !geometry.polygons.extrusionRings
          ? 'Polygon extrusion requires resident boundary rings.'
          : config.visConfig.enable3d &&
              config.heightField &&
              geometry.polygons &&
              geometry.polygons.getElevation === undefined
            ? 'The selected height channel has no expanded resident GPU result.'
            : config.elevationOffsetField &&
                geometry.polygons &&
                geometry.polygons.getElevationOffset === undefined
              ? 'The selected elevation offset channel has no expanded resident GPU result.'
              : config.strokeColorField &&
                  (drawnPaths.some(path => !path.getColor) ||
                    (pointStroked && geometry.points && !geometry.points.strokeColors) ||
                    (wireframe &&
                      config.visConfig.stroked &&
                      geometry.polygons &&
                      !geometry.polygons.getLineColor))
                ? 'The selected stroke color channel has no expanded resident GPU result.'
                : config.sizeField &&
                    (drawnPaths.some(path => !path.getWidth) ||
                      (pointStroked && geometry.points && !geometry.points.lineWidths))
                  ? 'The selected width channel has no expanded resident GPU result.'
                  : (config.colorField || binding.colorOverride) &&
                      config.visConfig.filled &&
                      geometry.polygons &&
                      !polygonColors
                    ? 'Polygon colors require a result expanded to tessellated vertices.'
                    : (config.colorField || binding.colorOverride) &&
                        config.visConfig.filled &&
                        geometry.points &&
                        !geometry.points.colors
                      ? 'Point colors require a result expanded to geometry components.'
                      : config.radiusField && geometry.points && !geometry.points.sizes
                        ? 'The selected radius channel has no expanded resident GPU result.'
                        : polygonStroked && geometry.polygons && !outlines
                          ? 'Polygon outlines were not uploaded for this dataset.'
                          : null;
    report(layer, reason);
    if (reason || !geometry) return [];
    const layers: Layer[] = [];
    const appendPolygons = (): void => {
      if (!geometry.polygons || (!config.visConfig.filled && !wireframe)) return;
      const polygons = geometry.polygons;
      const properties = baseProps(layer, binding, renderOptions, polygons);
      layers.push(
        new GPUSolidPolygonLayer({
          ...properties,
          id: `${layer.id}-fill`,
          parameters: {
            ...properties.parameters,
            depthWriteEnabled: Boolean(
              renderOptions.mapState?.dragRotate || config.visConfig.enable3d
            ),
            ...renderOptions.mapState?.layerParameters,
            ...(offsetActive ? {depthWriteEnabled: true} : {})
          },
          positions: polygons.positions,
          positionsLow: polygons.positionsLow,
          indices: polygons.indices,
          rowIndices: polygons.rowIndices,
          selectionMask: polygons.selectionMask ?? binding.selectionMask,
          getFillColor: normalizeResidentColor(polygonColors ?? config.color),
          getLineColor: normalizeResidentColor(
            polygons.getLineColor ?? config.visConfig.strokeColor ?? config.color
          ),
          getElevation:
            offsetActive && !config.visConfig.enable3d ? 0 : (polygons.getElevation ?? 500),
          getElevationOffset: polygons.getElevationOffset ?? config.visConfig.elevationOffset ?? 0,
          elevationScale:
            binding.styleRender?.elevationScale ??
            (config.visConfig.elevationScale ?? 1) *
              (config.visConfig.fixedHeight
                ? 1
                : 2 **
                  Math.max(
                    8 -
                      (renderOptions.mapState?.zoom ?? 14) +
                      (renderOptions.mapState?.zoomOffset ?? 0),
                    0
                  )),
          extrusionRings: polygons.extrusionRings,
          extruded,
          filled: config.visConfig.filled,
          wireframe
        })
      );
    };
    if (!extruded) appendPolygons();
    const appendPath = (path: ResidentGeometryPathBinding, suffix: string): void => {
      layers.push(
        new GPUPathLayer({
          ...baseProps(layer, binding, renderOptions, path),
          id: `${layer.id}-${suffix}`,
          selectionMask: path.selectionMask ?? binding.selectionMask,
          getPath: path.getPath,
          getPathLow: path.getPathLow,
          opacity: config.visConfig.strokeOpacity,
          getColor: normalizeResidentColor(
            path.getColor ?? config.visConfig.strokeColor ?? config.color
          ),
          getWidth: path.getWidth ?? 1,
          widthUnits: 'meters',
          billboard: false,
          // Native geojson overrides: lineMiterLimit 2, capRounded and jointRounded.
          miterLimit: 2,
          capRounded: true,
          jointRounded: true,
          widthScale:
            binding.styleRender?.widthScale ??
            config.visConfig.thickness *
              layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14}) *
              8
        })
      );
    };
    if (paths) appendPath(paths, 'paths');
    if (outlines && polygonStroked) appendPath(outlines, 'outlines');
    if (geometry.points && (config.visConfig.filled || pointStroked)) {
      const points = geometry.points;
      const fixedRadius = geometry.fixedRadius ?? layer.meta.fixedRadius ?? false;
      layers.push(
        new GPUScatterplotLayer({
          ...baseProps(layer, binding, renderOptions, points),
          id: `${layer.id}-points`,
          selectionMask: points.selectionMask ?? binding.selectionMask,
          getPosition: points.positions3D ?? points.positions,
          getFillColor: normalizeResidentColor(points.colors ?? config.color),
          getRadius: points.sizes ?? 1,
          getLineColor: normalizeResidentColor(
            points.strokeColors ?? config.visConfig.strokeColor ?? config.color
          ),
          getLineWidth: points.lineWidths ?? 1,
          filled: config.visConfig.filled,
          stroked: pointStroked,
          lineWidthUnits: 'meters',
          lineWidthScale:
            binding.styleRender?.widthScale ??
            config.visConfig.thickness *
              layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14}) *
              8,
          radiusUnits: 'meters',
          billboard: false,
          radiusScale:
            binding.styleRender?.radiusScale ??
            (fixedRadius
              ? 1
              : (config.radiusField ? 1 : config.visConfig.radius) *
                layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14}))
        })
      );
    }
    if (extruded) appendPolygons();
    // Native geojson labels sit on per-feature representative anchors (column mode only); resolve
    // empty configurations too so the owner retires removed label resources.
    if (config.columnMode === 'geojson' || config.columnMode === undefined) {
      const labelFixedRadius = geometry.fixedRadius ?? layer.meta.fixedRadius ?? false;
      const geojsonLabels = options.resolvePointLabels?.(layer, binding, renderOptions, {
        radius: 1,
        radiusScale:
          binding.styleRender?.radiusScale ??
          (labelFixedRadius
            ? 1
            : (config.radiusField ? 1 : config.visConfig.radius) *
              layer.getZoomFactor(renderOptions.mapState ?? {zoom: 14})),
        radiusMaxPixels: Number.MAX_SAFE_INTEGER,
        visible: config.isVisible && renderOptions.visible !== false
      });
      if (geojsonLabels?.reason) {
        report(layer, geojsonLabels.reason);
      } else {
        const labelErrorHandler = baseProps(layer, binding, renderOptions).onError;
        for (const label of geojsonLabels?.layers ?? [])
          layers.push(
            label.clone({
              visible: config.isVisible && renderOptions.visible !== false,
              pickable: false,
              onError: labelErrorHandler,
              ...getBrushingProps(renderOptions),
              ...{nativeLayerId: layer.id}
            })
          );
      }
    }
    return layers;
  }

  class ResidentPointLayer extends LayerClasses.point {
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['point']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['point']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override setInitialLayerConfig(
      ...parameters: Parameters<InstanceType<LayerClassesType['point']>['setInitialLayerConfig']>
    ): this {
      if (!isResident(this)) return super.setInitialLayerConfig(...parameters);
      updateMetadata(this);
      return this;
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['point']>['getHoverData']>
    ) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      return isResident(this) ? point(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  class ResidentIconLayer extends LayerClasses.icon {
    static override findDefaultLayerProps(
      ...parameters: Parameters<LayerClassesType['icon']['findDefaultLayerProps']>
    ): ReturnType<LayerClassesType['icon']['findDefaultLayerProps']> {
      return options.isResidentDataset(parameters[0].id)
        ? {props: []}
        : super.findDefaultLayerProps(...parameters);
    }
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['icon']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['icon']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override updateLayerDomain(
      ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
    ): KeplerLayer {
      return isResident(this) ? this : super.updateLayerDomain(...parameters);
    }
    override getHoverData(...parameters: Parameters<KeplerLayer['getHoverData']>) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      // The native catalog loader invokes this after asynchronous mesh preparation.
      this.onRedrawNeeded = renderOptions.layerCallbacks?.onRedrawNeeded;
      return isResident(this) ? icon(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  class ResidentH3Layer extends LayerClasses.hexagonId {
    static override findDefaultLayerProps(
      ...parameters: Parameters<LayerClassesType['hexagonId']['findDefaultLayerProps']>
    ): ReturnType<LayerClassesType['hexagonId']['findDefaultLayerProps']> {
      return options.isResidentDataset(parameters[0].id)
        ? {props: []}
        : super.findDefaultLayerProps(...parameters);
    }
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['hexagonId']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['hexagonId']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override updateLayerDomain(
      ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
    ): KeplerLayer {
      return isResident(this) ? this : super.updateLayerDomain(...parameters);
    }
    override getHoverData(...parameters: Parameters<KeplerLayer['getHoverData']>) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      return isResident(this) ? hexagonId(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  const LineBase: new (
    props: ConstructorParameters<LayerClassesType['line']>[0]
  ) => ResidentLineInstance = LayerClasses.line;
  class ResidentLineLayer extends LineBase {
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['line']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['line']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['line']>['getHoverData']>
    ) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      return isResident(this) ? connection(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  class ResidentArcLayer extends LayerClasses.arc {
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['arc']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['arc']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['arc']>['getHoverData']>
    ) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      return isResident(this) ? connection(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  class ResidentGeojsonLayer extends LayerClasses.geojson {
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['geojson']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      // Native initialization normally discovers feature types by scanning source geometry.
      // Restore and delayed publication must refresh the same metadata from prepared bindings.
      updateMetadata(this);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['geojson']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override setInitialLayerConfig(
      ...parameters: Parameters<InstanceType<LayerClassesType['geojson']>['setInitialLayerConfig']>
    ): this {
      if (!isResident(this)) return super.setInitialLayerConfig(...parameters);
      updateMetadata(this);
      return this;
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['geojson']>['getHoverData']>
    ) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      return isResident(this) ? geojson(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  class ResidentTripLayer extends LayerClasses.trip {
    static override findDefaultLayerProps(
      ...parameters: Parameters<typeof LayerClasses.trip.findDefaultLayerProps>
    ): ReturnType<typeof LayerClasses.trip.findDefaultLayerProps> {
      return options.isResidentDataset(parameters[0].id)
        ? {props: []}
        : super.findDefaultLayerProps(...parameters);
    }
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_LAYER_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['trip']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      const domain = resolve(this)?.trip?.animationDomainMilliseconds;
      this.updateAnimationDomain(domain ? [...domain] : null);
      return RESIDENT_LAYER_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['trip']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override updateLayerDomain(
      ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
    ): KeplerLayer {
      return isResident(this) ? this : super.updateLayerDomain(...parameters);
    }
    override setInitialLayerConfig(
      ...parameters: Parameters<InstanceType<LayerClassesType['trip']>['setInitialLayerConfig']>
    ): this {
      if (!isResident(this)) return super.setInitialLayerConfig(...parameters);
      updateMetadata(this);
      return this;
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['trip']>['getHoverData']>
    ) {
      if (!isResident(this)) return super.getHoverData(...parameters);
      const trip = resolve(this)?.trip;
      if (!trip?.resolveHoverSourceRow) return getResidentHoverIdentity(parameters);
      // Native table mode reports the datum at the animation time, not the trip's first row.
      const identity = getResidentHoverIdentity(parameters);
      const currentTime = parameters[3]?.currentTime;
      if (!identity || typeof currentTime !== 'number') return null;
      const info: unknown = (parameters as readonly unknown[])[4];
      const primitiveIndex =
        info && typeof info === 'object' && 'index' in info && typeof info.index === 'number'
          ? info.index
          : -1;
      const sourceRow = trip.resolveHoverSourceRow(primitiveIndex, currentTime);
      return sourceRow < 0 ? null : new ResidentHoverIdentity(sourceRow);
    }
    override getDeckRenderCallbackData(
      ...parameters: Parameters<InstanceType<LayerClassesType['trip']>['getDeckRenderCallbackData']>
    ) {
      return isResident(this) ? null : super.getDeckRenderCallbackData(...parameters);
    }
    override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
      return isResident(this) ? trip(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  /** Native S2/A5/geohash panels edit `token`; rendering reuses the published polygon footprints. */
  function createResidentCellLayer(NativeLayer: typeof KeplerLayer): typeof KeplerLayer {
    return class ResidentCellLayer extends NativeLayer {
      static override findDefaultLayerProps(
        ...parameters: Parameters<typeof KeplerLayer.findDefaultLayerProps>
      ): ReturnType<typeof KeplerLayer.findDefaultLayerProps> {
        return options.isResidentDataset(parameters[0].id)
          ? {props: []}
          : super.findDefaultLayerProps(...parameters);
      }
      override hasLayerData(
        layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
      ): boolean {
        return isResident(this)
          ? layerData === RESIDENT_LAYER_DATA
          : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
      }
      override formatLayerData(...parameters: Parameters<KeplerLayer['formatLayerData']>): object {
        if (!isResident(this)) return super.formatLayerData(...parameters);
        updateMetadata(this as unknown as ResidentKeplerLayer);
        return RESIDENT_LAYER_DATA;
      }
      override updateLayerMeta(...parameters: Parameters<KeplerLayer['updateLayerMeta']>): void {
        if (isResident(this)) updateMetadata(this as unknown as ResidentKeplerLayer);
        else super.updateLayerMeta(...parameters);
      }
      override updateLayerDomain(
        ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
      ): KeplerLayer {
        return isResident(this) ? this : super.updateLayerDomain(...parameters);
      }
      setInitialLayerConfig(dataset: unknown): this {
        if (isResident(this)) {
          updateMetadata(this as unknown as ResidentKeplerLayer);
          return this;
        }
        const initialize = Reflect.get(NativeLayer.prototype, 'setInitialLayerConfig');
        return typeof initialize === 'function' ? Reflect.apply(initialize, this, [dataset]) : this;
      }
      override getHoverData(...parameters: Parameters<KeplerLayer['getHoverData']>) {
        return isResident(this)
          ? getResidentHoverIdentity(parameters)
          : super.getHoverData(...parameters);
      }
      override renderLayer(renderOptions: ResidentRenderOptions = {}): Layer[] {
        // The cell layer shares GeoJSON's channels, so the polygon renderer accepts it unchanged.
        return isResident(this)
          ? geojson(this as unknown as InstanceType<LayerClassesType['geojson']>, renderOptions)
          : super.renderLayer(renderOptions);
      }
    };
  }
  // Kepler's registry is heterogeneous. Preserve each constructor's native static metadata and
  // nonresident behavior while guarding resident datasets before any CPU formatting work.
  function guardUnsupported(NativeLayer: typeof KeplerLayer): typeof KeplerLayer {
    return class UnsupportedResidentLayer extends NativeLayer {
      static override findDefaultLayerProps(
        ...parameters: Parameters<typeof KeplerLayer.findDefaultLayerProps>
      ): ReturnType<typeof KeplerLayer.findDefaultLayerProps> {
        // Kepler calls this static hook during a layer-type change before instance formatting.
        return options.isResidentDataset(parameters[0].id)
          ? {props: []}
          : super.findDefaultLayerProps(...parameters);
      }
      override hasLayerData(
        layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
      ): boolean {
        return isResident(this)
          ? layerData === RESIDENT_LAYER_DATA
          : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
      }
      override formatLayerData(...parameters: Parameters<KeplerLayer['formatLayerData']>): object {
        return isResident(this) ? RESIDENT_LAYER_DATA : super.formatLayerData(...parameters);
      }
      override updateLayerMeta(...parameters: Parameters<KeplerLayer['updateLayerMeta']>): void {
        if (!isResident(this)) super.updateLayerMeta(...parameters);
      }
      override updateLayerDomain(
        ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
      ): KeplerLayer {
        return isResident(this) ? this : super.updateLayerDomain(...parameters);
      }
      override getHoverData(...parameters: Parameters<KeplerLayer['getHoverData']>) {
        return isResident(this)
          ? getResidentHoverIdentity(parameters)
          : super.getHoverData(...parameters);
      }
      setInitialLayerConfig(dataset: unknown): this {
        if (isResident(this)) return this;
        const initialize = Reflect.get(NativeLayer.prototype, 'setInitialLayerConfig');
        return typeof initialize === 'function' ? Reflect.apply(initialize, this, [dataset]) : this;
      }
      override renderLayer(...parameters: unknown[]): Layer[] {
        if (!isResident(this)) return super.renderLayer(...parameters);
        report(this, `The ${this.type} layer has no resident GPU renderer.`);
        return [];
      }
    };
  }
  /** Native FlowmapLayer stays the renderer; its CPU flow list is rebuilt from the resident mask. */
  // Guarded: test doubles and older pinned layer sets may not provide the flow class.
  const ResidentFlowLayer = LayerClasses.flow
    ? class extends LayerClasses.flow {
        private residentFlowData: unknown;
        private residentFlowVersion = 0;
        override hasLayerData(
          layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_LAYER_DATA
        ): boolean {
          return isResident(this)
            ? layerData === RESIDENT_LAYER_DATA
            : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
        }
        override formatLayerData(
          ...parameters: Parameters<InstanceType<LayerClassesType['flow']>['formatLayerData']>
        ): Record<string, unknown> {
          if (!isResident(this)) return super.formatLayerData(...parameters);
          updateMetadata(this);
          return RESIDENT_LAYER_DATA;
        }
        override updateLayerMeta(
          ...parameters: Parameters<InstanceType<LayerClassesType['flow']>['updateLayerMeta']>
        ): void {
          if (isResident(this)) updateMetadata(this);
          else super.updateLayerMeta(...parameters);
        }
        override updateLayerDomain(
          ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
        ): KeplerLayer {
          return isResident(this) ? this : super.updateLayerDomain(...parameters);
        }
        override renderLayer(
          ...parameters: Parameters<InstanceType<LayerClassesType['flow']>['renderLayer']>
        ): Layer[] {
          if (!isResident(this)) return super.renderLayer(...parameters);
          const [renderOptions] = parameters;
          const binding = prepare(this, renderOptions as ResidentRenderOptions);
          const config = this.config;
          const columns: Record<string, {value?: string | null} | undefined> = config.columns;
          // H3 endpoints are centers converted into a derived lat/lng flow runtime.
          const h3Mode = config.columnMode === 'H3';
          const h3Flow = h3Mode ? options.resolveFlowH3?.(this) : undefined;
          const flow = h3Mode ? h3Flow?.binding : binding?.flow;
          const reason = !binding
            ? null
            : !flow
              ? h3Mode
                ? (h3Flow?.reason ?? 'Flow H3 mode requires a resident H3 endpoint adapter.')
                : 'No resident flow geometry is bound to this dataset.'
              : h3Mode
                ? null
                : config.columnMode !== 'LAT_LNG'
                  ? 'This flow column mode has no resident binding.'
                  : columns['lng0']?.value !== flow.coordinates.longitude ||
                      columns['lat0']?.value !== flow.coordinates.latitude ||
                      columns['lng1']?.value !== flow.roles.targetLongitude ||
                      columns['lat1']?.value !== flow.roles.targetLatitude ||
                      (columns['count']?.value ?? undefined) !== flow.roles.count ||
                      (columns['sourceName']?.value ?? undefined) !== flow.roles.sourceName ||
                      (columns['targetName']?.value ?? undefined) !== flow.roles.targetName
                    ? 'The selected flow columns do not match the prepared source.'
                    : null;
          if (binding) report(this, reason);
          if (!binding || !flow || reason) return [];
          flow.update(binding.selectionMask);
          // The H3 runtime resolves asynchronously, so its metadata is published on first render.
          if (h3Mode) {
            const nextMeta = getFlowLayerMeta(flow, binding.bounds);
            if (hasFlowLayerMetaChanged(this.meta, nextMeta))
              this.meta = {...this.meta, ...nextMeta};
          }
          // The native FlowmapLayer is GLSL-only. FlowmapGPULayer keeps the @flowmap.gl data provider
          // (clustering, scales, fade, top-N) and draws WGSL flow-line and flow-circle sublayers.
          const layerData = flow.getLayerData();
          const dataProvider = this.getDataProvider();
          if (this.residentFlowData !== layerData) {
            dataProvider.setFlowmapData(layerData);
            this.residentFlowData = layerData;
            this.residentFlowVersion = ++residentFlowDataVersion;
          }
          const visConfig = this.config.visConfig as FlowVisConfig;
          const blendingParameters = renderOptions.mapState?.layerParameters ?? {};
          const flowParameters = {
            cull: false,
            depthCompare: 'always' as const,
            depthWriteEnabled: false,
            ...blendingParameters
          };
          return [
            new FlowmapGPULayer({
              id: `${this.id}-resident-flow`,
              nativeLayerId: this.id,
              visible: this.config.isVisible && renderOptions.visible !== false,
              dataProvider,
              data: this.residentFlowVersion,
              ...FLOWMAP_DATA_ACCESSORS,
              pickable:
                (renderOptions as ResidentRenderOptions).interactionConfig?.tooltip?.enabled !==
                false,
              opacity: visConfig.opacity,
              darkMode: visConfig.darkBaseMapEnabled,
              colorScheme: visConfig.colorRange?.colors,
              fadeAmount: visConfig.flowFadeAmount,
              fadeEnabled: visConfig.flowFadeEnabled,
              fadeOpacityEnabled: false,
              locationTotalsEnabled: visConfig.flowLocationTotalsEnabled,
              flowLinesRenderingMode: visConfig.flowLinesRenderingMode || 'straight',
              flowLineThicknessScale: visConfig.flowLineThicknessScale,
              flowLineCurviness: visConfig.flowLineCurviness,
              clusteringEnabled: visConfig.flowClusteringEnabled,
              maxTopFlowsDisplayNum: visConfig.maxTopFlowsDisplayNum,
              clusteringAuto: true,
              adaptiveScalesEnabled: visConfig.flowAdaptiveScalesEnabled,
              locationLabelsEnabled: visConfig.flowLocationLabelsEnabled ?? false,
              scaleLockEnabled: visConfig.flowScaleLockEnabled ?? false,
              locationLabelStyle: (() => {
                const text = (this.config as {textLabel?: {color?: number[]; size?: number}[]})
                  .textLabel?.[0];
                return text ? {color: text.color, size: text.size} : undefined;
              })(),
              // A handler makes the layer resolve flow/location objects; Kepler's Deck-level onHover
              // then receives them through the layer's getPickingInfo, as with the native FlowmapLayer.
              onHover: () => undefined,
              parameters: flowParameters
            })
          ];
        }
      }
    : undefined;
  const ResidentFlowFieldLayer =
    LayerClasses.flowField && options.resolveFlowField
      ? createResidentFlowFieldLayerClass(guardUnsupported(LayerClasses.flowField), {
          isResident,
          report,
          resolve: layer => options.resolveFlowField?.(layer, undefined)
        })
      : undefined;
  /** Native 3D (glTF per point): resident positions and mask drive Deck's ScenegraphLayer. */
  const NativeScenegraphLayer = (LayerClasses as Record<string, typeof KeplerLayer | undefined>)[
    '3D'
  ];
  const ResidentScenegraphLayer = NativeScenegraphLayer
    ? class extends guardUnsupported(NativeScenegraphLayer) {
        override formatLayerData(
          ...parameters: Parameters<KeplerLayer['formatLayerData']>
        ): object {
          if (isResident(this)) updateMetadata(this as unknown as ResidentKeplerLayer);
          return super.formatLayerData(...parameters);
        }
        override updateLayerMeta(...parameters: Parameters<KeplerLayer['updateLayerMeta']>): void {
          if (isResident(this)) updateMetadata(this as unknown as ResidentKeplerLayer);
          else super.updateLayerMeta(...parameters);
        }
        override renderLayer(...parameters: unknown[]): Layer[] {
          if (!isResident(this)) return super.renderLayer(...parameters);
          return renderResidentScenegraph(
            {
              prepare: (layer, renderOptions) =>
                prepare(
                  layer as unknown as ResidentKeplerLayer,
                  renderOptions as ResidentRenderOptions
                ),
              baseProps: (layer, binding, renderOptions) =>
                baseProps(layer as unknown as ResidentKeplerLayer, binding, renderOptions),
              report
            },
            this,
            (parameters[0] ?? {}) as ResidentRenderOptions
          );
        }
      }
    : undefined;
  const supported = {
    ...(ResidentFlowFieldLayer ? {flowField: ResidentFlowFieldLayer} : {}),
    hexagonId: ResidentH3Layer,
    icon: ResidentIconLayer,
    trip: ResidentTripLayer,
    ...(ResidentFlowLayer ? {flow: ResidentFlowLayer} : {}),
    ...(ResidentScenegraphLayer ? {'3D': ResidentScenegraphLayer} : {}),
    point: ResidentPointLayer,
    line: ResidentLineLayer,
    arc: ResidentArcLayer,
    geojson: ResidentGeojsonLayer,
    s2: createResidentCellLayer(LayerClasses.s2 as unknown as typeof KeplerLayer),
    a5: createResidentCellLayer(LayerClasses.a5 as unknown as typeof KeplerLayer),
    geohash: createResidentCellLayer(LayerClasses.geohash as unknown as typeof KeplerLayer)
  };
  // Constructor types differ by native config, but wrappers retain each constructor verbatim.
  return Object.fromEntries(
    Object.entries(LayerClasses).map(([key, NativeLayer]) => [
      key,
      key in supported ? supported[key as keyof typeof supported] : guardUnsupported(NativeLayer)
    ])
  ) as LayerClassesType;
}

function matchesColumns(
  columns: {[name: string]: {value: string | null} | undefined},
  expected: Record<string, string>
): boolean {
  return Object.entries(expected).every(([key, value]) => columns[key]?.value === value);
}
/** Native Kepler colors may omit alpha; GPU unorm8x4 constants always need four bytes. */
function normalizeResidentColor<Format extends GPUVectorFormat>(
  color: Color | GPUVector<Format>
): Color | GPUVector<Format> {
  return 'data' in color ? color : [color[0], color[1], color[2], color[3] ?? 255];
}
