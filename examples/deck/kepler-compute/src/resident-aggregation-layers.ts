// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {naturalBreaks} from '@kepler.gl/utils';
import {getResidentHoverIdentity} from './resident-hover-identity';

import {LayerClasses, type LayerClassesType, type Layer as KeplerLayer} from '@kepler.gl/layers';
import type {Layer, PickingInfo} from '@deck.gl/core';
import {PathLayer} from '@deck.gl/layers';
import {GPUScatterplotLayer, GPUSolidPolygonLayer} from '../gpu-layers/index';
import {
  getAggregateCellOutline,
  getAggregationHighlightProps,
  getAggregationOutlineWidth
} from './aggregation-hover';
import type {NativeAggregationLayer, ResidentAggregationBinding} from './aggregation-runtime';
import type {ResidentClusterBinding, ResidentClusterSettings} from './cluster-runtime';

export type ResidentAggregationLayerOptions = {
  isResidentDataset(datasetId: string): boolean;
  /** Resolve using this layer's native source-row selection scope, including targeted polygons. */
  resolveAggregation(
    datasetId: string,
    layer: NativeAggregationLayer
  ): {binding?: ResidentAggregationBinding; reason?: string};
  onUnsupported(layerId: string, reason: string | null): void;
  onHover?(
    information: PickingInfo & {datasetId: string; aggregateLayerId: string; cellId: number}
  ): void;
  getBounds?(datasetId: string): [number, number, number, number] | undefined;
  /** Explicit CPU-readback supercluster adapter; `{}` means a bounded readback is pending. */
  resolveCluster?(
    datasetId: string,
    layer: NativeClusterLayer,
    request: {
      settings: ResidentClusterSettings;
      /** Native color field name and aggregation; absent means the member-count measure. */
      colorField?: {name: string; aggregation: string};
      view: {longitude: number; latitude: number; zoom: number; width: number; height: number};
    }
  ): {binding?: ResidentClusterBinding; reason?: string};
  onClusterHover?(
    information: PickingInfo & {
      datasetId: string;
      clusterLayerId: string;
      clusterIndex: number;
      members: ReturnType<ResidentClusterBinding['getMembers']>;
      value: number;
    }
  ): void;
};
interface NativeClusterLayer
  extends Omit<
    InstanceType<LayerClassesType['cluster']>,
    'renderLayer' | 'type' | 'hasLayerData' | 'formatLayerData' | 'updateLayerMeta' | 'getHoverData'
  > {
  type: 'cluster';
  getColorRange(colorRange: unknown): readonly (readonly number[])[];
  hasLayerData(layerData: unknown): boolean;
  formatLayerData(...parameters: Parameters<KeplerLayer['formatLayerData']>): object;
  updateLayerMeta(
    ...parameters: Parameters<InstanceType<LayerClassesType['grid']>['updateLayerMeta']>
  ): void;
  getHoverData(
    ...parameters: Parameters<KeplerLayer['getHoverData']>
  ): ReturnType<KeplerLayer['getHoverData']>;
  renderLayer(
    ...parameters: Parameters<KeplerLayer['renderLayer']>
  ): ReturnType<KeplerLayer['renderLayer']>;
}
// Native aggregate classes narrow Deck return types; plugin rendering accepts any Deck layer.
interface NativeLayer
  extends Omit<InstanceType<LayerClassesType['grid' | 'hexagon']>, 'renderLayer' | 'type'> {
  type: 'grid' | 'hexagon';
  hasLayerData(
    layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_AGGREGATION_DATA
  ): boolean;
  formatLayerData(...parameters: Parameters<KeplerLayer['formatLayerData']>): object;
  updateLayerMeta(
    ...parameters: Parameters<InstanceType<LayerClassesType['grid']>['updateLayerMeta']>
  ): void;
  getHoverData(
    ...parameters: Parameters<KeplerLayer['getHoverData']>
  ): ReturnType<KeplerLayer['getHoverData']>;
  renderLayer(
    ...parameters: Parameters<KeplerLayer['renderLayer']>
  ): ReturnType<KeplerLayer['renderLayer']>;
}
type RenderOptions = {
  idx?: number;
  visible?: boolean;
  mapState?: {
    globe?: {enabled?: boolean};
    longitude?: number;
    latitude?: number;
    zoom?: number;
    width?: number;
    height?: number;
  };
  interactionConfig?: {brush?: {enabled?: boolean}; tooltip?: {enabled?: boolean}};
  objectHovered?: {index?: number; picked?: boolean; layer?: {props?: {id?: string}}} | null;
};
const RESIDENT_AGGREGATION_DATA = Object.freeze({residentAggregation: true});

/** Merge these native layer overrides after createResidentLayerClasses. */
export function createResidentAggregationLayerClasses(
  options: ResidentAggregationLayerOptions
): Pick<LayerClassesType, 'grid' | 'hexagon' | 'cluster'> {
  const diagnostics = new Map<string, string | null>();
  const isResident = (layer: NativeLayer): boolean =>
    Boolean(layer.config.dataId && options.isResidentDataset(layer.config.dataId));
  const report = (layer: NativeLayer, reason: string | null): void => {
    if (diagnostics.get(layer.id) !== reason) {
      diagnostics.set(layer.id, reason);
      options.onUnsupported(layer.id, reason);
    }
  };
  const render = (layer: NativeLayer, renderOptions: RenderOptions): Layer[] => {
    if (!layer.config.isVisible || renderOptions.visible === false) return [];
    // Native grid, hexagon and cluster layers ignore the hover brush, so it needs no refusal.
    const {binding, reason} = options.resolveAggregation(layer.config.dataId!, layer);
    if (!binding) {
      report(layer, reason ?? null);
      return [];
    }
    report(layer, null);
    // Native flat cells draw the hovered cell's outline in the layer highlight color.
    const hovered = renderOptions.objectHovered;
    const outline =
      !binding.extruded &&
      hovered?.layer?.props?.id === layer.id &&
      typeof hovered.index === 'number' &&
      hovered.index >= 0 &&
      hovered.index < binding.layout.cellCount
        ? getAggregateCellOutline(binding.layout, hovered.index)
        : undefined;
    const zoomFactor = (
      layer as unknown as {getZoomFactor(state: {zoom: number; zoomOffset?: number}): number}
    ).getZoomFactor({
      zoom: renderOptions.mapState?.zoom ?? 0,
      zoomOffset: (renderOptions.mapState as {zoomOffset?: number} | undefined)?.zoomOffset
    });
    // Native grid/hexagon label every cell with its point count (default text label, collision on).
    const renderCountLabels = renderOptions.mapState
      ? (
          layer as unknown as {
            getCountLabelProps(options: unknown): {
              renderCountLabels?: (data: unknown) => Layer[];
            };
          }
        ).getCountLabelProps(renderOptions).renderCountLabels
      : undefined;
    const countLabels = renderCountLabels ? binding.getCountLabels?.() : undefined;
    return [
      new GPUSolidPolygonLayer({
        id: layer.id,
        ...{idx: renderOptions.idx},
        ...getAggregationHighlightProps(Boolean(binding.extruded)),
        positions: binding.positions,
        indices: binding.indices,
        rowIndices: binding.rowIndices,
        selectionMask: binding.selectionMask,
        getFillColor: binding.getFillColor,
        ...(binding.extruded
          ? {
              extruded: true,
              getElevation: binding.getElevation,
              extrusionRings: binding.extrusionRings,
              // Native: visConfig.elevationScale times the zoom factor; heights are in meters.
              elevationScale:
                (layer.config.visConfig.elevationScale ?? 5) *
                (
                  layer as unknown as {getElevationZoomFactor(state: {zoom: number}): number}
                ).getElevationZoomFactor({zoom: renderOptions.mapState?.zoom ?? 0})
            }
          : {}),
        coordinateSystem: binding.coordinateSystem,
        coordinateOrigin: binding.coordinateOrigin,
        modelMatrix: binding.modelMatrix,
        opacity: layer.config.visConfig.opacity,
        pickable: renderOptions.interactionConfig?.tooltip?.enabled !== false,
        onHover: information => {
          options.onHover?.({
            ...information,
            datasetId: layer.config.dataId!,
            aggregateLayerId: layer.id,
            cellId: information.index
          });
          return false;
        },
        // Native popovers resolve aggregate IDs through the bounded cell reader.
        onClick: () => false
      }),
      ...(outline
        ? [
            new PathLayer({
              id: `${layer.id}-hovered`,
              data: [{path: outline}],
              getPath: (item: {path: [number, number][]}) => item.path,
              getColor: layer.config.highlightColor as [number, number, number],
              getWidth: 1,
              widthUnits: 'meters',
              widthScale: getAggregationOutlineWidth(zoomFactor),
              pickable: false,
              coordinateSystem: binding.coordinateSystem,
              coordinateOrigin: binding.coordinateOrigin,
              modelMatrix: binding.modelMatrix
            })
          ]
        : []),
      ...(renderCountLabels && countLabels?.length ? renderCountLabels(countLabels) : [])
    ];
  };
  const updateMetadata = (layer: NativeLayer): void => {
    const bounds = layer.config.dataId ? options.getBounds?.(layer.config.dataId) : undefined;
    if (bounds) layer.meta = {...layer.meta, bounds};
  };
  const GridBase: new (props: ConstructorParameters<LayerClassesType['grid']>[0]) => NativeLayer =
    LayerClasses.grid;
  class ResidentGridLayer extends GridBase {
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_AGGREGATION_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_AGGREGATION_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['grid']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_AGGREGATION_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['grid']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['grid']>['getHoverData']>
    ) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: RenderOptions = {}): Layer[] {
      return isResident(this) ? render(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  const HexagonBase: new (
    props: ConstructorParameters<LayerClassesType['hexagon']>[0]
  ) => NativeLayer = LayerClasses.hexagon;
  class ResidentHexagonLayer extends HexagonBase {
    override hasLayerData(
      layerData: Parameters<KeplerLayer['hasLayerData']>[0] | typeof RESIDENT_AGGREGATION_DATA
    ): boolean {
      return isResident(this)
        ? layerData === RESIDENT_AGGREGATION_DATA
        : Boolean(layerData && 'data' in layerData && super.hasLayerData(layerData));
    }
    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['hexagon']>['formatLayerData']>
    ): object {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_AGGREGATION_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['hexagon']>['updateLayerMeta']>
    ): void {
      if (isResident(this)) updateMetadata(this);
      else super.updateLayerMeta(...parameters);
    }
    override getHoverData(
      ...parameters: Parameters<InstanceType<LayerClassesType['hexagon']>['getHoverData']>
    ) {
      return isResident(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: RenderOptions = {}): Layer[] {
      return isResident(this) ? render(this, renderOptions) : super.renderLayer(renderOptions);
    }
  }
  const ClusterBase: new (
    props: ConstructorParameters<LayerClassesType['cluster']>[0]
  ) => NativeClusterLayer = LayerClasses.cluster as never;
  const isResidentCluster = (layer: NativeClusterLayer): boolean =>
    Boolean(layer.config.dataId && options.isResidentDataset(layer.config.dataId));
  const reportCluster = (layer: NativeClusterLayer, reason: string | null): void =>
    report(layer as unknown as NativeLayer, reason);
  const renderCluster = (layer: NativeClusterLayer, renderOptions: RenderOptions): Layer[] => {
    if (!layer.config.isVisible || renderOptions.visible === false) return [];
    const {mapState} = renderOptions;
    const visConfig = layer.config.visConfig;
    const colorScale: PropertyKey | undefined = layer.config.colorScale;
    const reason = !options.resolveCluster
      ? 'Resident clusters are not configured.'
      : colorScale !== 'quantize' &&
          colorScale !== 'quantile' &&
          colorScale !== 'ordinal' &&
          colorScale !== 'jenks' &&
          colorScale !== 'custom'
        ? 'Resident clusters do not support this color scale.'
        : null;
    if (reason || !options.resolveCluster || !mapState) {
      reportCluster(layer, reason ?? 'Resident clusters need a map viewport.');
      return [];
    }
    const {binding, reason: resolveReason} = options.resolveCluster(layer.config.dataId!, layer, {
      settings: {
        clusterRadius: visConfig.clusterRadius,
        minClusterSize: visConfig.minClusterSize,
        radiusRange: visConfig.radiusRange,
        colorRange: layer.getColorRange(visConfig.colorRange),
        colorScaleType:
          colorScale === 'jenks' || colorScale === 'custom'
            ? 'threshold'
            : (colorScale as 'quantize' | 'quantile' | 'ordinal'),
        // Native: jenks breaks over the aggregated values; custom breaks from the color map.
        getThresholds:
          colorScale === 'jenks'
            ? values =>
                naturalBreaks(
                  [...values],
                  layer.getColorRange(visConfig.colorRange).length
                ) as number[]
            : colorScale === 'custom'
              ? () =>
                  (
                    (visConfig.colorRange as {colorMap?: readonly (readonly unknown[])[]})
                      .colorMap ?? []
                  ).flatMap(entry => (Number.isFinite(entry[0]) ? [entry[0] as number] : []))
              : undefined
      },
      ...(layer.config.colorField
        ? {
            colorField: {
              name: layer.config.colorField.name,
              aggregation: (visConfig as {colorAggregation?: string}).colorAggregation ?? 'count'
            }
          }
        : {}),
      view: {
        longitude: mapState.longitude ?? 0,
        latitude: mapState.latitude ?? 0,
        zoom: mapState.zoom ?? 0,
        width: mapState.width ?? 1,
        height: mapState.height ?? 1
      }
    });
    if (!binding) {
      reportCluster(layer, resolveReason ?? null);
      return [];
    }
    reportCluster(layer, null);
    if (!binding.clusterCount) return [];
    return [
      new GPUScatterplotLayer({
        id: `${layer.id}-resident-cluster`,
        ...{idx: renderOptions.idx},
        getPosition: binding.centers,
        getRadius: binding.radii,
        getFillColor: binding.colors,
        radiusUnits: 'pixels',
        radiusScale: 1,
        filled: true,
        opacity: visConfig.opacity,
        pickable: renderOptions.interactionConfig?.tooltip?.enabled !== false,
        // Parameters match native cluster scatterplots, which do not write depth.
        parameters: {depthWriteEnabled: false},
        onHover: information => {
          if (information.index >= 0)
            options.onClusterHover?.({
              ...information,
              datasetId: layer.config.dataId!,
              clusterLayerId: layer.id,
              clusterIndex: information.index,
              members: binding.getMembers(information.index, 50),
              value: binding.getValue(information.index)
            });
          return false;
        },
        onClick: () => false
      })
    ];
  };
  class ResidentClusterLayer extends ClusterBase {
    override hasLayerData(layerData: unknown): boolean {
      return isResidentCluster(this)
        ? layerData === RESIDENT_AGGREGATION_DATA
        : Boolean(layerData && 'data' in (layerData as object) && super.hasLayerData(layerData));
    }
    override formatLayerData(...parameters: Parameters<KeplerLayer['formatLayerData']>): object {
      if (!isResidentCluster(this)) return super.formatLayerData(...parameters);
      const bounds = this.config.dataId ? options.getBounds?.(this.config.dataId) : undefined;
      if (bounds) this.meta = {...this.meta, bounds};
      return RESIDENT_AGGREGATION_DATA;
    }
    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['grid']>['updateLayerMeta']>
    ): void {
      if (!isResidentCluster(this)) super.updateLayerMeta(...parameters);
    }
    override getHoverData(...parameters: Parameters<KeplerLayer['getHoverData']>) {
      return isResidentCluster(this)
        ? getResidentHoverIdentity(parameters)
        : super.getHoverData(...parameters);
    }
    override renderLayer(renderOptions: RenderOptions = {}): Layer[] {
      return isResidentCluster(this)
        ? renderCluster(this, renderOptions)
        : super.renderLayer(renderOptions);
    }
  }
  return {
    grid: ResidentGridLayer,
    hexagon: ResidentHexagonLayer,
    cluster: ResidentClusterLayer
  } as unknown as Pick<LayerClassesType, 'grid' | 'hexagon' | 'cluster'>;
}
