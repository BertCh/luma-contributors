// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) Flowmap.gl contributors
// SPDX-FileCopyrightText: Copyright (c) 2018-2020 Teralytics

// Derived from FlowmapLayer in @flowmap.gl/layers (Apache-2.0, Copyright (c) Flowmap.gl contributors,
// Copyright (c) 2018-2020 Teralytics). The GLSL sublayers are replaced by WGSL GPU layers; the data
// provider (clustering, color scale, fade, top-N, adaptive scales) is the unmodified @flowmap.gl/data.

import {
  CompositeLayer,
  type Layer,
  type LayerContext,
  type PickingInfo,
  type UpdateParameters
} from '@deck.gl/core';
import {ScatterplotLayer} from '@deck.gl/layers';
import {GPUFlowCirclesLayer, GPUFlowLinesLayer} from '@deck.gl-community/gpu-layers';
import {
  FlowLocationLabels,
  selectFlowLocationLabels,
  type FlowLocationLabelStyle
} from './flow-location-labels';
import type {
  FilterState,
  FlowCirclesLayerAttributes,
  FlowLinesLayerAttributes,
  FlowLinesRenderingMode,
  FlowmapDataProvider,
  LayersData,
  ScaleLockDomains
} from '@flowmap.gl/data';

/** Verification hook: latest computed flowmap state per Kepler layer id (read by browser scripts). */
export type FlowmapStats = {
  flowCount: number;
  locationCount: number;
  flowMagnitudeDomain?: [number, number];
  locationTotalsDomain?: [number, number];
  labelCount: number;
  labelDiagnostic?: string;
  scaleLocked: boolean;
};
const flowmapStats = new Map<string, FlowmapStats>();
export function getFlowmapStats(): Record<string, FlowmapStats> {
  return Object.fromEntries(flowmapStats);
}
const flowmapStatsListeners = new Set<() => void>();
/** Lets the layer configuration panel redraw its scale-lock indicator when the stats change. */
export function subscribeFlowmapStats(listener: () => void): () => void {
  flowmapStatsListeners.add(listener);
  return () => flowmapStatsListeners.delete(listener);
}

export type FlowmapPickingType = 'location' | 'flow';

/** Same shape as `@flowmap.gl/layers` picking objects, so Kepler's `getHoverData` renders them. */
export type FlowmapPickingObject =
  | {
      type: 'location';
      location: Record<string, unknown>;
      id: string | number;
      name: string;
      totals: {incomingCount: number; outgoingCount: number; internalCount: number};
      circleRadius: number;
    }
  | {
      type: 'flow';
      flow: Record<string, unknown>;
      origin: Record<string, unknown>;
      dest: Record<string, unknown>;
      count: number;
    };

type SourceEvent = {srcEvent: MouseEvent};
// biome-ignore lint/suspicious/noExplicitAny: Deck picking information is loosely typed.
type PickInformation = any;

export type FlowmapGPULayerProps = {
  id: string;
  /** Kepler layer id, reported by rendered-layer diagnostics and picking. */
  nativeLayerId?: string;
  dataProvider: FlowmapDataProvider<any, any>;
  /** Changes whenever the provider's flowmap data was replaced. */
  data: number;
  filter?: FilterState;
  locationsEnabled?: boolean;
  locationTotalsEnabled?: boolean;
  locationLabelsEnabled?: boolean;
  /** Label text style; defaults follow darkMode. */
  locationLabelStyle?: Partial<FlowLocationLabelStyle>;
  /** Freezes the flow and location-total scale domains at their current extents. */
  scaleLockEnabled?: boolean;
  adaptiveScalesEnabled?: boolean;
  flowLineThicknessScale?: number;
  flowLineCurviness?: number;
  flowLinesRenderingMode?: FlowLinesRenderingMode;
  clusteringEnabled?: boolean;
  clusteringLevel?: number;
  clusteringAuto?: boolean;
  fadeEnabled?: boolean;
  fadeOpacityEnabled?: boolean;
  fadeAmount?: number;
  darkMode?: boolean;
  colorScheme?: string | string[];
  highlightColor?: string | number[];
  maxTopFlowsDisplayNum?: number;
  flowEndpointsInViewportMode?: 'any' | 'both';
  getLocationId: (location: any) => string | number;
  getLocationName: (location: any) => string;
  getFlowOriginId: (flow: any) => string | number;
  getFlowDestId: (flow: any) => string | number;
  getFlowMagnitude: (flow: any) => number;
  onHover?: (info: PickInformation | undefined, event: SourceEvent) => void;
  onClick?: (info: PickInformation | undefined, event: SourceEvent) => void;
  [key: string]: unknown;
};

/** Native Kepler flow accessors over `FlowLayerData` locations and flows. */
export const FLOWMAP_DATA_ACCESSORS = {
  getLocationId: (location: {id: number}) => location.id,
  getLocationLon: (location: {lon: number}) => location.lon,
  getLocationLat: (location: {lat: number}) => location.lat,
  getLocationName: (location: {name: string}) => location.name,
  getFlowOriginId: (flow: {sourceId: number}) => flow.sourceId,
  getFlowDestId: (flow: {targetId: number}) => flow.targetId,
  getFlowMagnitude: (flow: {count: number}) => flow.count
};

const PROPS_TO_CAUSE_LAYER_DATA_UPDATE = [
  'filter',
  'locationsEnabled',
  'locationTotalsEnabled',
  'locationLabelsEnabled',
  'scaleLockEnabled',
  'adaptiveScalesEnabled',
  'flowLinesRenderingMode',
  'clusteringEnabled',
  'clusteringLevel',
  'fadeEnabled',
  'fadeOpacityEnabled',
  'clusteringAuto',
  'darkMode',
  'fadeAmount',
  'colorScheme',
  'highlightColor',
  'maxTopFlowsDisplayNum',
  'flowEndpointsInViewportMode',
  'flowLineThicknessScale'
];

type HighlightedObject =
  | {type: 'location'; coordinates: [number, number]; radius: number}
  | {type: 'flow'; lineAttributes: FlowLinesLayerAttributes};

type FlowmapGPULayerState = {
  layersData?: LayersData;
  highlightedObject?: HighlightedObject;
  pickingInfo?: PickInformation;
  lastHoverTime?: number;
  lastClickTime?: number;
  lockedDomains?: ScaleLockDomains;
  labels?: FlowLocationLabels;
};

const NAMED_COLORS: Record<string, [number, number, number]> = {
  orange: [255, 165, 0],
  white: [255, 255, 255],
  black: [0, 0, 0],
  red: [255, 0, 0],
  yellow: [255, 255, 0]
};

/** Parses the highlight color forms accepted by FlowmapLayer: RGB(A) array, #rgb[a], or a name. */
export function parseFlowmapColor(
  color: string | readonly number[] | undefined
): [number, number, number, number] {
  if (Array.isArray(color) || ArrayBuffer.isView(color)) {
    const values = color as readonly number[];
    return [values[0], values[1], values[2], values[3] ?? 255];
  }
  if (typeof color === 'string') {
    const named = NAMED_COLORS[color.toLowerCase()];
    if (named) return [...named, 255];
    const hex = color.replace('#', '');
    const expanded = hex.length <= 4 ? [...hex].map(digit => digit + digit).join('') : hex;
    const value = Number.parseInt(expanded.slice(0, 6), 16);
    if (expanded.length >= 6 && Number.isFinite(value)) {
      const alpha = expanded.length >= 8 ? Number.parseInt(expanded.slice(6, 8), 16) : 255;
      return [(value >> 16) & 255, (value >> 8) & 255, value & 255, alpha];
    }
  }
  return [255, 165, 0, 255];
}

/** One row of the interleaved binary attributes, as FlowmapLayer highlights a hovered flow. */
export function getFlowLineAttributesByIndex(
  lineAttributes: FlowLinesLayerAttributes,
  index: number,
  alpha?: number
): FlowLinesLayerAttributes {
  const entries = Object.entries(lineAttributes.attributes).map(([name, attribute]) => {
    const size = attribute.size;
    let value = attribute.value.slice(index * size, (index + 1) * size);
    if (alpha !== undefined && name === 'getColor') {
      value = new Uint8Array([value[0], value[1], value[2], alpha]);
    }
    return [name, {value, size}];
  });
  return {length: 1, attributes: Object.fromEntries(entries)} as FlowLinesLayerAttributes;
}

function getCircleCoordinates(
  circleAttributes: FlowCirclesLayerAttributes,
  index: number
): [number, number] {
  const {value, size} = circleAttributes.attributes.getPosition;
  return [value[index * size], value[index * size + 1]];
}

function getOuterCircleRadius(circleAttributes: FlowCirclesLayerAttributes, index: number): number {
  const {getInRadius, getOutRadius} = circleAttributes.attributes;
  return Math.max(getInRadius.value[index], getOutRadius.value[index]);
}

function isFlowLinesLayer(layer: unknown): boolean {
  return layer instanceof GPUFlowLinesLayer;
}

/**
 * WebGPU FlowmapLayer: the native data pipeline with WGSL flow-line and flow-circle sublayers,
 * hover highlight, and flow/location picking objects identical to `@flowmap.gl/layers`.
 */
export class FlowmapGPULayer extends CompositeLayer<FlowmapGPULayerProps> {
  static override layerName = 'FlowmapGPULayer';
  static override defaultProps = {
    darkMode: true,
    fadeAmount: 50,
    locationsEnabled: true,
    locationTotalsEnabled: true,
    locationLabelsEnabled: false,
    clusteringEnabled: true,
    fadeEnabled: true,
    fadeOpacityEnabled: false,
    clusteringAuto: true,
    adaptiveScalesEnabled: true,
    flowLineThicknessScale: 1,
    flowLineCurviness: 1,
    flowLinesRenderingMode: 'straight',
    colorScheme: 'Teal',
    highlightColor: 'orange',
    maxTopFlowsDisplayNum: 5000,
    flowEndpointsInViewportMode: 'any'
  } as const;

  declare state: FlowmapGPULayerState;
  private layersDataGeneration = 0;

  constructor(props: FlowmapGPULayerProps) {
    super({
      ...props,
      onHover: ((info: PickInformation, event: SourceEvent) => {
        const startTime = Date.now();
        this.setState({
          highlightedObject: this.getHighlightedObject(info),
          lastHoverTime: startTime
        });
        const {onHover} = props;
        if (onHover) {
          void this.getFlowmapPickingInfo(info).then(pickingInfo => {
            if ((this.state?.lastHoverTime ?? 0) <= startTime) {
              this.setState({pickingInfo});
              onHover(pickingInfo, event);
            }
          });
        }
      }) as never,
      onClick: ((info: PickInformation, event: SourceEvent) => {
        const startTime = Date.now();
        this.setState({lastClickTime: startTime});
        const {onClick} = props;
        if (onClick) {
          void this.getFlowmapPickingInfo(info).then(pickingInfo => {
            if ((this.state?.lastClickTime ?? 0) <= startTime) {
              this.setState({pickingInfo});
              if (pickingInfo) onClick(pickingInfo, event);
            }
          });
        }
      }) as never
    } as never);
  }

  override initializeState(_context: LayerContext): void {
    this.state = {};
  }

  /** Hover handlers set on the Deck component receive the resolved flow or location object. */
  override getPickingInfo({info}: {info: PickingInfo}): PickingInfo {
    if (!info.object) {
      const object = this.state?.pickingInfo?.object;
      if (object) return {...info, object, picked: true};
    }
    return info;
  }

  override shouldUpdateState(parameters: UpdateParameters<FlowmapGPULayer>): boolean {
    return Boolean(parameters.changeFlags.viewportChanged) || super.shouldUpdateState(parameters);
  }

  override updateState(parameters: UpdateParameters<FlowmapGPULayer>): void {
    super.updateState(parameters);
    const {oldProps, props, changeFlags} = parameters;
    if (changeFlags.viewportChanged || changeFlags.dataChanged) {
      this.setState({highlightedObject: undefined});
    }
    const settingsChanged = PROPS_TO_CAUSE_LAYER_DATA_UPDATE.some(
      name =>
        (oldProps as Record<string, unknown>)[name] !== (props as Record<string, unknown>)[name]
    );
    if (
      changeFlags.viewportChanged ||
      changeFlags.dataChanged ||
      (changeFlags.propsChanged && settingsChanged)
    ) {
      void this.refreshLayersData();
    }
  }

  private async refreshLayersData(): Promise<void> {
    const {dataProvider} = this.props;
    const generation = ++this.layersDataGeneration;
    await dataProvider.setFlowmapState(this.getFlowmapState());
    const layersData = await dataProvider.getLayersData();
    if (generation !== this.layersDataGeneration) return;
    this.setState({layersData, highlightedObject: undefined});
    if (layersData) this.updateLocationLabels(layersData);
    if (layersData) this.recordStats(layersData);
    // Scale lock freezes the extents of the first data it sees, then recomputes once with them.
    const scaleDomains = (layersData as {scaleDomains?: ScaleLockDomains}).scaleDomains;
    if (this.props.scaleLockEnabled && !this.state.lockedDomains && scaleDomains) {
      this.setState({lockedDomains: {...scaleDomains}});
      void this.refreshLayersData();
    } else if (!this.props.scaleLockEnabled && this.state.lockedDomains) {
      this.setState({lockedDomains: undefined});
    }
  }

  private recordStats(layersData: LayersData): void {
    const key = this.props.nativeLayerId ?? this.props.id;
    const domains = (layersData as {scaleDomains?: ScaleLockDomains}).scaleDomains;
    flowmapStats.set(key, {
      flowCount: layersData.lineAttributes?.length ?? 0,
      locationCount: layersData.circleAttributes?.length ?? 0,
      flowMagnitudeDomain: domains?.flowMagnitude,
      locationTotalsDomain: domains?.locationTotals,
      labelCount: this.state.labels?.labelCount ?? 0,
      labelDiagnostic: this.state.labels?.diagnostic,
      scaleLocked: Boolean(this.props.scaleLockEnabled && this.state.lockedDomains)
    });
    for (const listener of flowmapStatsListeners) listener();
  }

  /** Rebuilds the GPU location labels when the in-view label set changed (idempotent per set). */
  private updateLocationLabels(layersData: LayersData): void {
    const enabled = this.props.locationLabelsEnabled ?? false;
    if (!enabled) {
      this.state.labels?.setRows({
        names: [],
        positions: new Float32Array(0),
        radii: new Float32Array(0)
      });
      return;
    }
    const device = this.context.device;
    if (!device) return;
    let labels = this.state.labels;
    if (!labels) {
      labels = new FlowLocationLabels(device, () => this.setNeedsUpdate());
      this.setState({labels});
    }
    const viewport = this.context.viewport;
    const rows = selectFlowLocationLabels(
      layersData,
      {
        width: viewport.width,
        height: viewport.height,
        project: position => viewport.project(position) as [number, number]
      },
      undefined,
      {labelSize: this.props.locationLabelStyle?.size ?? 14}
    );
    if (labels.setRows(rows)) this.setNeedsUpdate();
  }

  override finalizeState(context: LayerContext): void {
    this.state.labels?.destroy();
    flowmapStats.delete(this.props.nativeLayerId ?? this.props.id);
    super.finalizeState(context);
  }

  private getFlowmapState() {
    const props = this.props;
    const {width, height, longitude, latitude, zoom, pitch, bearing} = this.context
      .viewport as unknown as Record<string, number>;
    const defaults = FlowmapGPULayer.defaultProps;
    return {
      viewport: {width, height, longitude, latitude, zoom, pitch, bearing},
      filter: props.filter,
      settings: {
        locationsEnabled: props.locationsEnabled ?? defaults.locationsEnabled,
        locationTotalsEnabled: props.locationTotalsEnabled ?? defaults.locationTotalsEnabled,
        locationLabelsEnabled: props.locationLabelsEnabled ?? defaults.locationLabelsEnabled,
        adaptiveScalesEnabled: props.adaptiveScalesEnabled ?? defaults.adaptiveScalesEnabled,
        flowLinesRenderingMode: props.flowLinesRenderingMode ?? defaults.flowLinesRenderingMode,
        clusteringEnabled: props.clusteringEnabled ?? defaults.clusteringEnabled,
        clusteringLevel: props.clusteringLevel,
        fadeEnabled: props.fadeEnabled ?? defaults.fadeEnabled,
        fadeOpacityEnabled: props.fadeOpacityEnabled ?? defaults.fadeOpacityEnabled,
        clusteringAuto: props.clusteringAuto ?? defaults.clusteringAuto,
        darkMode: props.darkMode ?? defaults.darkMode,
        fadeAmount: props.fadeAmount ?? defaults.fadeAmount,
        colorScheme: props.colorScheme,
        highlightColor: props.highlightColor ?? defaults.highlightColor,
        maxTopFlowsDisplayNum: props.maxTopFlowsDisplayNum ?? defaults.maxTopFlowsDisplayNum,
        flowEndpointsInViewportMode:
          props.flowEndpointsInViewportMode ?? defaults.flowEndpointsInViewportMode,
        flowLineThicknessScale: props.flowLineThicknessScale ?? defaults.flowLineThicknessScale,
        scaleLock: {
          enabled: Boolean(props.scaleLockEnabled && this.state?.lockedDomains),
          domains: this.state?.lockedDomains
        }
      }
    };
  }

  private async getFlowmapPickingInfo(info: PickInformation): Promise<PickInformation | undefined> {
    const {index, sourceLayer} = info;
    const {dataProvider} = this.props;
    const common = {
      ...info,
      picked: info.picked,
      layer: info.layer,
      index: info.index,
      x: info.x,
      y: info.y,
      coordinate: info.coordinate,
      event: info.event
    };
    if (isFlowLinesLayer(sourceLayer)) {
      const flow = index === -1 ? undefined : await dataProvider.getFlowByIndex(index);
      if (!flow) return undefined;
      const origin = await dataProvider.getLocationById(this.props.getFlowOriginId(flow));
      const dest = await dataProvider.getLocationById(this.props.getFlowDestId(flow));
      if (!origin || !dest) return undefined;
      return {
        ...common,
        object: {
          type: 'flow',
          flow,
          origin,
          dest,
          count: this.props.getFlowMagnitude(flow)
        } satisfies FlowmapPickingObject
      };
    }
    if (sourceLayer instanceof GPUFlowCirclesLayer) {
      const location = index === -1 ? undefined : await dataProvider.getLocationByIndex(index);
      const circleAttributes = this.state?.layersData?.circleAttributes;
      if (!location || !circleAttributes) return undefined;
      const id = this.props.getLocationId(location);
      const totals = await dataProvider.getTotalsForLocation(id);
      if (!totals) return undefined;
      return {
        ...common,
        object: {
          type: 'location',
          location,
          id,
          name: this.props.getLocationName(location),
          totals,
          circleRadius: getOuterCircleRadius(circleAttributes, index)
        } satisfies FlowmapPickingObject
      };
    }
    return undefined;
  }

  private getHighlightedObject(info: PickInformation): HighlightedObject | undefined {
    const {index, sourceLayer} = info;
    if (index < 0) return undefined;
    const {lineAttributes, circleAttributes} = this.state?.layersData ?? {};
    if (isFlowLinesLayer(sourceLayer) && lineAttributes) {
      return {
        type: 'flow',
        // The highlight color is always opaque, even when flows fade by opacity.
        lineAttributes: getFlowLineAttributesByIndex(
          lineAttributes,
          index,
          this.props.fadeOpacityEnabled ? 255 : undefined
        )
      };
    }
    if (sourceLayer instanceof GPUFlowCirclesLayer && circleAttributes) {
      return {
        type: 'location',
        coordinates: getCircleCoordinates(circleAttributes, index),
        radius: getOuterCircleRadius(circleAttributes, index)
      };
    }
    return undefined;
  }

  override renderLayers(): Layer[] {
    const props = this.props;
    const layersData = this.state?.layersData;
    if (!layersData?.circleAttributes || !layersData.lineAttributes) return [];
    const defaults = FlowmapGPULayer.defaultProps;
    const mode = props.flowLinesRenderingMode ?? defaults.flowLinesRenderingMode;
    const locationsEnabled = props.locationsEnabled ?? defaults.locationsEnabled;
    const thicknessUnit = 12 * (props.flowLineThicknessScale ?? defaults.flowLineThicknessScale);
    const curviness = props.flowLineCurviness ?? defaults.flowLineCurviness;
    const darkMode = props.darkMode ?? defaults.darkMode;
    const outlineColor = darkMode ? [0, 0, 0, 255] : [255, 255, 255, 255];
    const highlightColor = parseFlowmapColor(props.highlightColor ?? defaults.highlightColor);
    const highlightedObject = this.state.highlightedObject;
    const layers: Layer[] = [
      new GPUFlowLinesLayer(
        this.getSubLayerProps({
          nativeLayerId: this.props.nativeLayerId,
          id: mode === 'animated-straight' ? 'animated-flow-lines' : `${mode}-flow-lines`,
          data: layersData.lineAttributes,
          mode,
          drawOutline: mode !== 'animated-straight',
          outlineColor,
          thicknessUnit,
          curviness
        }) as never
      )
    ];
    if (locationsEnabled) {
      layers.push(
        new GPUFlowCirclesLayer(
          this.getSubLayerProps({
            nativeLayerId: this.props.nativeLayerId,
            id: 'circles',
            data: layersData.circleAttributes,
            emptyColor: darkMode ? [0, 0, 0, 255] : [255, 255, 255, 255],
            outlineEmptyMix: 0.4
          }) as never
        )
      );
    }
    if (highlightedObject?.type === 'location' && locationsEnabled) {
      layers.push(
        new ScatterplotLayer(
          this.getSubLayerProps({
            nativeLayerId: this.props.nativeLayerId,
            id: 'location-highlight',
            data: [highlightedObject],
            pickable: false,
            stroked: true,
            filled: false,
            lineWidthUnits: 'pixels',
            getLineWidth: 2,
            radiusUnits: 'pixels',
            getRadius: (object: {radius: number}) => object.radius,
            getLineColor: highlightColor,
            getPosition: (object: {coordinates: [number, number]}) => object.coordinates
          }) as never
        )
      );
    } else if (highlightedObject?.type === 'flow') {
      layers.push(
        new GPUFlowLinesLayer(
          this.getSubLayerProps({
            nativeLayerId: this.props.nativeLayerId,
            id: 'flow-highlight',
            data: highlightedObject.lineAttributes,
            mode: mode === 'curved' ? 'curved' : 'straight',
            drawOutline: true,
            pickable: false,
            outlineColor: highlightColor,
            outlineThickness: 1.5,
            thicknessUnit,
            curviness
          }) as never
        )
      );
    }
    const labels = this.state.labels;
    if (labels && (props.locationLabelsEnabled ?? false)) {
      const viewport = this.context.viewport as unknown as {
        longitude: number;
        latitude: number;
        zoom: number;
      };
      const style = props.locationLabelStyle ?? {};
      layers.push(
        ...labels.getLayers({
          id: `${this.props.id}-location-labels`,
          style: {
            color: style.color ?? (darkMode ? [255, 255, 255, 255] : [30, 30, 30, 255]),
            size: style.size ?? 14,
            outlineColor: style.outlineColor ?? (darkMode ? [0, 0, 0, 255] : [255, 255, 255, 255])
          },
          mapState: viewport
        })
      );
    }
    return layers;
  }
}
