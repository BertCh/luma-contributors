// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Resident drawing of decoded vector tiles.
 *
 * The live controller (vector-tile-live-controller.ts) forwards each admitted tile here. Tiles are
 * uploaded lazily when a device exists (`VectorTileResidentTile`), filtered and styled by a GPU
 * kernel, and drawn with the same GPU point, path and polygon layers that the resident GeoJSON
 * adapter uses, reading the geometry binding that `GeometryRuntime` produces. The controller turns
 * Deck's own MVT sublayer rendering off for the wrapped layer, so these layers are the only draw.
 *
 * Honest scope: drawing mirrors the resident GeoJSON adapter's GPU layers directly rather than
 * going through `createResidentLayerClasses`, which requires a Kepler table-backed layer and
 * dataset binding that a tile layer does not have. Draw layers are pickable; picks resolve to the tile
 * feature (vector-tile-picking.ts) and are reported through `onPick`, not into Kepler's hover state.
 */

import {COORDINATE_SYSTEM, type Color, type Layer, type PickingInfo} from '@deck.gl/core';
import {
  GPUPathLayer,
  GPUScatterplotLayer,
  GPUSolidPolygonLayer
} from '@deck.gl-community/gpu-layers';
import type {Device} from '@luma.gl/core';
import type {DecodedTileFeature, DecodedTileFilter} from './vector-tile-decoded-batch';
import type {NativeStyleLayer} from './native-style-plan';
import {
  planVectorTileStyle,
  type VectorTileExtraChannels,
  type VectorTileGpuStyle,
  type VectorTileStyleRender
} from './vector-tile-resident-style';
import {VectorTileResidentTile} from './vector-tile-resident-tile';
import {
  collectVectorTileStyleFields,
  deriveNativeStyleLayerFromDeckProps,
  deriveVectorTileConstants,
  deriveVectorTileDrawParameters,
  type VectorTileDeckProps,
  type VectorTileDrawParameters
} from './vector-tile-style-derive';
import {
  resolveVectorTilePick,
  type VectorTilePickInfo,
  type VectorTilePrimitiveKind
} from './vector-tile-picking';

export const VECTOR_TILE_DEFAULT_STYLE: VectorTileGpuStyle = {
  kind: 'constant',
  color: (18 | (147 << 8) | (154 << 16) | (255 << 24)) >>> 0
};
const MAXIMUM_TILE_UPLOADS_PER_FRAME = 4;
/** Labels beyond this get no code (drawn with the no-data color); bounds dictionary memory. */
const MAXIMUM_CATEGORY_COUNT = 4096;

type MemoryReservation = {release: () => void};

export type VectorTileResidentRendererOptions = {
  getDevice: () => Device | null;
  getReserveMemory: () =>
    | ((byteLength: number, memoryKind: 'cpu' | 'gpu', category?: 'source') => MemoryReservation)
    | null;
  /** Called when layers need a new render (pending uploads, filter or style changes). */
  onChange?: () => void;
  /** Called for hover and click on a resident draw layer; `pick` is null when nothing was hit. */
  onPick?: (event: {
    type: 'hover' | 'click';
    layerId: string;
    pick: VectorTilePickInfo | null;
    info: unknown;
  }) => void;
};

type TileEntry = {
  features: readonly DecodedTileFeature[];
  /** Registry ids per row (tile-independent); resolved from the feature when absent. */
  featureIds: readonly string[] | null;
  fieldNames: readonly string[];
  tile: VectorTileResidentTile | null;
  reservation: MemoryReservation | null;
  rejectedReason: string | null;
};

type LayerEntry = {
  layerId: string;
  tiles: Map<string, TileEntry>;
  filters: DecodedTileFilter[];
  style: VectorTileGpuStyle;
  render: Partial<VectorTileStyleRender>;
  channels: VectorTileExtraChannels;
  /** Draw settings read from the Deck layer; null keeps the native-plan defaults. */
  drawParameters: VectorTileDrawParameters | null;
  /** Native layer description the current style came from; re-planned as tiles and categories change. */
  nativeLayer: NativeStyleLayer | null;
  constants: {lineWidth?: number; radius?: number};
  /** Property names every tile must upload as columns for the style and filters. */
  requiredNumericFields: Set<string>;
  /** Categorical property fields with their append-only label to code dictionaries. */
  categories: Map<string, {labels: (string | number)[]; codes: Map<string | number, number>}>;
  styleDirty: boolean;
  styleReasons: readonly string[];
  lastPick: VectorTilePickInfo | null;
  visible: boolean;
  opacity: number;
  zoom: number;
  styleRevision: number;
};

function unpackColor(packed: number, alpha?: number): Color {
  return [
    packed & 255,
    (packed >>> 8) & 255,
    (packed >>> 16) & 255,
    alpha ?? (packed >>> 24) & 255
  ];
}

export class VectorTileResidentRenderer {
  private readonly layers = new Map<string, LayerEntry>();
  private device: Device | null = null;

  constructor(private readonly options: VectorTileResidentRendererOptions) {}

  /** Mirrors the wrapped Deck layer's visibility and opacity. */
  syncLayer(layerId: string, props: {visible?: boolean; opacity?: number}): void {
    const entry = this.getEntry(layerId);
    entry.visible = props.visible !== false;
    if (typeof props.opacity === 'number' && entry.render.opacity === undefined)
      entry.opacity = props.opacity;
  }

  setZoom(layerId: string, zoom: number): void {
    this.getEntry(layerId).zoom = zoom;
  }

  admitTile(
    layerId: string,
    tileKey: string,
    features: readonly DecodedTileFeature[],
    fieldNames: readonly string[],
    featureIds?: readonly string[]
  ): void {
    const entry = this.getEntry(layerId);
    this.releaseTile(entry, tileKey);
    if (this.extendCategories(entry, features) && entry.nativeLayer) entry.styleDirty = true;
    entry.tiles.set(tileKey, {
      features,
      featureIds: featureIds ?? null,
      fieldNames: [...fieldNames],
      tile: null,
      reservation: null,
      rejectedReason: null
    });
    this.options.onChange?.();
  }

  removeTile(layerId: string, tileKey: string): void {
    const entry = this.layers.get(layerId);
    if (!entry) return;
    this.releaseTile(entry, tileKey);
    entry.tiles.delete(tileKey);
    this.options.onChange?.();
  }

  setFilters(layerId: string, filters: readonly DecodedTileFilter[]): void {
    const entry = this.getEntry(layerId);
    entry.filters = [...filters];
    this.updateTiles(entry);
  }

  setStyle(
    layerId: string,
    style: VectorTileGpuStyle,
    render: Partial<VectorTileStyleRender> = {},
    channels: VectorTileExtraChannels = {}
  ): void {
    const entry = this.getEntry(layerId);
    entry.style = style;
    entry.render = render;
    entry.channels = channels;
    if (render.opacity !== undefined) entry.opacity = render.opacity;
    entry.styleRevision++;
    this.updateTiles(entry);
  }

  /** Translates a Kepler layer's channels with the native style plan and applies them. */
  setStyleFromNativeLayer(layerId: string, layer: NativeStyleLayer) {
    const entry = this.getEntry(layerId);
    entry.nativeLayer = layer;
    this.requireFields(entry, collectVectorTileStyleFields(layer, []));
    return this.planAndApplyStyle(entry);
  }

  /**
   * Derives the style, draw settings and required columns from the wrapped Deck layer's props
   * (Kepler's channel update triggers) and applies them. Filters are derived separately by the
   * controller, which also owns the CPU masks.
   */
  setStyleFromDeckProps(layerId: string, props: VectorTileDeckProps) {
    const entry = this.getEntry(layerId);
    entry.drawParameters = deriveVectorTileDrawParameters(props);
    entry.constants = deriveVectorTileConstants(props);
    entry.opacity = entry.drawParameters.opacity;
    const layer = deriveNativeStyleLayerFromDeckProps(layerId, props);
    entry.nativeLayer = layer;
    if (!layer) {
      this.options.onChange?.();
      return null;
    }
    this.requireFields(entry, collectVectorTileStyleFields(layer, []));
    return this.planAndApplyStyle(entry);
  }

  /** Makes sure columns used by filters exist on every tile. */
  requireFilterFields(layerId: string, filters: readonly DecodedTileFilter[]): void {
    this.requireFields(this.getEntry(layerId), collectVectorTileStyleFields(null, filters));
  }

  /** Re-plans a field-driven style against the tiles loaded now (domains follow the loaded values). */
  refreshStyle(layerId: string): void {
    const entry = this.layers.get(layerId);
    if (entry?.nativeLayer) this.planAndApplyStyle(entry);
  }

  getLastPick(layerId: string): VectorTilePickInfo | null {
    return this.layers.get(layerId)?.lastPick ?? null;
  }

  /** Test and diagnostics access to the GPU-computed mask for the loaded tiles of a layer. */
  async readFilterMasks(layerId: string): Promise<Record<string, number[]>> {
    const entry = this.layers.get(layerId);
    const result: Record<string, number[]> = {};
    if (!entry) return result;
    for (const [tileKey, tileEntry] of entry.tiles)
      if (tileEntry.tile) result[tileKey] = [...(await tileEntry.tile.readFilterMask())];
    return result;
  }

  async readColors(layerId: string): Promise<Record<string, number[]>> {
    const entry = this.layers.get(layerId);
    const result: Record<string, number[]> = {};
    if (!entry) return result;
    for (const [tileKey, tileEntry] of entry.tiles)
      if (tileEntry.tile) result[tileKey] = [...(await tileEntry.tile.readColors())];
    return result;
  }

  getState() {
    return [...this.layers.values()].map(entry => ({
      layerId: entry.layerId,
      style: entry.style,
      channels: entry.channels,
      styleReasons: entry.styleReasons,
      categoryCounts: Object.fromEntries(
        [...entry.categories].map(([fieldName, dictionary]) => [
          fieldName,
          dictionary.labels.length
        ])
      ),
      lastPick: entry.lastPick
        ? {
            tileKey: entry.lastPick.tileKey,
            featureId: entry.lastPick.featureId,
            properties: entry.lastPick.properties
          }
        : null,
      filters: entry.filters,
      visible: entry.visible,
      opacity: entry.opacity,
      pendingTileKeys: [...entry.tiles]
        .filter(([, tileEntry]) => !tileEntry.tile && !tileEntry.rejectedReason)
        .map(([tileKey]) => tileKey),
      residentTileKeys: [...entry.tiles]
        .filter(([, tileEntry]) => tileEntry.tile)
        .map(([tileKey]) => tileKey),
      rejectedTiles: [...entry.tiles]
        .filter(([, tileEntry]) => tileEntry.rejectedReason)
        .map(([tileKey, tileEntry]) => ({tileKey, reason: tileEntry.rejectedReason})),
      gpuByteLength: [...entry.tiles.values()].reduce(
        (total, tileEntry) => total + (tileEntry.tile?.gpuByteLength ?? 0),
        0
      )
    }));
  }

  /** Builds the draw layers; uploads a bounded number of pending tiles first. */
  getLayers(): Layer[] {
    const device = this.options.getDevice();
    if (device !== this.device) {
      this.destroyTiles();
      this.device = device;
    }
    if (!device) return [];
    const layers: Layer[] = [];
    for (const entry of this.layers.values()) {
      if (entry.styleDirty && entry.nativeLayer) this.planAndApplyStyle(entry);
      let uploadCount = 0;
      let hasPending = false;
      for (const [tileKey, tileEntry] of entry.tiles) {
        if (!tileEntry.tile && !tileEntry.rejectedReason) {
          if (uploadCount >= MAXIMUM_TILE_UPLOADS_PER_FRAME) {
            hasPending = true;
            continue;
          }
          uploadCount++;
          this.uploadTile(device, entry, tileKey, tileEntry);
        }
        if (tileEntry.tile)
          layers.push(...this.makeTileLayers(entry, tileEntry.tile, tileKey, tileEntry));
      }
      if (hasPending) queueMicrotask(() => this.options.onChange?.());
    }
    return layers;
  }

  destroy(): void {
    this.destroyTiles();
    this.layers.clear();
  }

  private getEntry(layerId: string): LayerEntry {
    let entry = this.layers.get(layerId);
    if (!entry) {
      entry = {
        layerId,
        tiles: new Map(),
        filters: [],
        style: VECTOR_TILE_DEFAULT_STYLE,
        render: {},
        channels: {},
        drawParameters: null,
        nativeLayer: null,
        constants: {},
        requiredNumericFields: new Set(),
        categories: new Map(),
        styleDirty: false,
        styleReasons: [],
        lastPick: null,
        visible: true,
        opacity: 0.8,
        zoom: 0,
        styleRevision: 0
      };
      this.layers.set(layerId, entry);
    }
    return entry;
  }

  private uploadTile(
    device: Device,
    entry: LayerEntry,
    tileKey: string,
    tileEntry: TileEntry
  ): void {
    try {
      const tile = VectorTileResidentTile.create({
        device,
        tileKey: `${entry.layerId}/${tileKey}`,
        features: tileEntry.features,
        fieldNames: this.getTileFieldNames(entry, tileEntry),
        filters: entry.filters,
        style: entry.style,
        channels: entry.channels,
        categories: this.getCategoryCodes(entry)
      });
      if (!tile) {
        tileEntry.rejectedReason = 'empty tile';
        return;
      }
      try {
        tileEntry.reservation =
          this.options.getReserveMemory()?.(Math.max(4, tile.gpuByteLength), 'gpu', 'source') ??
          null;
      } catch (error) {
        tile.destroy();
        tileEntry.rejectedReason = error instanceof Error ? error.message : String(error);
        return;
      }
      tileEntry.tile = tile;
    } catch (error) {
      tileEntry.rejectedReason = error instanceof Error ? error.message : String(error);
    }
  }

  private updateTiles(entry: LayerEntry): void {
    for (const tileEntry of entry.tiles.values())
      tileEntry.tile?.update(entry.filters, entry.style, entry.channels);
    this.options.onChange?.();
  }

  /** Appends unseen labels of the layer's categorical fields; returns whether any dictionary grew. */
  private extendCategories(entry: LayerEntry, features: readonly DecodedTileFeature[]): boolean {
    let grew = false;
    for (const [fieldName, dictionary] of entry.categories)
      for (const feature of features) {
        const value = feature.properties?.[fieldName];
        if (
          (typeof value === 'string' || typeof value === 'number') &&
          !dictionary.codes.has(value) &&
          dictionary.labels.length < MAXIMUM_CATEGORY_COUNT
        ) {
          dictionary.codes.set(value, dictionary.labels.length);
          dictionary.labels.push(value);
          grew = true;
        }
      }
    return grew;
  }

  /** Registers columns the style or filters read; tiles lacking one are re-uploaded. */
  private requireFields(
    entry: LayerEntry,
    fields: {numeric: readonly string[]; categorical: readonly string[]}
  ): void {
    let hasNewField = false;
    let hasNewCategory = false;
    for (const fieldName of fields.numeric)
      if (!entry.requiredNumericFields.has(fieldName)) {
        entry.requiredNumericFields.add(fieldName);
        hasNewField = true;
      }
    for (const fieldName of fields.categorical)
      if (!entry.categories.has(fieldName)) {
        entry.categories.set(fieldName, {labels: [], codes: new Map()});
        hasNewField = true;
        hasNewCategory = true;
      }
    if (!hasNewField) return;
    if (hasNewCategory)
      for (const tileEntry of entry.tiles.values())
        this.extendCategories(entry, tileEntry.features);
    for (const [tileKey, tileEntry] of entry.tiles) {
      const tile = tileEntry.tile;
      if (!tile) continue;
      const needed = this.getTileFieldNames(entry, tileEntry);
      if (hasNewCategory || needed.some(fieldName => !tile.fieldNames.includes(fieldName)))
        this.releaseTile(entry, tileKey);
    }
    this.options.onChange?.();
  }

  private getTileFieldNames(entry: LayerEntry, tileEntry: TileEntry): string[] {
    return [
      ...new Set([
        ...tileEntry.fieldNames,
        ...entry.requiredNumericFields,
        ...entry.categories.keys()
      ])
    ];
  }

  private getCategoryCodes(entry: LayerEntry): Map<string, Map<string | number, number>> {
    return new Map([...entry.categories].map(([fieldName, value]) => [fieldName, value.codes]));
  }

  private planAndApplyStyle(entry: LayerEntry) {
    const layer = entry.nativeLayer!;
    entry.styleDirty = false;
    const result = planVectorTileStyle(layer, {
      layerId: entry.layerId,
      zoom: entry.zoom,
      constants: entry.constants,
      getFieldValues: function* (fieldName) {
        for (const tileEntry of entry.tiles.values())
          for (const feature of tileEntry.features) {
            const value = feature.properties?.[fieldName];
            if (typeof value === 'number') yield value;
          }
      },
      getFieldCategories: fieldName => entry.categories.get(fieldName)?.labels ?? null
    });
    if (!result.supported) {
      entry.styleReasons = result.reasons;
      return result;
    }
    entry.styleReasons = [];
    const hasChanged =
      JSON.stringify([result.style, result.channels]) !==
      JSON.stringify([entry.style, entry.channels]);
    if (hasChanged || !entry.styleRevision)
      this.setStyle(entry.layerId, result.style, result.render, result.channels);
    return result;
  }

  private makePickHandlers(
    entry: LayerEntry,
    tileKey: string,
    tileEntry: TileEntry,
    primitive: VectorTilePrimitiveKind,
    resolveSourceRow?: (primitiveIndex: number) => number
  ) {
    const handle =
      (type: 'hover' | 'click') =>
      (info: PickingInfo): boolean => {
        const pick =
          info.index >= 0
            ? resolveVectorTilePick({
                layerId: entry.layerId,
                tileKey,
                primitive,
                primitiveIndex: info.index,
                features: tileEntry.features,
                featureIds: tileEntry.featureIds ?? undefined,
                resolveSourceRow,
                coordinate: info.coordinate ?? null
              })
            : null;
        if (pick || entry.lastPick?.tileKey === tileKey) entry.lastPick = pick;
        this.options.onPick?.({type, layerId: entry.layerId, pick, info});
        // Native hover state carries identity only; the host popover reads the pick.
        return false;
      };
    return {onHover: handle('hover'), onClick: handle('click')};
  }

  private releaseTile(entry: LayerEntry, tileKey: string): void {
    const tileEntry = entry.tiles.get(tileKey);
    if (!tileEntry) return;
    tileEntry.tile?.destroy();
    tileEntry.reservation?.release();
    tileEntry.tile = null;
    tileEntry.reservation = null;
  }

  private destroyTiles(): void {
    for (const entry of this.layers.values())
      for (const tileKey of entry.tiles.keys()) this.releaseTile(entry, tileKey);
  }

  private makeTileLayers(
    entry: LayerEntry,
    tile: VectorTileResidentTile,
    tileKey: string,
    tileEntry: TileEntry
  ): Layer[] {
    const {binding} = tile;
    const render = entry.render;
    const draw = entry.drawParameters;
    const idPrefix = `${entry.layerId}-resident-tile-${tile.tileKey}`;
    const constantColor = unpackColor(
      entry.style.kind === 'constant' ? entry.style.color : entry.style.noDataColor
    );
    const shared = {
      visible: entry.visible,
      pickable: draw?.pickable ?? true,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      // Tiles are flat map content: painter order, no depth interaction.
      parameters: {depthCompare: 'always' as const, depthWriteEnabled: false}
    };
    const layers: Layer[] = [];
    if (binding.polygons && (draw?.filled ?? true))
      layers.push(
        new GPUSolidPolygonLayer({
          ...shared,
          ...this.makePickHandlers(
            entry,
            tileKey,
            tileEntry,
            'polygons',
            binding.polygons.resolveSourceRow
          ),
          id: `${idPrefix}-fill`,
          opacity: entry.opacity,
          positions: binding.polygons.positions,
          indices: binding.polygons.indices,
          rowIndices: binding.polygons.rowIndices,
          selectionMask: binding.polygons.selectionMask,
          getFillColor: binding.polygons.getFillColor ?? constantColor,
          getElevation: 0,
          extruded: false,
          filled: true
        })
      );
    const lineProps = {
      ...shared,
      opacity: draw?.strokeOpacity ?? render.strokeOpacity ?? entry.opacity,
      widthUnits: draw?.lineWidthUnits ?? render.widthUnits ?? ('pixels' as const),
      widthScale: draw ? 1 : (render.widthScale ?? 1),
      billboard: false,
      miterLimit: 2
    };
    for (const [suffix, path, primitive] of [
      [
        'outline',
        draw?.stroked === false ? undefined : binding.polygonOutlines,
        'polygon-outlines'
      ],
      ['path', binding.paths, 'paths']
    ] as const) {
      if (!path || typeof path !== 'object' || !('getPath' in path)) continue;
      layers.push(
        new GPUPathLayer({
          ...lineProps,
          ...this.makePickHandlers(entry, tileKey, tileEntry, primitive, path.resolveSourceRow),
          id: `${idPrefix}-${suffix}`,
          selectionMask: path.selectionMask,
          getPath: path.getPath,
          getWidth: path.getWidth ?? 1,
          getColor: path.getColor ?? constantColor
        })
      );
    }
    if (binding.points)
      layers.push(
        new GPUScatterplotLayer({
          ...shared,
          ...this.makePickHandlers(
            entry,
            tileKey,
            tileEntry,
            'points',
            binding.points.resolveSourceRow
          ),
          id: `${idPrefix}-points`,
          opacity: entry.opacity,
          selectionMask: binding.points.selectionMask,
          getPosition: binding.points.positions3D ?? binding.points.positions,
          getFillColor: binding.points.colors ?? constantColor,
          getRadius: binding.points.sizes ?? 3,
          radiusUnits: draw?.radiusUnits ?? 'pixels',
          radiusScale: draw?.radiusScale ?? 1,
          radiusMinPixels: draw?.radiusMinPixels ?? 0,
          getLineColor: binding.points.strokeColors ?? constantColor,
          getLineWidth: binding.points.lineWidths ?? 1,
          lineWidthUnits: draw?.lineWidthUnits ?? 'pixels',
          filled: draw?.filled ?? true,
          stroked: draw ? draw.stroked : false
        })
      );
    return layers;
  }
}
