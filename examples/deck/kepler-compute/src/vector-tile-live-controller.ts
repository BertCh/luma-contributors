// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Live wiring of the vector-tile working set to Kepler's native vectorTile (MVT) layer.
 *
 * The host's layer pipeline passes every Deck layer through `wrapLayer`. A Deck `MVTLayer` is cloned
 * with chained `onTileLoad`, `onTileUnload` and `onViewportLoad` callbacks (Kepler's own handlers
 * keep running). Each decoded tile is admitted to a `VectorTileWorkingSet` with a cpu memory
 * reservation, receives stable feature ids, is converted to columnar batches, and the batch
 * positions plus a per-feature filter mask are uploaded to GPU buffers (with a gpu reservation).
 *
 * Honest scope: Deck still draws the tile geometry through its own sublayers. The GPU buffers here
 * are the resident population for host filtering/analysis; routing the tile batches through the
 * resident point/line/polygon adapters (selection masks, native style plan) is not done, and
 * filters use the CPU mask planning of vector-tile-decoded-batch.ts as the source of truth.
 */

import {
  buildDecodedTileBatches,
  computeDecodedTileFilterMask,
  type DecodedTileBatch,
  type DecodedTileFeature,
  type DecodedTileFilter
} from './vector-tile-decoded-batch';
import {
  decideVectorTileAnalysis,
  describeVectorTileAnalysisScope,
  type VectorTileAnalysisDecision,
  type VectorTileAnalysisRequest,
  type VectorTileAnalysisScope
} from './vector-tile-analysis-scope';
import type {VectorTileResidentRenderer} from './vector-tile-resident-layers';
import {Tileset2D} from '@deck.gl/geo-layers';
import {createVectorTileVisibilityTilesetClass} from './vector-tile-visibility';
import {isGlobeBasemapVectorTile, renderGlobeBasemapTile} from './globe-basemap-tiles';
import {computeVectorTileCacheSize} from './vector-tile-cache-budget';
import {
  collectVectorTileStyleFields,
  deriveNativeStyleLayerFromDeckProps,
  deriveVectorTileFiltersFromDeckProps,
  getVectorTileDeckStyleSignature,
  hasVectorTileStyleTriggers
} from './vector-tile-style-derive';
import {
  VectorTileWorkingSet,
  type VectorTileReservation,
  type VectorTileReserveMemory
} from './vector-tile-working-set';

export const VECTOR_TILE_LIVE_DEFAULT_BUDGET_BYTES = 64 * 1024 * 1024;
const MAXIMUM_NUMERIC_FIELD_COUNT = 8;
const FIELD_DISCOVERY_FEATURE_COUNT = 64;

/** Minimal GPU buffer surface so the controller is testable without a device. */
export type VectorTileGpuBuffer = {
  readonly byteLength: number;
  write: (data: Uint32Array) => void;
  destroy: () => void;
};

export type VectorTileLiveOptions = Readonly<{
  budgetBytes?: number;
  /** Current registry reserveMemory; may return null before a runtime exists. */
  getReserveMemory: () =>
    | ((
        byteLength: number,
        memoryKind: 'cpu' | 'gpu',
        category?: 'source' | 'output' | 'scratch' | 'readback'
      ) => VectorTileReservation)
    | null;
  /** Creates a vertex/storage buffer from typed data; null when no device is ready. */
  getCreateBuffer: () => ((data: Float32Array | Uint32Array) => VectorTileGpuBuffer) | null;
  onChange?: () => void;
  /**
   * When present, admitted tiles are drawn by resident GPU layers (GPU filter mask and style) and
   * Deck's own MVT sublayer rendering is suppressed for wrapped layers.
   */
  residentRenderer?: VectorTileResidentRenderer;
  /**
   * Derive style, draw settings and filters from the wrapped Deck layer (Kepler's channel update
   * triggers and GPU filter ranges) whenever they change. Default true; needs a resident renderer.
   */
  autoStyle?: boolean;
  /** Bound Deck's TileLayer cache by the working set budget. Default true. */
  limitDeckCache?: boolean;
}>;

type TileResident = {
  features: readonly DecodedTileFeature[];
  batches: readonly DecodedTileBatch[];
  masks: Uint8Array[];
  positionBuffers: VectorTileGpuBuffer[];
  maskBuffers: VectorTileGpuBuffer[];
  gpuReservation?: VectorTileReservation;
  gpuByteLength: number;
};

type LayerState = {
  layerId: string;
  workingSet: VectorTileWorkingSet;
  residents: Map<string, TileResident>;
  visibleTileKeys: string[];
  zoom: number;
  filters: DecodedTileFilter[];
  numericFieldNames: string[];
  loadCount: number;
  unloadCount: number;
  gpuByteLength: number;
  /** Signature of the Deck props the current derived style and filters came from. */
  styleSignature: string | null;
  sourceTilesetClass?: typeof Tileset2D;
  visibilityTilesetClass?: typeof Tileset2D;
  hasNativeVisibility?: boolean;
};

type TileLike = {
  index?: {x: number; y: number; z: number};
  id?: string;
  content?: unknown;
};

type LayerLike = {
  id: string;
  props: Record<string, unknown>;
  constructor: {layerName?: string};
  clone: (props: Record<string, unknown>) => unknown;
};

const WRAPPED_MARK = Symbol.for('kepler.vectorTileLive');

export function isVectorTileDeckLayer(layer: unknown): layer is LayerLike {
  const candidate = layer as LayerLike | null;
  return (
    Boolean(candidate) &&
    typeof candidate?.clone === 'function' &&
    candidate.constructor?.layerName === 'MVTLayer'
  );
}

function getTileKey(tile: TileLike): string {
  const index = tile.index;
  return index ? `${index.z}/${index.x}/${index.y}` : String(tile.id);
}

function countCoordinates(coordinates: unknown): number {
  if (!Array.isArray(coordinates)) return 0;
  if (typeof coordinates[0] === 'number') return 1;
  let count = 0;
  for (const child of coordinates) count += countCoordinates(child);
  return count;
}

function toDecodedFeatures(content: unknown): DecodedTileFeature[] {
  if (!Array.isArray(content)) return [];
  return content.filter(feature => feature && typeof feature === 'object') as DecodedTileFeature[];
}

function discoverNumericFieldNames(features: readonly DecodedTileFeature[]): string[] {
  const names = new Set<string>();
  for (const feature of features.slice(0, FIELD_DISCOVERY_FEATURE_COUNT))
    for (const [name, value] of Object.entries(feature.properties ?? {}))
      if (typeof value === 'number' && names.size < MAXIMUM_NUMERIC_FIELD_COUNT) names.add(name);
  return [...names];
}

function estimateTileByteLength(features: readonly DecodedTileFeature[]): number {
  let byteLength = 0;
  for (const feature of features) {
    byteLength += 64 + countCoordinates(feature.geometry?.coordinates) * 16;
    byteLength += feature.properties ? JSON.stringify(feature.properties).length * 2 : 0;
  }
  return byteLength;
}

export class VectorTileLiveController {
  private readonly layers = new Map<string, LayerState>();
  private readonly handlers = new Map<string, Record<string, unknown>>();

  constructor(private readonly options: VectorTileLiveOptions) {}

  /** Returns the layer unchanged unless it is a Deck MVT layer; MVT layers are cloned with hooks. */
  wrapLayer<LayerType>(layer: LayerType): LayerType {
    if (!isVectorTileDeckLayer(layer)) return layer;
    // Basemap accessors and tile-local positions belong to deck's MVT draw path, not the resident
    // lng/lat analysis population. Let MVT retain its own selected/cached tile visibility too.
    if (isGlobeBasemapVectorTile(layer)) {
      return layer.clone({renderSubLayers: renderGlobeBasemapTile}) as LayerType;
    }
    const original = layer.props;
    if ((original['onTileLoad'] as {[WRAPPED_MARK]?: boolean} | undefined)?.[WRAPPED_MARK])
      return layer;
    const layerId = layer.id;
    const state = this.getLayerState(layerId);
    const chain = (name: string, own: (argument: never) => void) => {
      const wrapped = (argument: never) => {
        (original[name] as ((value: never) => void) | undefined)?.(argument);
        own(argument);
      };
      (wrapped as unknown as Record<symbol, boolean>)[WRAPPED_MARK] = true;
      return wrapped;
    };
    const renderer = this.options.residentRenderer;
    if (renderer) {
      const sourceTilesetClass =
        (original['TilesetClass'] as typeof Tileset2D | undefined) ?? Tileset2D;
      if (state.sourceTilesetClass !== sourceTilesetClass) {
        state.sourceTilesetClass = sourceTilesetClass;
        state.visibilityTilesetClass = createVectorTileVisibilityTilesetClass(
          sourceTilesetClass,
          tileKeys => {
            state.hasNativeVisibility = true;
            renderer.setVisibleTileKeys?.(layerId, tileKeys);
          }
        );
      }
    }
    renderer?.syncLayer(layerId, original as {visible?: boolean; opacity?: number});
    if (renderer && this.options.autoStyle !== false) this.syncDerivedStyle(state, original);
    const handlers = {
      ...(this.options.limitDeckCache === false
        ? {}
        : {
            maxCacheSize: computeVectorTileCacheSize({
              budgetBytes: this.options.budgetBytes ?? VECTOR_TILE_LIVE_DEFAULT_BUDGET_BYTES,
              usedBytes: state.workingSet.getStatus().usedBytes,
              residentTileCount: state.workingSet.getStatus().tileCount,
              visibleTileCount: state.visibleTileKeys.length,
              existingMaxCacheSize:
                typeof original['maxCacheSize'] === 'number' ? original['maxCacheSize'] : null
            })
          }),
      // Deck keeps fetching, decoding and caching tiles; only its per-tile sublayer drawing is off.
      ...(renderer ? {renderSubLayers: () => [], TilesetClass: state.visibilityTilesetClass} : {}),
      onTileLoad: chain('onTileLoad', (tile: TileLike) => this.loadTile(state, tile)),
      onTileUnload: chain('onTileUnload', (tile: TileLike) => this.unloadTile(state, tile)),
      onViewportLoad: chain('onViewportLoad', (tiles: TileLike[] | null) =>
        this.updateViewport(state, tiles ?? [])
      )
    };
    this.handlers.set(layerId, handlers);
    return layer.clone(handlers) as LayerType;
  }

  setFilters(layerId: string, filters: readonly DecodedTileFilter[]): void {
    const state = this.layers.get(layerId);
    if (!state) return;
    state.filters = [...filters];
    for (const resident of state.residents.values()) this.refreshMasks(resident, state.filters);
    this.options.residentRenderer?.setFilters(layerId, state.filters);
    this.options.onChange?.();
  }

  /** Last hovered or clicked feature of a layer, as resolved by the resident renderer. */
  getLastPick(layerId: string) {
    return this.options.residentRenderer?.getLastPick?.(layerId) ?? null;
  }

  getLayerIds(): string[] {
    return [...this.layers.keys()];
  }

  getScope(layerId: string): VectorTileAnalysisScope | null {
    const state = this.layers.get(layerId);
    if (!state) return null;
    const status = state.workingSet.getStatus();
    return describeVectorTileAnalysisScope({
      datasetId: layerId,
      zoom: state.zoom,
      visibleTileKeys: state.visibleTileKeys,
      loadedTileKeys: state.workingSet.getResidentTileKeys(),
      rejectedTileKeys: status.rejectedTileKeys,
      featureSummary: state.workingSet.featureRegistry?.summarize(),
      isSourceFullyMaterialized: false
    });
  }

  /** Analysis entry point: viewport populations are refused for complete-dataset analyses. */
  decideAnalysis(
    layerId: string,
    request: VectorTileAnalysisRequest
  ): VectorTileAnalysisDecision | null {
    const scope = this.getScope(layerId);
    return scope ? decideVectorTileAnalysis(scope, request) : null;
  }

  getState() {
    return [...this.layers.values()].map(state => {
      const scope = this.getScope(state.layerId);
      let featureCount = 0;
      let passingFeatureCount = 0;
      for (const resident of state.residents.values())
        resident.batches.forEach((batch, index) => {
          featureCount += batch.featureIds.length;
          for (const flag of resident.masks[index]) passingFeatureCount += flag;
        });
      return {
        layerId: state.layerId,
        workingSet: state.workingSet.getStatus(),
        residentTileKeys: state.workingSet.getResidentTileKeys(),
        visibleTileKeys: state.visibleTileKeys,
        featureSummary: state.workingSet.featureRegistry?.summarize() ?? null,
        scope,
        completeDatasetDecision: scope
          ? decideVectorTileAnalysis(scope, {requiresCompleteDataset: true})
          : null,
        numericFieldNames: state.numericFieldNames,
        filters: state.filters,
        featureCount,
        passingFeatureCount,
        gpuByteLength: state.gpuByteLength,
        loadCount: state.loadCount,
        unloadCount: state.unloadCount
      };
    });
  }

  destroy(): void {
    for (const state of this.layers.values()) state.workingSet.destroy();
    this.layers.clear();
    this.handlers.clear();
  }

  private getLayerState(layerId: string): LayerState {
    let state = this.layers.get(layerId);
    if (state) return state;
    const created: LayerState = {
      layerId,
      residents: new Map(),
      visibleTileKeys: [],
      zoom: 0,
      filters: [],
      numericFieldNames: [],
      loadCount: 0,
      unloadCount: 0,
      gpuByteLength: 0,
      styleSignature: null,
      workingSet: new VectorTileWorkingSet({
        budgetBytes: this.options.budgetBytes ?? VECTOR_TILE_LIVE_DEFAULT_BUDGET_BYTES,
        featureIdOptions: {sourceLayer: layerId},
        reserveMemory: ((byteLength, memoryKind, category) => {
          const reserveMemory = this.options.getReserveMemory();
          if (!reserveMemory) return {resize() {}, release() {}};
          return reserveMemory(byteLength, memoryKind, category);
        }) as VectorTileReserveMemory,
        onEvictTile: tileKey => this.releaseResident(created, tileKey)
      })
    };
    state = created;
    this.layers.set(layerId, state);
    return state;
  }

  private loadTile(state: LayerState, tile: TileLike): void {
    const tileKey = getTileKey(tile);
    const features = toDecodedFeatures(tile.content);
    state.loadCount++;
    this.releaseResident(state, tileKey);
    for (const name of discoverNumericFieldNames(features))
      if (
        !state.numericFieldNames.includes(name) &&
        state.numericFieldNames.length < MAXIMUM_NUMERIC_FIELD_COUNT
      )
        state.numericFieldNames.push(name);
    state.workingSet.setVisibleTileKeys(state.visibleTileKeys);
    const admission = state.workingSet.admitTile(
      tileKey,
      estimateTileByteLength(features),
      features
    );
    if (admission.admitted) {
      const batches = buildDecodedTileBatches(features, {
        sourceLayer: state.layerId,
        numericFieldNames: state.numericFieldNames
      });
      const resident: TileResident = {
        features,
        batches,
        masks: [],
        positionBuffers: [],
        maskBuffers: [],
        gpuByteLength: 0
      };
      this.uploadResident(state, resident);
      this.refreshMasks(resident, state.filters);
      state.residents.set(tileKey, resident);
      this.options.residentRenderer?.admitTile(
        state.layerId,
        tileKey,
        features,
        state.numericFieldNames,
        state.workingSet.featureRegistry?.getTileFeatureIds(tileKey)
      );
    }
    this.options.onChange?.();
  }

  private unloadTile(state: LayerState, tile: TileLike): void {
    const tileKey = getTileKey(tile);
    state.unloadCount++;
    state.workingSet.removeTile(tileKey, false);
    this.releaseResident(state, tileKey);
    this.options.onChange?.();
  }

  private updateViewport(state: LayerState, tiles: readonly TileLike[]): void {
    state.visibleTileKeys = tiles.map(getTileKey);
    // Controllers without a live native traversal (e.g. externally supplied decoded tiles) use
    // the declared viewport. Native traversal reports refinement visibility, which takes precedence.
    if (!state.hasNativeVisibility)
      this.options.residentRenderer?.setVisibleTileKeys?.(state.layerId, state.visibleTileKeys);
    state.zoom = tiles[0]?.index?.z ?? state.zoom;
    this.options.residentRenderer?.setZoom(state.layerId, state.zoom);
    // Domains follow the loaded values, so a field-driven style is re-planned once the viewport settles.
    this.options.residentRenderer?.refreshStyle?.(state.layerId);
    state.workingSet.setVisibleTileKeys(state.visibleTileKeys);
    for (const tileKey of state.visibleTileKeys) state.workingSet.touchTile(tileKey);
    this.options.onChange?.();
  }

  /**
   * Re-derives style, draw settings and filters from the Deck layer's props when Kepler changed any
   * of them. The first sight of a layer always derives; later renders with identical props do nothing,
   * so filters set through setFilters stay until Kepler's own filters change.
   */
  private syncDerivedStyle(state: LayerState, props: Record<string, unknown>): void {
    if (!hasVectorTileStyleTriggers(props)) return;
    const signature = getVectorTileDeckStyleSignature(props);
    if (signature === state.styleSignature) return;
    state.styleSignature = signature;
    const filters = deriveVectorTileFiltersFromDeckProps(props);
    const fields = collectVectorTileStyleFields(
      deriveNativeStyleLayerFromDeckProps(state.layerId, props),
      filters
    );
    this.requireNumericFields(state, [...fields.numeric, ...fields.categorical]);
    const renderer = this.options.residentRenderer;
    renderer?.requireFilterFields?.(state.layerId, filters);
    renderer?.setStyleFromDeckProps?.(state.layerId, props);
    this.setFilters(state.layerId, filters);
  }

  /** Adds property columns the CPU batches must carry (filters read them); rebuilds resident batches. */
  private requireNumericFields(state: LayerState, fieldNames: readonly string[]): void {
    let hasNewField = false;
    for (const fieldName of fieldNames)
      if (!state.numericFieldNames.includes(fieldName)) {
        state.numericFieldNames.push(fieldName);
        hasNewField = true;
      }
    if (!hasNewField) return;
    for (const resident of state.residents.values()) {
      resident.batches = buildDecodedTileBatches(resident.features, {
        sourceLayer: state.layerId,
        numericFieldNames: state.numericFieldNames
      });
      this.refreshMasks(resident, state.filters);
    }
  }

  private uploadResident(state: LayerState, resident: TileResident): void {
    const createBuffer = this.options.getCreateBuffer();
    if (!createBuffer) return;
    let byteLength = 0;
    for (const batch of resident.batches)
      byteLength += batch.positions.byteLength + batch.featureIds.length * 4;
    try {
      resident.gpuReservation = this.options.getReserveMemory()?.(
        Math.max(4, byteLength),
        'gpu',
        'source'
      );
    } catch {
      // Over the gpu budget: keep the tile CPU resident only.
      return;
    }
    for (const batch of resident.batches) {
      resident.positionBuffers.push(createBuffer(batch.positions));
      resident.maskBuffers.push(
        createBuffer(new Uint32Array(Math.max(1, batch.featureIds.length)))
      );
    }
    resident.gpuByteLength = byteLength;
    state.gpuByteLength += byteLength;
  }

  private refreshMasks(resident: TileResident, filters: readonly DecodedTileFilter[]): void {
    resident.masks = resident.batches.map(batch => computeDecodedTileFilterMask(batch, filters));
    resident.masks.forEach((mask, index) => {
      resident.maskBuffers[index]?.write(Uint32Array.from(mask));
    });
  }

  private releaseResident(state: LayerState, tileKey: string): void {
    const resident = state.residents.get(tileKey);
    if (!resident) return;
    state.residents.delete(tileKey);
    this.options.residentRenderer?.removeTile(state.layerId, tileKey);
    for (const buffer of [...resident.positionBuffers, ...resident.maskBuffers]) buffer.destroy();
    resident.gpuReservation?.release();
    state.gpuByteLength -= resident.gpuByteLength;
  }
}
