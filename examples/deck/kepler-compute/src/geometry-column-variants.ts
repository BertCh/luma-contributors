// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {PreparedGeometryGPUVectors} from '@luma.gl/engine/arrow';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {makeGeometryGPUVectors} from '@luma.gl/engine/arrow';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerDatasetSnapshot,
  KeplerMemoryReservation,
  KeplerSelectionLayer
} from '@luma-contributors/kepler';

import {isCellLayerType} from './cell-layer-types';
import type {NativeCellIndex} from './cell-layer-types';
import {appendDGGSFootprintColumn} from './dggs-source';
import {makeH3GPUVectors, type H3GPUVectors} from './h3-runtime';
import {preparePublishedGeometry, preparePublishedH3} from './source-publication';

/** What the dataset's single published geometry role set already serves. */
export type GeometryVariantPrimary = {
  geometryColumn?: string;
  h3Column?: string;
  cellIndex?: NativeCellIndex;
  /** True when the geometry role is a timed trip column (no polygon geometry is resident). */
  isTrip: boolean;
  /** True when a planar polygon geometry (or its released summary) is published. */
  hasGeometry: boolean;
  hasH3: boolean;
};

export type GeometryVariantLayer = {
  id: string;
  type: string | null;
  config: {isVisible?: boolean; columns: object};
};

type GeometryVariantKind = 'geojson' | 'h3' | 'cell';

type GeometryVariantRequest = {
  kind: GeometryVariantKind;
  /** GeoJSON/WKB column, H3 cell column or cell token column. */
  column: string;
  family?: string;
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
};

export type GeometryColumnVariant = {
  /** Monotonic, usable as a selector geometry revision because a variant never mutates. */
  generation: number;
  kind: GeometryVariantKind;
  column: string;
  /** Column the style host reads geometry from (the derived footprint for cell tokens). */
  geometryColumn?: string;
  geometry?: PreparedGeometryGPUVectors;
  h3?: H3GPUVectors;
  bounds?: [number, number, number, number];
  gpuByteLength: number;
};

export type GeometryColumnVariantResult =
  | {status: 'ready'; variant: GeometryColumnVariant}
  | {status: 'unavailable'; reason: string};

type VariantEntry = {
  key: string;
  datasetId: string;
  variant: GeometryColumnVariant;
  layerIds: Set<string>;
  lease: KeplerDatasetLease;
  reservations: KeplerMemoryReservation[];
};

function getFootprintColumnName(tokenColumn: string, family: string): string {
  return `__variant_${family}_${tokenColumn}_footprint`;
}

function getArrowDataBytes(data: arrow.Data): number {
  return (
    (data.values?.byteLength ?? 0) +
    (data.valueOffsets?.byteLength ?? 0) +
    (data.nullBitmap?.byteLength ?? 0) +
    (data.children ?? []).reduce((bytes, child) => bytes + getArrowDataBytes(child), 0)
  );
}

/** Variant key: one immutable upload of one geometry column of one dataset revision. */
export function getGeometryColumnVariantKey(request: GeometryVariantRequest): string {
  return [
    request.dataset.id,
    request.dataset.revision,
    request.kind,
    request.family ?? '',
    request.column
  ].join('\u0000');
}

/**
 * Additional resident geometry for layers bound to a geometry, H3 or cell-token column other than
 * the dataset's primary published one. Prepared on demand through the same preparation and GPU
 * upload path as publication (memory is reserved in the session registry as 'source' bytes),
 * shared by every layer using the column, and destroyed once no layer references it and the GPU
 * has finished the frames that used it. The style host expands per-layer style channels from the
 * variant exactly as it does from the primary geometry.
 */
export class GeometryColumnVariantRuntime {
  private readonly entries = new Map<string, VariantEntry>();
  private readonly layerKeys = new Map<string, string>();
  private readonly failures = new Map<string, string>();
  private readonly retirements = new Set<Promise<void>>();
  private readonly errors: unknown[] = [];
  private generation = 0;
  private destroyed = false;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly requestUpdate: () => void
  ) {}

  get variantCount(): number {
    return this.entries.size;
  }

  get gpuByteLength(): number {
    let total = 0;
    for (const entry of this.entries.values()) total += entry.variant.gpuByteLength;
    return total;
  }

  getVariant(layerId: string): GeometryColumnVariant | undefined {
    const key = this.layerKeys.get(layerId);
    return key ? this.entries.get(key)?.variant : undefined;
  }

  get hasUnreferencedVariants(): boolean {
    for (const entry of this.entries.values()) if (!entry.layerIds.size) return true;
    return false;
  }

  resolve(layerId: string, request: GeometryVariantRequest): GeometryColumnVariantResult {
    if (this.destroyed || this.session.deviceLost)
      return {status: 'unavailable', reason: 'Geometry column variants are unavailable.'};
    const key = getGeometryColumnVariantKey(request);
    const previousKey = this.layerKeys.get(layerId);
    if (previousKey && previousKey !== key) this.entries.get(previousKey)?.layerIds.delete(layerId);
    const existing = this.entries.get(key);
    if (existing && existing.lease.isValid()) {
      existing.layerIds.add(layerId);
      this.layerKeys.set(layerId, key);
      return {status: 'ready', variant: existing.variant};
    }
    const failure = this.failures.get(key);
    if (failure) return {status: 'unavailable', reason: failure};
    const reservations: KeplerMemoryReservation[] = [];
    let lease: KeplerDatasetLease | undefined;
    let geometry: PreparedGeometryGPUVectors | undefined;
    let h3: H3GPUVectors | undefined;
    try {
      lease = this.session.registry.acquireDataset(request.dataset.id);
      const source = request.dataset.value.source;
      const variant: GeometryColumnVariant = {
        generation: ++this.generation,
        kind: request.kind,
        column: request.column,
        gpuByteLength: 0
      };
      if (request.kind === 'h3') {
        const preparedCells = preparePublishedH3(
          source,
          request.column,
          request.dataset.batches.map(batch => batch.id)
        );
        h3 = makeH3GPUVectors(this.session.device, preparedCells, {
          memoryRegistry: this.session.registry,
          workingMemoryBudgetBytes:
            4 * 1024 * 1024 +
            preparedCells.typedStorageByteLength * 64 +
            request.dataset.rowCount * 128
        });
        variant.h3 = h3;
        variant.gpuByteLength = h3.gpuByteLength;
        const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
        for (const batch of preparedCells.batches)
          for (let row = 0; row < batch.rowCount; row++) {
            if (!batch.validity[row]) continue;
            bounds[0] = Math.min(bounds[0], batch.centers[row * 2]);
            bounds[1] = Math.min(bounds[1], batch.centers[row * 2 + 1]);
            bounds[2] = Math.max(bounds[2], batch.centers[row * 2]);
            bounds[3] = Math.max(bounds[3], batch.centers[row * 2 + 1]);
          }
        if (bounds.every(Number.isFinite)) variant.bounds = bounds;
      } else {
        let table = source;
        let geometryColumn = request.column;
        if (request.kind === 'cell') {
          geometryColumn = getFootprintColumnName(request.column, request.family!);
          table = appendDGGSFootprintColumn(
            source,
            request.column,
            request.family as never,
            geometryColumn
          ).table;
        }
        const prepared = preparePublishedGeometry(table, geometryColumn);
        const sourceBytes = prepared.batches.reduce(
          (bytes, batch) =>
            bytes +
            batch.points.positions.byteLength +
            batch.paths.positions.byteLength +
            batch.polygonOutlines.positions.byteLength +
            (batch.polygonOutlines.sideOrientations?.byteLength ?? 0) +
            (batch.representative?.positions.byteLength ?? 0) +
            (batch.representative?.validity.byteLength ?? 0) +
            Object.values(batch.properties ?? {}).reduce(
              (propertyBytes, property) =>
                propertyBytes + property.values.byteLength + property.validity.byteLength,
              0
            ) +
            batch.polygons.data.reduce((sum, data) => sum + getArrowDataBytes(data), 0),
          0
        );
        const budget = 8 * 1024 * 1024 + sourceBytes * 64 + request.dataset.rowCount * 128;
        reservations.push(this.session.registry.reserveMemory(budget, 'gpu', 'source'));
        geometry = makeGeometryGPUVectors(this.session.device, prepared);
        // Mapping bytes are CPU-resident index structures retained with the GPU vectors.
        reservations.push(
          this.session.registry.reserveMemory(
            geometry.mappingByteLength + request.dataset.rowCount * 64 + 4096,
            'cpu',
            'source'
          )
        );
        if (geometry.gpuByteLength > budget)
          throw new Error('Prepared geometry exceeds the admitted geometry budget');
        variant.geometry = geometry;
        variant.geometryColumn = geometryColumn;
        variant.gpuByteLength = geometry.gpuByteLength;
        if (prepared.metadata.bounds) variant.bounds = [...prepared.metadata.bounds];
      }
      this.entries.set(key, {
        key,
        datasetId: request.dataset.id,
        variant,
        layerIds: new Set([layerId]),
        lease,
        reservations
      });
      this.layerKeys.set(layerId, key);
      this.requestUpdate();
      return {status: 'ready', variant};
    } catch (error) {
      geometry?.destroy();
      if (h3) this.track(h3.destroy(Promise.resolve()));
      for (const reservation of reservations) reservation.release();
      if (lease) this.track(lease.release());
      const reason = `Geometry column ${request.column} cannot be prepared: ${
        error instanceof Error ? error.message : String(error)
      }`;
      this.failures.set(key, reason);
      return {status: 'unavailable', reason};
    }
  }

  releaseLayer(layerId: string): void {
    const key = this.layerKeys.get(layerId);
    if (!key) return;
    this.layerKeys.delete(layerId);
    this.entries.get(key)?.layerIds.delete(layerId);
  }

  async retainLayers(ids: readonly string[], completion: PromiseLike<unknown>): Promise<void> {
    const retained = new Set(ids);
    for (const layerId of [...this.layerKeys.keys()])
      if (!retained.has(layerId)) this.releaseLayer(layerId);
    for (const [key, entry] of this.entries)
      if (!entry.layerIds.size) {
        this.entries.delete(key);
        this.retire(entry, completion);
      }
    if (this.failures.size > 32) this.failures.clear();
    await this.drain();
  }

  async destroy(completion: PromiseLike<unknown>): Promise<void> {
    this.destroyed = true;
    await this.retainLayers([], completion);
  }

  private retire(entry: VariantEntry, completion: PromiseLike<unknown>): void {
    this.track(
      (async () => {
        try {
          await completion;
          entry.variant.geometry?.destroy();
          await entry.variant.h3?.destroy(Promise.resolve());
        } finally {
          for (const reservation of entry.reservations) reservation.release();
          await entry.lease.release();
        }
      })()
    );
  }

  private track(promise: Promise<void>): void {
    this.retirements.add(promise);
    void promise
      .catch(error => this.errors.push(error))
      .finally(() => this.retirements.delete(promise));
  }

  private async drain(): Promise<void> {
    await Promise.allSettled([...this.retirements]);
    if (this.errors.length)
      throw new AggregateError(this.errors.splice(0), 'Geometry variant cleanup failed');
  }
}

/** Describes the geometry column a visible layer needs when it differs from the primary one. */
export function getGeometryVariantRequest(
  layer: GeometryVariantLayer,
  primary: GeometryVariantPrimary,
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>
): GeometryVariantRequest | undefined {
  const columns = layer.config.columns as Record<string, {value?: string | null} | undefined>;
  if (layer.type === 'hexagonId') {
    const column = columns['hex_id']?.value;
    return column && !(primary.hasH3 && primary.h3Column === column)
      ? {kind: 'h3', column, dataset}
      : undefined;
  }
  if (layer.type === 'geojson') {
    const column = columns['geojson']?.value;
    const isPrimary =
      primary.hasGeometry &&
      !primary.isTrip &&
      !primary.cellIndex &&
      primary.geometryColumn === column;
    return column && !isPrimary ? {kind: 'geojson', column, dataset} : undefined;
  }
  if (isCellLayerType(layer.type)) {
    const column = columns['token']?.value;
    const isPrimary =
      primary.hasGeometry &&
      primary.cellIndex?.column === column &&
      primary.cellIndex?.family === layer.type;
    return column && !isPrimary ? {kind: 'cell', column, family: layer.type, dataset} : undefined;
  }
  return undefined;
}

export type GeometryVariantResolution =
  | {status: 'primary'}
  | {status: 'unavailable'; reason: string}
  | {status: 'variant'; variant: GeometryColumnVariant; request: GeometryVariantRequest};

/** Resolves (preparing on first use) the variant for a layer; hidden layers hold none. */
export function resolveGeometryColumnVariant(
  variants: GeometryColumnVariantRuntime,
  layer: GeometryVariantLayer,
  primary: GeometryVariantPrimary,
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>
): GeometryVariantResolution {
  const request = getGeometryVariantRequest(layer, primary, dataset);
  if (!request || layer.config.isVisible === false) {
    variants.releaseLayer(layer.id);
    return {status: 'primary'};
  }
  const result = variants.resolve(layer.id, request);
  return result.status === 'ready' ? {status: 'variant', variant: result.variant, request} : result;
}

/** Style-host binding and options overrides for a variant layer. */
export function makeGeometryVariantOverrides(
  variant: GeometryColumnVariant,
  layerId: string,
  scopes: {
    registeredLayerIds: readonly string[];
    currentRevision: number;
    getLayerTopologyMask(layerId: string): GPUVector<'uint32'>;
  }
): {
  binding: {bounds?: [number, number, number, number]; cellColumn?: string};
  options: {
    geometry?: PreparedGeometryGPUVectors;
    geometryColumn?: string;
    h3?: H3GPUVectors;
    h3Column?: string;
    h3MetadataMask?: GPUVector<'uint32'>;
    h3MetadataRevision?: number;
  };
} {
  return {
    binding: {
      ...(variant.bounds ? {bounds: variant.bounds} : {}),
      ...(variant.kind === 'cell' ? {cellColumn: variant.column} : {})
    },
    options: {
      ...(variant.geometry
        ? {geometry: variant.geometry, geometryColumn: variant.geometryColumn}
        : {}),
      ...(variant.h3
        ? {
            h3: variant.h3,
            h3Column: variant.column,
            h3MetadataMask: scopes.registeredLayerIds.includes(layerId)
              ? scopes.getLayerTopologyMask(layerId)
              : undefined,
            h3MetadataRevision: scopes.currentRevision + 1
          }
        : {})
    }
  };
}

/** Selector registering a variant layer so native filters and topology masks reach it. */
export function makeGeometryVariantSelector(
  layer: GeometryVariantLayer,
  variant: GeometryColumnVariant
): KeplerSelectionLayer | undefined {
  const base = {
    id: layer.id,
    visible: layer.config.isVisible,
    // Offset keeps variant revisions distinct from dataset revisions.
    geometryRevision: 1_000_000_000 + variant.generation
  };
  if (variant.h3)
    return {
      ...base,
      geometry: {kind: 'point', positions: variant.h3.centers, validity: variant.h3.validity}
    } as KeplerSelectionLayer;
  if (variant.geometry?.representativeUnsupportedReason)
    return {...base, geometry: {kind: 'no-representative'}} as KeplerSelectionLayer;
  const representative = variant.geometry?.representative;
  return representative ? ({...base, geometry: representative} as KeplerSelectionLayer) : undefined;
}

let sharedSession: KeplerComputeSession | undefined;
let sharedRuntime: GeometryColumnVariantRuntime | undefined;

/** One runtime per compute session, created lazily on the first non-primary geometry layer. */
export function getGeometryColumnVariantRuntime(
  session: KeplerComputeSession,
  requestUpdate: () => void
): GeometryColumnVariantRuntime {
  if (sharedSession !== session || !sharedRuntime) {
    if (sharedRuntime) void sharedRuntime.destroy(Promise.resolve()).catch(() => {});
    sharedSession = session;
    sharedRuntime = new GeometryColumnVariantRuntime(session, requestUpdate);
  }
  return sharedRuntime;
}

/** The live runtime for a session, if any layer ever needed one. */
export function peekGeometryColumnVariantRuntime(
  session: KeplerComputeSession
): GeometryColumnVariantRuntime | undefined {
  return sharedSession === session ? sharedRuntime : undefined;
}

type PublishedGeometryLike = {
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  geometryColumn?: string;
  h3Column?: string;
  cellIndex?: NativeCellIndex;
  trip?: unknown;
  h3?: unknown;
  planarGeometryStatus?: unknown;
};

/** Convenience wrapper for a published dataset: resolves the layer's variant, if it needs one. */
export function resolvePublishedGeometryVariant(
  session: KeplerComputeSession,
  requestUpdate: () => void,
  published: PublishedGeometryLike,
  layer: GeometryVariantLayer
): GeometryVariantResolution {
  const request = getGeometryVariantRequest(layer, getPrimary(published), published.dataset);
  const existing = peekGeometryColumnVariantRuntime(session);
  if (!request && !existing) return {status: 'primary'};
  return resolveGeometryColumnVariant(
    getGeometryColumnVariantRuntime(session, requestUpdate),
    layer,
    getPrimary(published),
    published.dataset
  );
}

function getPrimary(published: PublishedGeometryLike): GeometryVariantPrimary {
  return {
    geometryColumn: published.geometryColumn,
    h3Column: published.h3Column,
    cellIndex: published.cellIndex,
    isTrip: Boolean(published.trip),
    hasGeometry: Boolean(published.planarGeometryStatus),
    hasH3: Boolean(published.h3)
  };
}
