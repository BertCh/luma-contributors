// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUMask, createTransientVectorView, type CompiledGPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {
  GPUKeplerSpatialSelection,
  prepareKeplerSpatialSelection,
  type KeplerComputeSession,
  type KeplerSessionExplorer,
  type KeplerMemoryReservation,
  type KeplerSpatialSelection,
  type KeplerSelectionPolygon,
  type KeplerSelectionPosition,
  type KeplerPolygonMaskFilter,
  type GPUResidentDimension,
  type GPUResidentSummary
} from '@luma-contributors/kepler';
import {describePolygonFilterProblem} from './binding2-polygon-reasons';
import type {KeplerComputeFilter, KeplerComputeTableBinding} from './kepler-table-bridge';

/** Exact/native ranges may own separate masks for display and dynamic-domain scope. */
export type ResidentExternalRangeBinding = {
  displayDimension: Extract<GPUResidentDimension, {kind: 'external'}>;
  domainDimension: Extract<GPUResidentDimension, {kind: 'external'}>;
  setDisplayRange(minimum: number, maximum: number): void;
  setDomainRange(minimum: number, maximum: number): void;
  setDisplayEmpty(): void;
  setDomainEmpty(): void;
};

export type ResidentTableBindingProps = {
  device: Device;
  session: KeplerComputeSession;
  datasetId: string;
  positions: GPUVector<'float32x2'>;
  /** This provider implements polygon masks for one explicitly named point layer. */
  layerId: string;
  fields: KeplerComputeTableBinding['fields'];
  externalRanges?: Readonly<Record<string, ResidentExternalRangeBinding>>;
  onUpdate(): void;
  onUnsupported(reason: string): void;
  onPlotUnsupported?(filterId: string, reason: string | null): void;
  maximumSpatialFilters?: number;
  domainWorkingMemoryBudgetBytes?: number;
  /** Admission reservation per polygon operation, including its candidate/scratch buffers. */
  spatialWorkingMemoryBudgetBytes?: number;
};

export type PreparedResidentPolygon = {
  id: string;
  layerIds: readonly string[];
  selection: KeplerSpatialSelection;
  topologyKey: string;
  inverted: boolean;
};

type SpatialEntry = {
  selection: GPUKeplerSpatialSelection;
  topologyKey: string;
  inverted: boolean;
  reservation: KeplerMemoryReservation;
  generation: number;
};

/** Native FeatureValue -> source-dataset layer targets, without assuming one point layer. */
export function prepareNativePolygonFilter(
  filter: KeplerComputeFilter,
  datasetId: string,
  layers: readonly unknown[],
  registeredLayerIds: readonly string[]
): KeplerPolygonMaskFilter | undefined {
  if (filter.enabled === false || filter.type !== 'polygon') return undefined;
  const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
  if (!datasetIds.includes(datasetId)) return undefined;
  const datasetLayers = layers.filter(
    layer => isRecord(layer) && isRecord(layer['config']) && layer['config']['dataId'] === datasetId
  );
  const layerIds = [...new Set(filter.layerId ?? [])].filter(id =>
    datasetLayers.some(layer => isRecord(layer) && layer['id'] === id)
  );
  if (!layerIds.length) return undefined;
  const unregistered = layerIds.filter(id => !registeredLayerIds.includes(id));
  if (unregistered.length)
    throw new Error(
      `Polygon target requires a registered resident geometry selector; layers without one: ${unregistered.join(', ')}`
    );
  const feature = filter.value;
  if (!isRecord(feature) || !isRecord(feature['geometry']))
    throw new Error('Polygon filter requires a GeoJSON FeatureValue');
  // Specific invalid-shape reasons (self-intersection, collapsed rings...) reach the notice.
  const problem = describePolygonFilterProblem(filter);
  if (problem) throw new Error(problem);
  const geometry = feature['geometry'];
  let polygons: KeplerSelectionPolygon[];
  if (geometry['type'] === 'Polygon') polygons = [readPolygon(geometry['coordinates'])];
  else if (geometry['type'] === 'MultiPolygon' && Array.isArray(geometry['coordinates']))
    polygons = geometry['coordinates'].map(readPolygon);
  else throw new Error('Only Polygon and MultiPolygon filter geometries are supported');
  const selection: KeplerSpatialSelection = {kind: 'polygon', polygons};
  prepareKeplerSpatialSelection(selection, 'native-linear-degrees');
  const properties = feature['properties'];
  const bbox =
    isRecord(properties) && properties['shape'] === 'Rectangle' ? properties['bbox'] : undefined;
  let rectangleBounds: readonly [number, number, number, number] | undefined;
  if (bbox !== undefined) {
    if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(Number.isFinite))
      throw new Error('Native rectangle metadata requires four finite bounds');
    rectangleBounds = [bbox[0], bbox[1], bbox[2], bbox[3]];
  }
  return {id: filter.id, layerIds, selection, inverted: filter.inverted, rectangleBounds};
}

/** Converts the actual Kepler FeatureValue shape, retaining polygon union and hole structure. */
export function prepareResidentPolygonFilter(
  filter: KeplerComputeFilter,
  datasetId: string,
  layerId: string,
  layers: readonly unknown[]
): PreparedResidentPolygon {
  if (filter.type !== 'polygon') throw new Error('Expected a native Kepler polygon filter');
  const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
  if (!datasetIds.includes(datasetId))
    throw new Error('Polygon filter does not target this dataset');
  const isDatasetLayer = (layer: unknown): layer is Record<string, unknown> =>
    isRecord(layer) && isRecord(layer['config']) && layer['config']['dataId'] === datasetId;
  const datasetLayers = layers.filter(isDatasetLayer);
  const layerIds = [...new Set(filter.layerId ?? [])];
  // Duplicated point layers share this binding's one dataset mask, so a filter must target every
  // point layer of the dataset or none: a partial target would need a mask per layer.
  if (!layerIds.length || !layerIds.includes(layerId)) {
    throw new Error('This polygon binding supports only its declared resident point layer');
  }
  if (
    datasetLayers.some(layer => layer['id'] !== layerId && !layerIds.includes(String(layer['id'])))
  ) {
    throw new Error(
      'Additional layers on this dataset require separate per-layer spatial explorers'
    );
  }
  const targets = layerIds.map(id => datasetLayers.find(layer => layer['id'] === id));
  if (targets.some(target => !target || target['type'] !== 'point')) {
    throw new Error('Polygon target must be the declared point layer for this dataset');
  }
  const mappings = new Set(targets.map(target => getPointColumnMapping(target)));
  if (mappings.size > 1) {
    throw new Error('Point layers with different lng/lat mappings need separate position masks');
  }
  const feature = filter.value;
  if (!isRecord(feature) || !isRecord(feature['geometry']))
    throw new Error('Polygon filter requires a GeoJSON FeatureValue');
  const geometry = feature['geometry'];
  let polygons: KeplerSelectionPolygon[];
  if (geometry['type'] === 'Polygon') polygons = [readPolygon(geometry['coordinates'])];
  else if (geometry['type'] === 'MultiPolygon' && Array.isArray(geometry['coordinates']))
    polygons = geometry['coordinates'].map(readPolygon);
  else throw new Error('Only Polygon and MultiPolygon filter geometries are supported');
  const selection: KeplerSpatialSelection = {kind: 'polygon', polygons};
  // Drawn coordinates are rounded to the same Float32 frame as resident points. Native isInPolygon
  // is linear booleanWithin on raw longitude/latitude, so antimeridian-spanning edges and
  // coordinates outside +-180 are ordinary planar vertices; nothing is split or wrapped.
  const prepared = prepareKeplerSpatialSelection(selection, 'native-linear-degrees');
  return {
    id: filter.id,
    layerIds,
    selection,
    topologyKey: prepared.topologyKey,
    inverted: Boolean(filter.inverted)
  };
}

/** lng/lat column identity of a native point layer; duplicates share a position buffer only if equal. */
export function getPointColumnMapping(layer: unknown): string {
  const config = isRecord(layer) && isRecord(layer['config']) ? layer['config'] : undefined;
  const columns = config && isRecord(config['columns']) ? config['columns'] : {};
  const read = (role: string): string => {
    const column = columns[role];
    return isRecord(column) && typeof column['value'] === 'string' ? column['value'] : '';
  };
  return `${read('lng')}|${read('lat')}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function readPolygon(value: unknown): KeplerSelectionPolygon {
  if (!Array.isArray(value)) throw new Error('Polygon coordinates must contain rings');
  return value.map(ring => {
    if (!Array.isArray(ring)) throw new Error('Polygon ring must contain positions');
    return ring.map((position: unknown): KeplerSelectionPosition => {
      if (
        !Array.isArray(position) ||
        position.length < 2 ||
        typeof position[0] !== 'number' ||
        typeof position[1] !== 'number'
      ) {
        throw new Error('Polygon positions require numeric longitude and latitude');
      }
      return [position[0], position[1]];
    });
  });
}

/**
 * Concrete native-table providers. A separate explorer receives only dynamic-domain filters;
 * the primary display explorer also receives fixed-domain and point-layer polygon filters.
 *
 * Construct before the primary explorer, include externalDimension in its dimensions, then bind it.
 * The primary explorer invokes the external encode callback before reading the stable spatial mask.
 * Read summaries after submission, and destroy with the same completion fence as the primary explorer.
 */
export class ResidentTableBinding {
  readonly externalDimension: GPUResidentDimension;
  readonly domainExplorer: KeplerSessionExplorer;
  private readonly props: ResidentTableBindingProps;
  private readonly spatialMask: GPUVector<'uint32'>;
  private readonly spatialMaskReservation: KeplerMemoryReservation;
  private readonly spatial = new Map<string, SpatialEntry>();
  private activeSpatialIds: string[] = [];
  private preparedSpatialCount = 0;
  private generation = 0;
  private primary?: KeplerSessionExplorer;
  private compiled?: CompiledGPUCommandGraph;
  private compositionKey = '';
  private readonly retired: (() => void)[] = [];
  private revision = 0;
  private encodedRevision = -1;
  private domainSummary?: {revision: number; domains: GPUResidentSummary['domains']};
  private pendingRead?: Promise<boolean>;
  private destroyed = false;
  private completion?: Promise<void>;

  constructor(props: ResidentTableBindingProps) {
    this.props = props;
    const maximumSpatialFilters = props.maximumSpatialFilters ?? 8;
    if (!Number.isSafeInteger(maximumSpatialFilters) || maximumSpatialFilters < 1)
      throw new Error('Spatial filter capacity must be positive');
    this.spatialMaskReservation = props.session.registry.reserveMemory(
      props.positions.data.reduce((bytes, chunk) => bytes + Math.max(4, chunk.length * 4), 0),
      'gpu',
      'scratch'
    );
    const chunks: GPUData<'uint32'>[] = [];
    try {
      for (const [index, chunk] of props.positions.data.entries()) {
        chunks.push(
          new GPUData({
            buffer: props.device.createBuffer({
              id: `kepler-spatial-mask-${index}`,
              byteLength: Math.max(4, chunk.length * 4),
              usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
            }),
            format: 'uint32',
            length: chunk.length,
            ownsBuffer: true
          })
        );
      }
      this.spatialMask = new GPUVector({
        type: 'data',
        name: 'kepler-layer-spatial-mask',
        format: 'uint32',
        data: chunks,
        ownsData: true
      });
      const dimensions: GPUResidentDimension[] = Object.entries(props.fields).map(
        ([column, field]) =>
          props.externalRanges?.[field.dimension]?.domainDimension ??
          (field.kind === 'category'
            ? {
                id: field.dimension,
                kind: 'category',
                column,
                capacity: (field.categories ?? field.domain).length
              }
            : {id: field.dimension, kind: 'range', column})
      );
      this.domainExplorer = props.session.openExplorer(props.datasetId, {
        workingMemoryBudgetBytes:
          props.domainWorkingMemoryBudgetBytes ?? props.positions.length * 128 + 8 * 1024 * 1024,
        dimensions,
        domains: Object.entries(props.fields)
          .filter(
            ([, field]) => field.kind !== 'category' && !props.externalRanges?.[field.dimension]
          )
          .map(([column]) => ({id: column, column}))
      });
      this.externalDimension = {
        id: 'kepler-layer-spatial-selection',
        kind: 'external',
        mask: this.spatialMask,
        encode: encoder => this.encode(encoder)
      };
    } catch (error) {
      for (const chunk of chunks) chunk.destroy();
      this.spatialMaskReservation.release();
      throw error;
    }
  }

  get currentRevision(): number {
    return this.revision;
  }

  bind(primary: KeplerSessionExplorer): KeplerComputeTableBinding {
    if (this.primary && this.primary !== primary)
      throw new Error('A table provider binds one display explorer');
    if (
      primary.dataset.id !== this.props.datasetId ||
      primary.dataset.revision !== this.domainExplorer.dataset.revision
    ) {
      throw new Error('Display and domain explorers require the same dataset revision');
    }
    this.primary = primary;
    return {
      fields: this.props.fields,
      explorer: {
        clear: id => primary.clear(id),
        setRange: (id, range) => this.setRange(primary, id, range, 'display'),
        setCategories: (id, values) => primary.setCategories(id, values),
        setEmpty: id => this.setEmpty(primary, id, 'display')
      },
      onSelectionChange: (_plan, revision) => {
        this.revision = revision;
        this.preparedSpatialCount = 0;
        primary.setExternalActive(this.externalDimension.id, this.activeSpatialIds.length > 0);
        this.retireInactiveSpatial();
        this.props.onUpdate();
      },
      onPlotRecovery: () => this.props.onUpdate(),
      onPlotUnsupported: (filterId, reason) => this.props.onPlotUnsupported?.(filterId, reason),
      onUnsupported: reason => {
        this.preparedSpatialCount = 0;
        this.props.onUnsupported(reason);
      },
      applyDomainSelection: (commands, revision) => {
        this.revision = revision;
        this.domainSummary = undefined;
        this.domainExplorer.clearAll();
        for (const command of commands) {
          if (command.kind === 'range')
            this.setRange(this.domainExplorer, command.dimension, command.value, 'domain');
          else if (command.kind === 'category')
            this.domainExplorer.setCategories(command.dimension, command.value);
          else this.setEmpty(this.domainExplorer, command.dimension, 'domain');
        }
      },
      getDynamicLayerDomain: (field, scale, revision) => {
        if (this.props.externalRanges?.[this.props.fields[field]?.dimension]) {
          this.props.onUnsupported(
            'Exact timestamp layer domains require a matching GPU time-domain reduction'
          );
          return null;
        }
        if (!['linear', 'quantize', 'sqrt', 'custom'].includes(scale)) {
          this.props.onUnsupported(
            `Dynamic ${scale} domains require a matching GPU classification provider`
          );
          return null;
        }
        return this.domainSummary?.revision === revision
          ? (this.domainSummary.domains[field] ?? null)
          : null;
      },
      clearSpatialFilters: () => {
        this.activeSpatialIds = [];
      },
      prepareSpatialFilter: (filter, layers) => {
        const prepared = prepareResidentPolygonFilter(
          filter,
          this.props.datasetId,
          this.props.layerId,
          layers
        );
        if (++this.preparedSpatialCount > (this.props.maximumSpatialFilters ?? 8))
          throw new KeplerAllocationError('capacity', {
            message: 'Resident spatial filter capacity exceeded'
          });
        return {layerIds: prepared.layerIds, apply: () => this.applySpatial(prepared)};
      }
    };
  }

  /** Called before primary crossfilter consumes externalDimension.mask, on the same encoder. */
  encode(encoder: CommandEncoder): void {
    this.assertActive();
    this.domainExplorer.encode(encoder);
    if (this.activeSpatialIds.length) {
      this.prepareComposition();
      for (const id of this.activeSpatialIds) this.spatial.get(id)!.selection.encode(encoder);
      this.compiled!.encode(encoder, {parameters: undefined});
    }
    this.encodedRevision = this.revision;
  }

  /** Reads only exact extrema and completeness diagnostics, never source rows or masks. */
  readSummary(): Promise<boolean> {
    if (this.destroyed) return Promise.resolve(false);
    if (this.pendingRead) return this.pendingRead;
    this.pendingRead = this.readCurrentSummary().finally(() => {
      this.pendingRead = undefined;
    });
    return this.pendingRead;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.completion) return this.completion;
    this.destroyed = true;
    this.completion = (async () => {
      const outcomes = await Promise.allSettled([
        Promise.resolve(completion),
        this.pendingRead,
        this.domainExplorer.close(completion)
      ]);
      this.compiled?.destroy();
      this.compiled = undefined;
      for (const entry of this.spatial.values()) destroySpatialEntry(entry);
      this.spatial.clear();
      for (const dispose of this.retired.splice(0)) dispose();
      this.spatialMask.destroy();
      this.spatialMaskReservation.release();
      const failure = outcomes.find(outcome => outcome.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
    })();
    return this.completion;
  }

  private setRange(
    explorer: KeplerSessionExplorer,
    id: string,
    range: readonly [number, number],
    scope: 'display' | 'domain'
  ): void {
    const external = this.props.externalRanges?.[id];
    if (!external) {
      explorer.setRange(id, range[0], range[1]);
      return;
    }
    if (scope === 'display') external.setDisplayRange(range[0], range[1]);
    else external.setDomainRange(range[0], range[1]);
    explorer.setExternalActive(id, true);
  }

  private setEmpty(explorer: KeplerSessionExplorer, id: string, scope: 'display' | 'domain'): void {
    const external = this.props.externalRanges?.[id];
    if (!external) {
      explorer.setEmpty(id);
      return;
    }
    if (scope === 'display') external.setDisplayEmpty();
    else external.setDomainEmpty();
    explorer.setExternalActive(id, true);
  }

  private applySpatial(prepared: PreparedResidentPolygon): void {
    this.assertActive();
    const previous = this.spatial.get(prepared.id);
    if (previous?.topologyKey === prepared.topologyKey) {
      previous.selection.updateSelection(prepared.selection);
      previous.inverted = prepared.inverted;
    } else {
      const reservation = this.props.session.registry.reserveMemory(
        this.props.spatialWorkingMemoryBudgetBytes ??
          this.props.positions.length * 128 + 1024 * 1024,
        'gpu',
        'scratch'
      );
      let selection: GPUKeplerSpatialSelection;
      try {
        selection = new GPUKeplerSpatialSelection({
          device: this.props.device,
          positions: this.props.positions,
          selection: prepared.selection,
          coordinateSpace: 'native-linear-degrees',
          boundaryRule: 'open'
        });
      } catch (error) {
        reservation.release();
        throw error;
      }
      if (previous) this.retired.push(() => destroySpatialEntry(previous));
      this.spatial.set(prepared.id, {
        selection,
        topologyKey: prepared.topologyKey,
        inverted: prepared.inverted,
        reservation,
        generation: ++this.generation
      });
    }
    if (!this.activeSpatialIds.includes(prepared.id)) this.activeSpatialIds.push(prepared.id);
  }

  private prepareComposition(): void {
    const key = this.activeSpatialIds
      .map(id => {
        const entry = this.spatial.get(id)!;
        return `${id}:${entry.generation}:${entry.inverted}`;
      })
      .join('|');
    if (key === this.compositionKey && this.compiled) return;
    const graph = new GPUCommandGraph(this.props.device, {id: 'kepler-polygon-composition'});
    const output = graph.importGPUVector('spatial-mask', this.spatialMask);
    let previous: GraphVectorView<'uint32'> | undefined;
    for (const [index, id] of this.activeSpatialIds.entries()) {
      const entry = this.spatial.get(id)!;
      let input = graph.importGPUVector(`filter-${index}`, entry.selection.mask);
      if (entry.inverted) {
        const inverted = createTransientVectorView(graph, `inverted-${index}`, output);
        graph.add(
          new GPUMask({id: `invert-${index}`, inputs: [input], output: inverted, operation: 'not'})
        );
        input = inverted;
      }
      const target =
        index === this.activeSpatialIds.length - 1
          ? output
          : createTransientVectorView(graph, `intersection-${index}`, output);
      graph.add(
        new GPUMask({
          id: `intersect-${index}`,
          inputs: previous ? [previous, input] : [input],
          output: target
        })
      );
      previous = target;
    }
    const compiled = graph.compile();
    const previousCompiled = this.compiled;
    if (previousCompiled) this.retired.push(() => previousCompiled.destroy());
    this.compiled = compiled;
    this.compositionKey = key;
  }

  private retireInactiveSpatial(): void {
    for (const [id, entry] of this.spatial) {
      if (!this.activeSpatialIds.includes(id)) {
        this.spatial.delete(id);
        this.retired.push(() => destroySpatialEntry(entry));
      }
    }
    // A graph importing a removed adapter must not be reused if the same filter id is added later.
    if (!this.activeSpatialIds.length && this.compiled) {
      const compiled = this.compiled;
      this.retired.push(() => compiled.destroy());
      this.compiled = undefined;
      this.compositionKey = '';
    }
  }

  private async readCurrentSummary(): Promise<boolean> {
    // Capture only resources retired before this submitted frame. Concurrent edits may retire more
    // resources while readback is pending; the next submitted frame will fence those separately.
    const retired = this.retired.splice(0);
    try {
      const revision = this.encodedRevision;
      if (revision < 0 || revision !== this.revision) return false;
      const selections = this.activeSpatialIds.map(id => this.spatial.get(id)!.selection);
      const [summary, statuses] = await Promise.all([
        this.domainExplorer.readSummary(),
        Promise.all(selections.map(selection => selection.readStatus()))
      ]);
      if (this.destroyed || revision !== this.revision || !summary) return false;
      if (statuses.some(status => status.stale || !status.complete)) {
        this.props.onUnsupported(
          'Polygon selection did not complete with reliable spatial diagnostics'
        );
        return false;
      }
      this.domainSummary = {revision, domains: summary.domains};
      return true;
    } finally {
      // Retire even stale/failed summaries so rapid edits do not strand admission reservations.
      // Callers read only after submitting the primary frame that last referenced these resources.
      if (retired.length) {
        let fence: ReturnType<Device['createFence']> | undefined;
        try {
          fence = this.props.device.createFence();
          await fence.signaled;
        } finally {
          fence?.destroy();
          for (const dispose of retired) dispose();
        }
      }
    }
  }

  private assertActive(): void {
    if (this.destroyed) throw new Error('Resident table binding is destroyed');
  }
}

function destroySpatialEntry(entry: SpatialEntry): void {
  entry.selection.destroy();
  entry.reservation.release();
}
