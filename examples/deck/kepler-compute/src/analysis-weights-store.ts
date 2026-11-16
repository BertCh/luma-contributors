// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {trackDeferredOperation} from './tracked-operations';
import {GPUCommandGraph, GPUHistogram, createTransientView} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  GPUSpatialWeightsSummary,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  parseKeplerPointWeightsSettings,
  parseKeplerPolygonWeightsSettings,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetReference,
  type KeplerDatasetSnapshot,
  type KeplerJSONValue,
  type KeplerPointWeightsSettings,
  type KeplerPolygonWeightsSettings
} from '@luma-contributors/kepler';
import {AnalysisPointWeightsCache} from './analysis-point-weights-cache';
import {
  AnalysisPolygonWeightsCache,
  type PolygonAnalysisSource
} from './analysis-polygon-weights-cache';

export type AnalysisWeightsSettings = KeplerPointWeightsSettings | KeplerPolygonWeightsSettings;
export type AnalysisWeightsDefinition = {
  id: string;
  name: string;
  version: number;
  source: KeplerDatasetReference;
  settings: AnalysisWeightsSettings;
};
export type AnalysisWeightsConsumer = {
  id: string;
  weightsId: string;
  version: number;
  staleReason?: string;
};
export type AnalysisWeightsInspection = {
  weightsId: string;
  version: number;
  islands: number;
  minimumNeighbors: number;
  maximumNeighbors: number;
  /** Equal-width bins over [0, maximumNeighbors + 1). */
  histogram: {minimum: number; maximumExclusive: number; count: number}[];
};

type SourceEntry = {
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  references: number;
  point?: AnalysisPointWeightsCache;
  polygon?: AnalysisPolygonWeightsCache;
  polygonSource?: PolygonAnalysisSource;
};
const stores = new WeakMap<KeplerComputeSession, AnalysisWeightsStore>();
const OPERATION = 'AnalysisWeightsStore';
/** Keep the on-demand inspector readable and bound its compact readback. */
const MAXIMUM_INSPECTION_HISTOGRAM_BINS = 32;
/** Conservative allowance for summary sort/scan scratch beyond its source-sized workspaces. */
const INSPECTION_BASE_SCRATCH_BYTES = 4 * 1024 * 1024;
const INSPECTION_SCRATCH_BYTES_PER_ELEMENT = 128;
const SUMMARY_COUNTS = GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts;

export function getAnalysisWeightsStore(session: KeplerComputeSession): AnalysisWeightsStore {
  let store = stores.get(session);
  if (!store) {
    store = new AnalysisWeightsStore(session);
    stores.set(session, store);
  }
  return store;
}

/** Definitions persist independently of GPU leases. Source caches are shared by runtime owners. */
export class AnalysisWeightsStore {
  private definitions = new Map<string, AnalysisWeightsDefinition>();
  private consumers = new Map<string, AnalysisWeightsConsumer>();
  private listeners = new Set<() => void>();
  private sources = new Map<KeplerDatasetSnapshot<KeplerArrowDataset>, SourceEntry>();
  private sequence = 0;

  constructor(private readonly session: KeplerComputeSession) {}

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  list(source?: KeplerDatasetReference): AnalysisWeightsDefinition[] {
    return [...this.definitions.values()]
      .filter(definition => !source || matchesSource(definition.source, source))
      .map(cloneDefinition);
  }

  get(id: string): AnalysisWeightsDefinition {
    const definition = this.definitions.get(id);
    validateRequirement(OPERATION, Boolean(definition), 'id must name existing weights');
    return cloneDefinition(definition!);
  }

  create(source: KeplerDatasetReference, settings: AnalysisWeightsSettings, name: string) {
    let id: string;
    do id = `weights-${++this.sequence}`;
    while (this.definitions.has(id));
    const definition = parseDefinition({id, name, version: 1, source, settings});
    this.assertSourceCurrent(definition);
    this.definitions.set(id, definition);
    this.notify();
    return cloneDefinition(definition);
  }

  findOrCreate(source: KeplerDatasetReference, settings: AnalysisWeightsSettings) {
    const normalized = parseSettings(settings);
    const existing = this.list(source).find(
      definition => JSON.stringify(definition.settings) === JSON.stringify(normalized)
    );
    return existing ?? this.create(source, normalized, `${normalized.kind} neighbors`);
  }

  rename(id: string, name: string): void {
    const definition = parseDefinition({...this.get(id), name});
    this.definitions.set(id, definition);
    this.notify();
  }

  edit(id: string, settings: AnalysisWeightsSettings): void {
    const previous = this.get(id);
    this.assertSourceCurrent(previous);
    const normalized = parseSettings(settings);
    validateRequirement(
      OPERATION,
      (normalized.kind === 'contiguity') === (previous.settings.kind === 'contiguity'),
      'settings.kind must preserve the point or polygon weights family'
    );
    const definition = parseDefinition({
      ...previous,
      settings: normalized,
      version: previous.version + 1
    });
    this.definitions.set(id, definition);
    this.invalidate(id, `Weights “${definition.name}” changed to version ${definition.version}`);
    this.notify();
  }

  bind(consumerId: string, weightsId: string, version?: number): void {
    const definition = this.get(weightsId);
    this.assertSourceCurrent(definition);
    const acceptedVersion = version ?? definition.version;
    validateRequirement(
      OPERATION,
      Number.isSafeInteger(acceptedVersion) &&
        acceptedVersion >= 1 &&
        acceptedVersion <= definition.version,
      'consumer.version must name an existing weights version'
    );
    this.consumers.set(consumerId, {
      id: consumerId,
      weightsId,
      version: acceptedVersion,
      ...(acceptedVersion === definition.version
        ? {}
        : {staleReason: `Weights “${definition.name}” changed to version ${definition.version}`})
    });
    this.notify();
  }

  unbind(consumerId: string): void {
    if (this.consumers.delete(consumerId)) this.notify();
  }

  getConsumer(consumerId: string): AnalysisWeightsConsumer | undefined {
    this.refreshSources();
    const consumer = this.consumers.get(consumerId);
    return consumer && {...consumer};
  }

  getConsumers(weightsId: string): AnalysisWeightsConsumer[] {
    this.refreshSources();
    return [...this.consumers.values()]
      .filter(consumer => consumer.weightsId === weightsId)
      .map(consumer => ({...consumer}));
  }

  /** Called before using definitions as well as on source-owner changes; no source rows are read. */
  refreshSources(): void {
    let changed = false;
    for (const definition of this.definitions.values()) {
      if (
        this.session.registry.getDataset(definition.source.datasetId)?.revision !==
        definition.source.revision
      )
        changed =
          this.invalidate(
            definition.id,
            `Source ${definition.source.datasetId} revision changed or was removed`
          ) || changed;
    }
    if (changed) this.notify();
  }

  retainSource(
    dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
    polygonSource?: PolygonAnalysisSource
  ): () => void {
    let entry = this.sources.get(dataset);
    if (!entry) {
      entry = {dataset, references: 0, polygonSource};
      this.sources.set(dataset, entry);
    }
    entry.polygonSource ??= polygonSource;
    entry.references++;
    this.refreshSources();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      if (--entry!.references === 0) {
        entry!.point?.destroy();
        entry!.polygon?.destroy();
        this.sources.delete(dataset);
      }
    };
  }

  pointCache(dataset: KeplerDatasetSnapshot<KeplerArrowDataset>): AnalysisPointWeightsCache {
    const entry = this.requireSource(dataset);
    entry.point ??= new AnalysisPointWeightsCache(this.session, dataset);
    return entry.point;
  }

  polygonCache(dataset: KeplerDatasetSnapshot<KeplerArrowDataset>): AnalysisPolygonWeightsCache {
    const entry = this.requireSource(dataset);
    validateRequirement(
      OPERATION,
      Boolean(entry.polygonSource),
      'polygonSource must retain original polygon geometry'
    );
    entry.polygon ??= new AnalysisPolygonWeightsCache(this.session, dataset, entry.polygonSource!);
    return entry.polygon;
  }

  serialize(): KeplerJSONValue {
    return JSON.parse(JSON.stringify({schemaVersion: 1, definitions: this.list()}));
  }

  /** Replay validates every definition first. GPU topology is rebuilt lazily, never serialized. */
  restore(value: unknown, rebind = false): void {
    const definitions = parseAnalysisWeightsDefinitions(value).map(definition => {
      const dataset = this.session.registry.getDataset(definition.source.datasetId);
      validateRequirement(
        OPERATION,
        Boolean(dataset),
        'source.datasetId must name a restored dataset'
      );
      if (rebind)
        return {...definition, source: {datasetId: dataset!.id, revision: dataset!.revision}};
      this.assertSourceCurrent(definition);
      return definition;
    });
    for (const id of this.definitions.keys()) this.invalidate(id, 'Weights definitions replayed');
    this.definitions = new Map(definitions.map(definition => [definition.id, definition]));
    this.notify();
  }

  /** Explicit on-demand inspection; reads only six summary words and histogram bins from the GPU. */
  async inspect(id: string): Promise<AnalysisWeightsInspection> {
    const definition = this.get(id);
    this.assertSourceCurrent(definition);
    const entry = [...this.sources.values()].find(source =>
      matchesSource(definition.source, {
        datasetId: source.dataset.id,
        revision: source.dataset.revision
      })
    );
    validateRequirement(OPERATION, Boolean(entry), 'source must have a retained analysis owner');
    const dataset = entry!.dataset;
    const request = this.session.registry.beginRequest(`named-weights-inspector:${id}`, [
      dataset.id
    ]);
    const tracked = trackDeferredOperation(`Inspect weights “${definition.name}”`, request.cancel);
    const ensureCurrent = () => {
      validateRequirement(
        OPERATION,
        !request.signal.aborted,
        'inspection must not be cancelled or superseded'
      );
      this.assertSourceCurrent(definition);
      validateRequirement(
        OPERATION,
        this.get(id).version === definition.version,
        'version must match the inspected version'
      );
    };
    const releaseSource = this.retainSource(dataset);
    const buffers: Buffer[] = [];
    let lease:
      | ReturnType<AnalysisPointWeightsCache['acquire']>
      | ReturnType<AnalysisPolygonWeightsCache['acquire']>
      | undefined;
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    let histogramCompiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    let reservation: ReturnType<KeplerComputeSession['registry']['reserveMemory']> | undefined;
    try {
      const graph = new GPUCommandGraph(this.session.device, {id: 'named-weights-inspector'});
      const settings = definition.settings;
      lease =
        settings.kind === 'contiguity'
          ? this.polygonCache(dataset).acquire(graph, settings)
          : this.pointCache(dataset).acquire(graph, settings);
      const weights = lease.binding.weights;
      const capacity =
        'chunks' in weights
          ? weights.chunks.reduce((total, chunk) => total + chunk.weights.neighbors.length, 0)
          : weights.neighbors.length;
      // Conservative sort/scan allowance for the existing summary contributor, plus outputs.
      reservation = this.session.registry.reserveMemory(
        (capacity + dataset.rowCount) * INSPECTION_SCRATCH_BYTES_PER_ELEMENT +
          INSPECTION_BASE_SCRATCH_BYTES,
        'gpu',
        'scratch'
      );
      const allocate = <Format extends 'uint32' | 'float32'>(
        name: string,
        length: number,
        format: Format
      ) => {
        const buffer = this.session.device.createBuffer({
          id: name,
          byteLength: Math.max(1, length) * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return graph.createDataView(
          graph.importBuffer(
            {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          ),
          {format, length}
        );
      };
      const counts = allocate('weights-inspector-counts', SUMMARY_COUNTS.length, 'uint32');
      const cardinality = allocate('weights-inspector-cardinality', dataset.rowCount, 'uint32');
      const encoder = this.session.device.createCommandEncoder();
      lease.encode(encoder);
      lease.markConsumerEncoded(encoder);
      this.session.device.submit(encoder.finish());
      (settings.kind === 'contiguity'
        ? this.polygonCache(dataset)
        : this.pointCache(dataset)
      ).markSubmitted(encoder);
      await waitForWeightsWork(this.session);
      ensureCurrent();
      const diagnostics = await Promise.all(
        lease.diagnostics.map(buffer => buffer.readAsync(0, 4))
      );
      const diagnosticWords = diagnostics.map(
        bytes => new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0]
      );
      validateRequirement(
        OPERATION,
        !diagnosticWords[0] && (settings.kind !== 'contiguity' || !diagnosticWords[2]),
        'weights.status must have no topology or candidate overflow before inspection'
      );
      ensureCurrent();
      graph.add(
        new GPUSpatialWeightsSummary({
          weights,
          output: {
            statistics: createTransientView(
              graph,
              'weights-inspector-statistics',
              'float32',
              GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length
            ),
            counts,
            cardinality
          }
        })
      );
      compiled = graph.compile();
      validateRequirement(
        OPERATION,
        compiled.stats.physicalTransientResourceBytes +
          dataset.rowCount * 4 +
          SUMMARY_COUNTS.length * 4 <=
          reservation.byteLength,
        'inspection memory must fit the admitted scratch reservation'
      );
      const summaryEncoder = this.session.device.createCommandEncoder();
      compiled.encode(summaryEncoder, {parameters: undefined});
      this.session.device.submit(summaryEncoder.finish());
      await waitForWeightsWork(this.session);
      ensureCurrent();
      const countBytes = await buffers[0].readAsync(0, SUMMARY_COUNTS.length * 4);
      const summary = new Uint32Array(
        countBytes.buffer,
        countBytes.byteOffset,
        SUMMARY_COUNTS.length
      );
      const maximumExclusive = summary[SUMMARY_COUNTS.maximumCardinality] + 1;
      const binCount = Math.min(MAXIMUM_INSPECTION_HISTOGRAM_BINS, maximumExclusive);
      ensureCurrent();
      const histogramGraph = new GPUCommandGraph(this.session.device, {
        id: 'named-weights-histogram'
      });
      const cardinalityBuffer = buffers[1];
      const input = histogramGraph.createDataView(
        histogramGraph.importBuffer(
          {
            id: 'cardinality',
            byteLength: cardinalityBuffer.byteLength,
            usage: cardinalityBuffer.usage
          },
          cardinalityBuffer
        ),
        {format: 'uint32', length: dataset.rowCount}
      );
      const histogramBuffer = this.session.device.createBuffer({
        id: 'weights-histogram',
        byteLength: binCount * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(histogramBuffer);
      const histogram = histogramGraph.createDataView(
        histogramGraph.importBuffer(
          {id: 'histogram', byteLength: histogramBuffer.byteLength, usage: histogramBuffer.usage},
          histogramBuffer
        ),
        {format: 'uint32', length: binCount}
      );
      histogramGraph.add(
        new GPUHistogram({input, output: histogram, domain: [0, maximumExclusive]})
      );
      histogramCompiled = histogramGraph.compile();
      validateRequirement(
        OPERATION,
        compiled.stats.physicalTransientResourceBytes +
          histogramCompiled.stats.physicalTransientResourceBytes +
          buffers.reduce((total, buffer) => total + buffer.byteLength, 0) <=
          reservation.byteLength,
        'inspection memory must fit the admitted scratch reservation'
      );
      const histogramEncoder = this.session.device.createCommandEncoder();
      histogramCompiled.encode(histogramEncoder, {parameters: undefined});
      this.session.device.submit(histogramEncoder.finish());
      await waitForWeightsWork(this.session);
      const histogramBytes = await histogramBuffer.readAsync(0, binCount * 4);
      ensureCurrent();
      const values = new Uint32Array(histogramBytes.buffer, histogramBytes.byteOffset, binCount);
      return {
        weightsId: id,
        version: definition.version,
        islands: summary[SUMMARY_COUNTS.isolates],
        minimumNeighbors: summary[SUMMARY_COUNTS.minimumCardinality],
        maximumNeighbors: summary[SUMMARY_COUNTS.maximumCardinality],
        histogram: Array.from(values, (count, index) => ({
          minimum: (index * maximumExclusive) / binCount,
          maximumExclusive: ((index + 1) * maximumExclusive) / binCount,
          count
        }))
      };
    } finally {
      try {
        await waitForWeightsWork(this.session);
      } finally {
        histogramCompiled?.destroy();
        compiled?.destroy();
        for (const buffer of buffers) buffer.destroy();
        reservation?.release();
        lease?.release();
        releaseSource();
        tracked.finish();
        await this.session.registry.finishRequest(request);
      }
    }
  }

  private requireSource(dataset: KeplerDatasetSnapshot<KeplerArrowDataset>): SourceEntry {
    const entry = this.sources.get(dataset);
    validateRequirement(
      OPERATION,
      Boolean(entry),
      'source must be retained before acquiring topology'
    );
    return entry!;
  }

  private assertSourceCurrent(definition: AnalysisWeightsDefinition): void {
    this.refreshSources();
    validateRequirement(
      OPERATION,
      this.session.registry.getDataset(definition.source.datasetId)?.revision ===
        definition.source.revision,
      'source.revision must match the current dataset revision'
    );
  }

  private invalidate(weightsId: string, reason: string): boolean {
    let changed = false;
    for (const consumer of this.consumers.values())
      if (consumer.weightsId === weightsId && consumer.staleReason !== reason) {
        consumer.staleReason = reason;
        changed = true;
      }
    return changed;
  }

  private notify(): void {
    for (const listener of this.listeners) listener();
  }
}

export function parseAnalysisWeightsDefinitions(value: unknown): AnalysisWeightsDefinition[] {
  const record = value as {schemaVersion?: unknown; definitions?: unknown};
  validateRequirement(
    OPERATION,
    record?.schemaVersion === 1 && Array.isArray(record.definitions),
    'schemaVersion must contain version 1 definitions'
  );
  const definitions = (record.definitions as unknown[]).map(parseDefinition);
  validateRequirement(
    OPERATION,
    new Set(definitions.map(definition => definition.id)).size === definitions.length,
    'id must be unique'
  );
  return definitions;
}

function parseDefinition(value: unknown): AnalysisWeightsDefinition {
  const definition = value as AnalysisWeightsDefinition;
  validateRequirement(
    OPERATION,
    Boolean(definition) &&
      typeof definition.id === 'string' &&
      Boolean(definition.id.trim()) &&
      typeof definition.name === 'string' &&
      Boolean(definition.name.trim()),
    'id/name must be nonempty strings'
  );
  validateRequirement(
    OPERATION,
    Number.isSafeInteger(definition.version) && definition.version >= 1,
    'version must be a positive safe integer'
  );
  validateRequirement(
    OPERATION,
    typeof definition.source?.datasetId === 'string' &&
      Boolean(definition.source.datasetId) &&
      Number.isSafeInteger(definition.source.revision) &&
      definition.source.revision >= 1,
    'source must name a dataset and positive revision'
  );
  return {
    id: definition.id,
    name: definition.name.trim(),
    version: definition.version,
    source: {datasetId: definition.source.datasetId, revision: definition.source.revision},
    settings: parseSettings(definition.settings)
  };
}

function parseSettings(settings: AnalysisWeightsSettings): AnalysisWeightsSettings {
  return settings?.kind === 'contiguity'
    ? parseKeplerPolygonWeightsSettings(settings)
    : parseKeplerPointWeightsSettings(settings);
}
function cloneDefinition(definition: AnalysisWeightsDefinition): AnalysisWeightsDefinition {
  return {...definition, source: {...definition.source}, settings: {...definition.settings}};
}
function matchesSource(first: KeplerDatasetReference, second: KeplerDatasetReference): boolean {
  return first.datasetId === second.datasetId && first.revision === second.revision;
}

export const SANDBOX_EXTENSION_NAMED_WEIGHTS = 'namedSpatialWeights';

export function parseAnalysisWeightsWorkspace(value: unknown): {
  definitions: AnalysisWeightsDefinition[];
  bindings: {datasetId: string; weightsId: string; version: number}[];
} {
  if (value === undefined) return {definitions: [], bindings: []};
  const definitions = parseAnalysisWeightsDefinitions(value);
  const bindings = (value as {bindings?: unknown}).bindings;
  validateRequirement(
    OPERATION,
    Array.isArray(bindings),
    'bindings must list saved analysis references'
  );
  const parsedBindings = (
    bindings as {datasetId: string; weightsId: string; version: number}[]
  ).map(binding => {
    const definition = definitions.find(candidate => candidate.id === binding?.weightsId);
    validateRequirement(
      OPERATION,
      typeof binding?.datasetId === 'string' &&
        definition?.source.datasetId === binding.datasetId &&
        Number.isSafeInteger(binding.version) &&
        binding.version >= 1 &&
        binding.version <= definition.version,
      'bindings must refer to a saved weights version on the same dataset'
    );
    return {datasetId: binding.datasetId, weightsId: binding.weightsId, version: binding.version};
  });
  validateRequirement(
    OPERATION,
    new Set(parsedBindings.map(binding => binding.datasetId)).size === parsedBindings.length,
    'bindings.datasetId must have at most one active analysis per dataset'
  );
  return {definitions, bindings: parsedBindings};
}

async function waitForWeightsWork(session: KeplerComputeSession): Promise<void> {
  const fence = session.device.createFence();
  try {
    await fence.signaled;
  } finally {
    fence.destroy();
  }
}
