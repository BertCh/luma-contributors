// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {
  POLYGON_CONTIGUITY_MAXIMUM_ROWS,
  POLYGON_CONTIGUITY_PREPARATION_BYTES,
  preparePolygonContiguityTopology,
  type PreparedKeplerPolygonTopologySource
} from './geometry-preparation/index';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  addKeplerPolygonWeights,
  parseKeplerPolygonWeightsSettings,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation,
  type KeplerPolygonWeightsBinding,
  type KeplerPolygonWeightsSettings
} from '@luma-contributors/kepler';

/** Borrowed original-coordinate polygon data, charged to the publication's CPU reservation. */
export type PolygonAnalysisSource = {
  column: string;
  source: PreparedKeplerPolygonTopologySource;
};

type PolygonDiagnostics = readonly [Buffer, Buffer, Buffer, Buffer];

type CacheEntry = {
  key: string;
  references: number;
  ready: boolean;
  pendingEncoder?: CommandEncoder;
  successfulEncoder?: CommandEncoder;
  byteLength: number;
  diagnostics: PolygonDiagnostics;
  importBinding(graph: GPUCommandGraph): KeplerPolygonWeightsBinding;
  compiled: CompiledGPUCommandGraph;
  destroy(): void;
};

type PolygonTopology = {
  vertexCount: number;
  byteLength: number;
  importToGraph(graph: GPUCommandGraph): {
    polygons: {
      kind: 'polygons';
      positions: GraphDataView<'float32x2'>;
      ringOffsets: GraphDataView<'uint32'>;
      polygonOffsets: GraphDataView<'uint32'>;
      featureOffsets: GraphDataView<'uint32'>;
    };
    geometryMask: GraphDataView<'uint32'>;
  };
  destroy(): void;
};

export type AnalysisPolygonWeightsLease = {
  /** Opaque identity of the shared cache entry, identical across consumer imports. */
  topology?: object;
  binding: KeplerPolygonWeightsBinding;
  /** Overflow, required slots, candidate overflow, and excluded geometry rows, in that order. */
  diagnostics: PolygonDiagnostics;
  encode(commandEncoder: CommandEncoder): void;
  markConsumerEncoded(commandEncoder: CommandEncoder): void;
  retain(): () => void;
  release(): void;
};

/** Fixed source topology, reusable across numeric fields, population masks and inference settings. */
export class AnalysisPolygonWeightsCache {
  private readonly entries = new Map<string, CacheEntry>();
  private topology?: PolygonTopology;
  private mostRecentKey?: string;
  private sequence = 0;
  private buildCount = 0;
  private encodeCount = 0;
  private destroyed = false;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
    private readonly source: PolygonAnalysisSource
  ) {
    if (
      dataset.rowCount < 1 ||
      dataset.rowCount > POLYGON_CONTIGUITY_MAXIMUM_ROWS ||
      source.source.metadata.rowCount !== dataset.rowCount ||
      !dataset.value.source.schema.fields.some(field => field.name === source.column)
    )
      throw new Error('Polygon topology must match the retained source geometry and row capacity');
  }

  get stats(): Readonly<{
    buildCount: number;
    encodeCount: number;
    entryCount: number;
    byteLength: number;
  }> {
    return {
      buildCount: this.buildCount,
      encodeCount: this.encodeCount,
      entryCount: this.entries.size,
      byteLength:
        (this.topology?.byteLength ?? 0) +
        [...this.entries.values()].reduce((total, entry) => total + entry.byteLength, 0)
    };
  }

  acquire(
    graph: GPUCommandGraph,
    requestedSettings: KeplerPolygonWeightsSettings
  ): AnalysisPolygonWeightsLease {
    if (this.destroyed) throw new Error('Polygon weight cache is destroyed');
    if (graph.device !== this.session.device)
      throw new Error('Polygon weights require the source session device');
    const settings = parseKeplerPolygonWeightsSettings(requestedSettings);
    if (settings.geometryColumn !== this.source.column)
      throw new Error('Polygon weights require the prepared source geometry field');
    const key = JSON.stringify([this.dataset.id, this.dataset.revision, settings]);
    let entry = this.entries.get(key);
    if (!entry) {
      this.evictUnused();
      if (this.entries.size >= 4)
        throw new Error('Submit pending analysis work before changing polygon topology again');
      const hadTopology = Boolean(this.topology);
      try {
        entry = this.createEntry(key, settings);
        this.entries.set(key, entry);
      } catch (error) {
        if (!hadTopology && this.entries.size === 0) {
          this.topology?.destroy();
          this.topology = undefined;
        }
        throw error;
      }
    }
    this.mostRecentKey = key;
    const cached = entry;
    cached.references++;
    this.evictUnused();
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      this.releaseEntry(cached);
    };
    try {
      return {
        topology: cached,
        binding: cached.importBinding(graph),
        diagnostics: cached.diagnostics,
        encode: commandEncoder => {
          if (released || this.destroyed) throw new Error('Polygon weight lease is unavailable');
          if (cached.ready || cached.pendingEncoder === commandEncoder) return;
          cached.compiled.encode(commandEncoder, {parameters: undefined});
          cached.pendingEncoder = commandEncoder;
          this.encodeCount++;
        },
        markConsumerEncoded: commandEncoder => {
          if (released || this.destroyed) throw new Error('Polygon weight lease is unavailable');
          if (cached.pendingEncoder === commandEncoder) cached.successfulEncoder = commandEncoder;
        },
        retain: () => {
          if (released || this.destroyed) throw new Error('Polygon weight lease is unavailable');
          cached.references++;
          let finished = false;
          return () => {
            if (!finished) {
              finished = true;
              this.releaseEntry(cached);
            }
          };
        },
        release
      };
    } catch (error) {
      release();
      throw error;
    }
  }

  /** A successful consumer encoding is acknowledged only after submission of that same encoder. */
  markSubmitted(commandEncoder: CommandEncoder): void {
    for (const entry of this.entries.values()) {
      if (
        !entry.ready &&
        entry.pendingEncoder === commandEncoder &&
        entry.successfulEncoder === commandEncoder
      ) {
        entry.ready = true;
        entry.pendingEncoder = undefined;
        entry.successfulEncoder = undefined;
        this.buildCount++;
      }
    }
  }

  /** Call after all program fences and asynchronous readers have released their leases. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const entry of this.entries.values()) entry.destroy();
    this.entries.clear();
    this.topology?.destroy();
    this.topology = undefined;
  }

  private releaseEntry(entry: CacheEntry): void {
    entry.references--;
    if (
      entry.references === 0 &&
      (entry.key !== this.mostRecentKey || (!entry.ready && !entry.successfulEncoder))
    )
      this.evict(entry);
  }

  private evictUnused(): void {
    for (const entry of this.entries.values())
      if (entry.references === 0 && entry.key !== this.mostRecentKey) this.evict(entry);
  }

  private evict(entry: CacheEntry): void {
    if (!this.entries.delete(entry.key)) return;
    entry.destroy();
    if (this.mostRecentKey === entry.key) this.mostRecentKey = [...this.entries.keys()].at(-1);
  }

  private prepareTopology(): PolygonTopology {
    if (this.topology) return this.topology;
    const staging = this.session.registry.reserveMemory(
      POLYGON_CONTIGUITY_PREPARATION_BYTES,
      'cpu',
      'scratch'
    );
    const buffers: Buffer[] = [];
    let reservation: KeplerMemoryReservation | undefined;
    try {
      const prepared = preparePolygonContiguityTopology(this.source.source);
      if (prepared.cpuByteLength > staging.byteLength)
        throw new Error('Polygon topology exceeds the admitted CPU preparation budget');
      const byteLength =
        Math.max(8, prepared.positions.byteLength) +
        Math.max(4, prepared.ringOffsets.byteLength) +
        Math.max(4, prepared.polygonOffsets.byteLength) +
        Math.max(4, prepared.featureOffsets.byteLength) +
        Math.max(4, prepared.geometryMask.byteLength);
      reservation = this.session.registry.reserveMemory(byteLength, 'gpu', 'scratch');
      const allocate = (name: string, values: Float32Array | Uint32Array, minimumBytes = 4) => {
        const buffer = this.session.device.createBuffer({
          id: `kepler-polygon-topology-${name}`,
          byteLength: Math.max(minimumBytes, values.byteLength),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        if (values.byteLength) buffer.write(values);
        return buffer;
      };
      const positions = allocate('positions', prepared.positions, 8);
      const ringOffsets = allocate('ring-offsets', prepared.ringOffsets);
      const polygonOffsets = allocate('polygon-offsets', prepared.polygonOffsets);
      const featureOffsets = allocate('feature-offsets', prepared.featureOffsets);
      const geometryMask = allocate('geometry-mask', prepared.geometryMask);
      // Retain scalar lengths only. The preparation arrays and interner are released after upload.
      const vertexCount = prepared.vertexCount;
      const ringOffsetCount = prepared.ringOffsets.length;
      const polygonOffsetCount = prepared.polygonOffsets.length;
      const featureOffsetCount = prepared.featureOffsets.length;
      const rowCount = prepared.rowCount;
      const ownedReservation = reservation;
      this.topology = {
        vertexCount,
        byteLength,
        importToGraph: graph => ({
          polygons: {
            kind: 'polygons',
            positions: importBuffer(
              graph,
              'polygon-token-positions',
              positions,
              'float32x2',
              vertexCount
            ),
            ringOffsets: importBuffer(
              graph,
              'polygon-ring-offsets',
              ringOffsets,
              'uint32',
              ringOffsetCount
            ),
            polygonOffsets: importBuffer(
              graph,
              'polygon-part-offsets',
              polygonOffsets,
              'uint32',
              polygonOffsetCount
            ),
            featureOffsets: importBuffer(
              graph,
              'polygon-feature-offsets',
              featureOffsets,
              'uint32',
              featureOffsetCount
            )
          },
          geometryMask: importBuffer(
            graph,
            'polygon-geometry-mask',
            geometryMask,
            'uint32',
            rowCount
          )
        }),
        destroy: () => {
          for (const buffer of buffers) buffer.destroy();
          ownedReservation.release();
        }
      };
      return this.topology;
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      reservation?.release();
      throw error;
    } finally {
      staging.release();
    }
  }

  private createEntry(key: string, settings: KeplerPolygonWeightsSettings): CacheEntry {
    const topology = this.prepareTopology();
    const rowCount = this.dataset.rowCount;
    const outputBytes = (rowCount + 1) * 4 + settings.neighborCapacity * 8 + 16;
    // Include conservative sort/scan workspace beyond the contributor's declared transients.
    const scratchBytes =
      (topology.vertexCount ? topology.vertexCount * 256 + settings.pairCapacity * 192 : 0) +
      rowCount * 128 +
      4 * 1024 * 1024;
    const buffers: Buffer[] = [];
    let outputReservation: KeplerMemoryReservation | undefined;
    let scratchReservation: KeplerMemoryReservation | undefined;
    let compiled: CompiledGPUCommandGraph | undefined;
    let result: ReturnType<typeof addKeplerPolygonWeights> | undefined;
    const identifier = `kepler-polygon-cache-${++this.sequence}`;
    try {
      outputReservation = this.session.registry.reserveMemory(outputBytes, 'gpu', 'output');
      scratchReservation = this.session.registry.reserveMemory(scratchBytes, 'gpu', 'scratch');
      const allocate = (name: string, length: number) => {
        const buffer = this.session.device.createBuffer({
          id: `${identifier}-${name}`,
          byteLength: Math.max(1, length) * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return buffer;
      };
      const offsets = allocate('offsets', rowCount + 1);
      const neighbors = allocate('neighbors', settings.neighborCapacity);
      const weights = allocate('weights', settings.neighborCapacity);
      const diagnostics: PolygonDiagnostics = [
        allocate('overflow', 1),
        allocate('required-count', 1),
        allocate('candidate-overflow', 1),
        allocate('invalid-count', 1)
      ];
      const importOutputs = (graph: GPUCommandGraph) => ({
        offsets: importBuffer(graph, `${identifier}-offsets`, offsets, 'uint32', rowCount + 1),
        neighbors: importBuffer(
          graph,
          `${identifier}-neighbors`,
          neighbors,
          'uint32',
          settings.neighborCapacity
        ),
        weights: importBuffer(
          graph,
          `${identifier}-weights`,
          weights,
          'float32',
          settings.neighborCapacity
        ),
        status: {
          overflow: importBuffer(graph, `${identifier}-overflow`, diagnostics[0], 'uint32', 1),
          requiredCount: importBuffer(
            graph,
            `${identifier}-required-count`,
            diagnostics[1],
            'uint32',
            1
          ),
          candidateOverflow: importBuffer(
            graph,
            `${identifier}-candidate-overflow`,
            diagnostics[2],
            'uint32',
            1
          ),
          invalidCount: importBuffer(
            graph,
            `${identifier}-invalid-count`,
            diagnostics[3],
            'uint32',
            1
          )
        }
      });
      const graph = new GPUCommandGraph(this.session.device, {id: identifier});
      const imported = topology.importToGraph(graph);
      const output = importOutputs(graph);
      result = addKeplerPolygonWeights(graph, {
        id: identifier,
        source: {datasetId: this.dataset.id, revision: this.dataset.revision},
        settings,
        polygons: imported.polygons,
        geometryMask: imported.geometryMask,
        output
      });
      compiled = graph.compile();
      if (compiled.stats.physicalTransientResourceBytes + result.parameterByteLength > scratchBytes)
        throw new Error('Compiled polygon weights exceed their memory reservation');
      const ownedCompiled = compiled;
      const ownedResult = result;
      const ownedOutputReservation = outputReservation;
      const ownedScratchReservation = scratchReservation;
      return {
        key,
        references: 0,
        ready: false,
        byteLength: outputBytes + scratchBytes,
        diagnostics,
        compiled: ownedCompiled,
        importBinding: consumerGraph => {
          const importedOutputs = importOutputs(consumerGraph);
          const importedTopology = topology.importToGraph(consumerGraph);
          return {
            source: ownedResult.source,
            settings: {...settings},
            geometryMask: importedTopology.geometryMask,
            status: importedOutputs.status,
            weights: {
              offsets: importedOutputs.offsets,
              neighbors: importedOutputs.neighbors,
              weights: importedOutputs.weights,
              status: importedOutputs.status,
              information: ownedResult.weights.information
            }
          };
        },
        destroy: () => {
          ownedCompiled.destroy();
          ownedResult.destroyParameters();
          for (const buffer of buffers) buffer.destroy();
          ownedOutputReservation.release();
          ownedScratchReservation.release();
        }
      };
    } catch (error) {
      compiled?.destroy();
      result?.destroyParameters();
      for (const buffer of buffers) buffer.destroy();
      outputReservation?.release();
      scratchReservation?.release();
      throw error;
    }
  }
}

function importBuffer<Format extends 'float32x2' | 'float32' | 'uint32'>(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(
    graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
    {format, length}
  );
}
