// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  createTransientView,
  type CompiledGPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  addKeplerPointWeights,
  assertKeplerStorageBindingFits,
  getKeplerPointWeightCoordinateColumns,
  getKeplerPointWeightsCapacity,
  parseKeplerPointWeightsSettings,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerPointWeightsBinding,
  type KeplerPointWeightsSettings,
  type KeplerMemoryReservation
} from '@luma-contributors/kepler';
import {getPointWeightsScratchBytes} from './analysis-point-weights-settings';

type CacheEntry = {
  key: string;
  references: number;
  ready: boolean;
  pendingEncoder?: CommandEncoder;
  successfulEncoder?: CommandEncoder;
  byteLength: number;
  diagnostics: readonly Buffer[];
  importBinding(graph: GPUCommandGraph): KeplerPointWeightsBinding;
  compiled: CompiledGPUCommandGraph;
  destroy(): void;
};

export type AnalysisPointWeightsLease = {
  binding: KeplerPointWeightsBinding;
  diagnostics: readonly Buffer[];
  /** Which storage built these weights (single CSR here; streamed chunks in the streamed cache). */
  path?: {
    path: 'single-csr' | 'streamed';
    chunkCount: number;
    neighborCount?: number;
    maximumSlotsPerChunk?: number;
    reason?: 'device-limit' | 'forced';
  };
  encode(commandEncoder: CommandEncoder): void;
  /** Only a successfully encoded consumer can acknowledge the topology at submission. */
  markConsumerEncoded(commandEncoder: CommandEncoder): void;
  retain(): () => void;
  release(): void;
};

/** Source-owned topology, independent of event fields, population selection, rates and inference. */
export class AnalysisPointWeightsCache {
  private readonly entries = new Map<string, CacheEntry>();
  private mostRecentKey?: string;
  private sequence = 0;
  private buildCount = 0;
  private encodeCount = 0;
  private destroyed = false;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>
  ) {}

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
      byteLength: [...this.entries.values()].reduce((total, entry) => total + entry.byteLength, 0)
    };
  }

  acquire(
    graph: GPUCommandGraph,
    requestedSettings: KeplerPointWeightsSettings
  ): AnalysisPointWeightsLease {
    if (this.destroyed) throw new Error('Point weight cache is destroyed');
    const settings = parseKeplerPointWeightsSettings(requestedSettings);
    const key = JSON.stringify([this.dataset.id, this.dataset.revision, settings]);
    let entry = this.entries.get(key);
    if (!entry) {
      this.evictUnused();
      if (this.entries.size >= 4)
        throw new Error('Submit pending analysis work before changing point topology again');
      entry = this.createEntry(key, settings);
      this.entries.set(key, entry);
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
      const binding = cached.importBinding(graph);
      return {
        binding,
        diagnostics: cached.diagnostics,
        path: {path: 'single-csr', chunkCount: 1},
        encode: commandEncoder => {
          if (released || this.destroyed) throw new Error('Point weight lease is unavailable');
          if (cached.ready || cached.pendingEncoder === commandEncoder) return;
          cached.compiled.encode(commandEncoder, {parameters: undefined});
          cached.pendingEncoder = commandEncoder;
          this.encodeCount++;
        },
        markConsumerEncoded: commandEncoder => {
          if (released || this.destroyed) throw new Error('Point weight lease is unavailable');
          if (cached.pendingEncoder === commandEncoder) cached.successfulEncoder = commandEncoder;
        },
        retain: () => {
          if (released || this.destroyed) throw new Error('Point weight lease is unavailable');
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

  /** Host calls this only after submitting this encoder, never after mere recording. */
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

  /** All program fences and asynchronous readers must have released their leases first. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const entry of this.entries.values()) entry.destroy();
    this.entries.clear();
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

  private createEntry(key: string, settings: KeplerPointWeightsSettings): CacheEntry {
    const rowCount = this.dataset.rowCount;
    const neighborCapacity = getKeplerPointWeightsCapacity(settings, rowCount);
    const outputBytes = (rowCount + 1) * 4 + neighborCapacity * 8 + rowCount * 4 + 12;
    const scratchBytes = getPointWeightsScratchBytes(settings, rowCount);
    const buffers: Buffer[] = [];
    let outputReservation: KeplerMemoryReservation | undefined;
    let scratchReservation: KeplerMemoryReservation | undefined;
    let compiled: CompiledGPUCommandGraph | undefined;
    let result: ReturnType<typeof addKeplerPointWeights> | undefined;
    const identifier = `kepler-point-cache-${++this.sequence}`;
    try {
      // Fail before allocating when the neighbor slots cannot be bound as one storage buffer.
      assertKeplerStorageBindingFits(
        this.session.device.limits,
        neighborCapacity,
        4,
        'Neighbor slots'
      );
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
      const neighbors = allocate('neighbors', neighborCapacity);
      const weights = allocate('weights', neighborCapacity);
      const coordinateMask = allocate('coordinate-mask', rowCount);
      const diagnostics = ['overflow', 'required-count', 'invalid-count'].map(name =>
        allocate(name, 1)
      );
      const importOutputs = (graph: GPUCommandGraph) => {
        const view = <Format extends 'uint32' | 'float32'>(
          name: string,
          buffer: Buffer,
          format: Format,
          length: number
        ): GraphDataView<Format> =>
          graph.createDataView(
            graph.importBuffer(
              {id: `${identifier}-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
              buffer
            ),
            {format, length}
          );
        const status = {
          overflow: view('overflow', diagnostics[0], 'uint32', 1),
          requiredCount: view('required-count', diagnostics[1], 'uint32', 1),
          invalidCount: view('invalid-count', diagnostics[2], 'uint32', 1)
        };
        return {
          offsets: view('offsets', offsets, 'uint32', rowCount + 1),
          neighbors: view('neighbors', neighbors, 'uint32', neighborCapacity),
          weights: view('weights', weights, 'float32', neighborCapacity),
          coordinateMask: view('coordinate-mask', coordinateMask, 'uint32', rowCount),
          status
        };
      };
      const graph = new GPUCommandGraph(this.session.device, {id: identifier});
      const columns = new Map<string, GraphDataView<'float32'>>();
      const validity: GraphDataView<'uint32'>[] = [];
      for (const [index, name] of [
        ...new Set(getKeplerPointWeightCoordinateColumns(settings))
      ].entries()) {
        const source = this.dataset.value.columns[name];
        if (!source || source.format !== 'float32')
          throw new Error('Point weights require prepared Float32 coordinates');
        columns.set(
          name,
          packAnalysisVector(
            graph,
            `packed-coordinate-${index}`,
            graph.importGPUVector(`source-coordinate-${index}`, source as GPUVector<'float32'>)
          )
        );
        const sourceValidity = this.dataset.value.analytics.validity[name];
        if (sourceValidity)
          validity.push(
            packAnalysisVector(
              graph,
              `packed-validity-${index}`,
              graph.importGPUVector(`coordinate-validity-${index}`, sourceValidity)
            )
          );
      }
      let coordinateValidityMask = validity[0];
      if (validity.length > 1) {
        coordinateValidityMask = createTransientView(
          graph,
          'coordinate-validity',
          'uint32',
          rowCount
        );
        graph.add(
          new GPUMask({id: 'coordinate-validity', inputs: validity, output: coordinateValidityMask})
        );
      }
      const outputs = importOutputs(graph);
      result = addKeplerPointWeights(graph, {
        id: identifier,
        source: {datasetId: this.dataset.id, revision: this.dataset.revision},
        ...(settings.coordinateSpace === 'planar'
          ? {settings, x: columns.get(settings.xColumn)!, y: columns.get(settings.yColumn)!}
          : {
              settings,
              longitude: columns.get(settings.longitudeColumn)!,
              latitude: columns.get(settings.latitudeColumn)!
            }),
        coordinateValidityMask,
        coordinateMask: outputs.coordinateMask,
        output: {
          offsets: outputs.offsets,
          neighbors: outputs.neighbors,
          weights: outputs.weights,
          status: outputs.status
        }
      });
      compiled = graph.compile();
      if (compiled.stats.physicalTransientResourceBytes + result.parameterByteLength > scratchBytes)
        throw new Error('Compiled point weights exceed their memory reservation');
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
          const imported = importOutputs(consumerGraph);
          return {
            source: ownedResult.source,
            settings: {...settings},
            coordinateMask: imported.coordinateMask,
            status: imported.status,
            weights: {
              offsets: imported.offsets,
              neighbors: imported.neighbors,
              weights: imported.weights,
              status: imported.status,
              information: ownedResult.weights.information,
              spatialContext: ownedResult.weights.spatialContext
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

/** Explicit source-order workspace; never changes the borrowed source chunk topology. */
export function packAnalysisVector<Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph,
  id: string,
  vector: GraphVectorView<Format>
): GraphDataView<Format> {
  if (vector.data.length === 1) return vector.data[0];
  const output = createTransientView(
    graph,
    id,
    vector.format,
    vector.length,
    Buffer.STORAGE | Buffer.COPY_DST
  );
  let rowOffset = 0;
  for (const [index, chunk] of vector.data.entries()) {
    if (chunk.length) {
      const target = graph.createDataView(output.buffer, {
        format: vector.format,
        byteOffset: rowOffset * 4,
        length: chunk.length
      });
      graph.addCopyPass({
        id: `${id}-${index}`,
        resources: [
          {buffer: chunk, usage: 'copy-source'},
          {buffer: target, usage: 'copy-destination'}
        ],
        compile: () => ({
          encode: ({commandEncoder, getBuffer}) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: getBuffer(chunk),
              sourceOffset: chunk.byteOffset,
              destinationBuffer: getBuffer(target),
              destinationOffset: target.byteOffset,
              size: chunk.length * 4
            })
        })
      });
    }
    rowOffset += chunk.length;
  }
  return output;
}
