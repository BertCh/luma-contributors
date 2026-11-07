// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  type CompiledGPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  buildKeplerStreamedPointWeights,
  getKeplerMaximumBindingElements,
  getKeplerPointWeightCoordinateColumns,
  importKeplerStreamedPointWeights,
  parseKeplerPointWeightsSettings,
  type KeplerAnalysisRecipe,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerMemoryReservation,
  type KeplerPointWeightsBinding,
  type KeplerPointWeightsSettings,
  type KeplerStreamedPointWeights
} from '@luma-contributors/kepler';
import {packAnalysisVector, type AnalysisPointWeightsLease} from './analysis-point-weights-cache';

/** Which storage a point-weights build used, shown in the Analysis panel. */
export type AnalysisPointWeightsPath = {
  path: 'single-csr' | 'streamed';
  /** CSR chunks (1 for the single CSR). */
  chunkCount: number;
  /** Neighbor links, when known without a readback (streamed builds only). */
  neighborCount?: number;
  /** Per-chunk slot limit that sized the chunks (streamed builds only). */
  maximumSlotsPerChunk?: number;
  /** Why the streamed path was chosen. */
  reason?: 'device-limit' | 'forced';
};

export const SINGLE_CSR_POINT_WEIGHTS_PATH: AnalysisPointWeightsPath = Object.freeze({
  path: 'single-csr',
  chunkCount: 1
});

export type AnalysisStreamedWeightsOptions = {
  /** Debug and test override: stream even when the single CSR fits. */
  forceStreamed?: boolean;
  /** Debug and test override of the per-chunk slot limit (defaults to the device limit). */
  maximumSlotsPerChunk?: number;
};

/**
 * Point weights of the operations whose consumers dispatch their weights kernels once per chunk
 * (point hotspots, Local Moran, quantile LISA, spatial rates and multivariate Geary); other operations always use the
 * single CSR.
 */
export function getStreamedWeightsSettings(
  operation: KeplerAnalysisRecipe['operation']
): KeplerPointWeightsSettings | undefined {
  return operation.kind === 'hot-spots-points' ||
    operation.kind === 'local-moran-points' ||
    operation.kind === 'quantile-lisa-points' ||
    operation.kind === 'spatial-rate-points' ||
    operation.kind === 'multivariate-local-geary-points'
    ? operation.weights
    : undefined;
}

/**
 * Chooses the point-weights storage: streamed row chunks when the distance-band single CSR would
 * not fit one storage binding of the device (or when forced), else the single CSR.
 */
export function choosePointWeightsPath(
  limits: {maxStorageBufferBindingSize: number; maxBufferSize: number},
  settings: KeplerPointWeightsSettings,
  options: AnalysisStreamedWeightsOptions = {}
): AnalysisPointWeightsPath {
  const parsed = parseKeplerPointWeightsSettings(settings);
  if (parsed.kind !== 'distance-band') return SINGLE_CSR_POINT_WEIGHTS_PATH;
  const maximumSlotsPerChunk =
    options.maximumSlotsPerChunk ?? getKeplerMaximumBindingElements(limits, 4);
  const exceedsLimit = parsed.neighborCapacity > getKeplerMaximumBindingElements(limits, 4);
  if (!exceedsLimit && !options.forceStreamed) return SINGLE_CSR_POINT_WEIGHTS_PATH;
  return {
    path: 'streamed',
    chunkCount: 0,
    maximumSlotsPerChunk,
    reason: exceedsLimit ? 'device-limit' : 'forced'
  };
}

type StreamedEntry = {
  key: string;
  streamed: KeplerStreamedPointWeights;
  report: AnalysisPointWeightsPath;
  statusBuffers: Buffer[];
  reservation: KeplerMemoryReservation;
  references: number;
  retired: boolean;
};

/** Copies a (possibly chunked) vector into one caller-owned buffer; source chunks are borrowed. */
function addCopyVectorToView<Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph,
  id: string,
  vector: GraphVectorView<Format>,
  destination: GraphDataView<Format>
): void {
  let rowOffset = 0;
  for (const [index, chunk] of vector.data.entries()) {
    if (chunk.length) {
      const target = graph.createDataView(destination.buffer, {
        format: vector.format,
        byteOffset: destination.byteOffset + rowOffset * 4,
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
}

function createStorageBuffer(device: Device, id: string, wordCount: number): Buffer {
  return device.createBuffer({
    id,
    byteLength: Math.max(wordCount, 1) * 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
}

function importDestination<Format extends 'float32' | 'uint32'>(
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

/**
 * Source-owned streamed point weights (chunked CSR) for one dataset revision. The build is
 * asynchronous (a count pass is read back to size the chunks), so the runtime starts it with
 * {@link AnalysisStreamedPointWeights.prepare} and acquires leases once it is ready. A lease has
 * the same shape as the single-CSR cache lease; its binding carries `GPUChunkedSpatialWeights`.
 */
export class AnalysisStreamedPointWeights {
  private readonly entries = new Map<string, StreamedEntry>();
  private readonly pending = new Map<string, Promise<void>>();
  private readonly failures = new Map<string, string>();
  private destroyed = false;
  private sequence = 0;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>
  ) {}

  private getKey(settings: KeplerPointWeightsSettings): string {
    return JSON.stringify([this.dataset.id, this.dataset.revision, settings]);
  }

  isReady(settings: KeplerPointWeightsSettings): boolean {
    return this.entries.has(this.getKey(parseKeplerPointWeightsSettings(settings)));
  }

  isPending(settings: KeplerPointWeightsSettings): boolean {
    return this.pending.has(this.getKey(parseKeplerPointWeightsSettings(settings)));
  }

  /** The error of a failed build for these settings, if any. */
  getFailure(settings: KeplerPointWeightsSettings): string | undefined {
    return this.failures.get(this.getKey(parseKeplerPointWeightsSettings(settings)));
  }

  /** Starts (once) the asynchronous build for `settings`; resolves when the entry is ready. */
  prepare(
    requestedSettings: KeplerPointWeightsSettings,
    options: AnalysisStreamedWeightsOptions & {reason?: 'device-limit' | 'forced'} = {}
  ): Promise<void> {
    if (this.destroyed) return Promise.reject(new Error('Streamed point weights are destroyed'));
    const settings = parseKeplerPointWeightsSettings(requestedSettings);
    const key = this.getKey(settings);
    if (this.entries.has(key)) return Promise.resolve();
    const existing = this.pending.get(key);
    if (existing) return existing;
    this.failures.delete(key);
    const build = this.build(key, settings, options)
      .catch(error => {
        this.failures.set(key, error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        this.pending.delete(key);
      });
    this.pending.set(key, build);
    return build;
  }

  private async build(
    key: string,
    settings: KeplerPointWeightsSettings,
    options: AnalysisStreamedWeightsOptions & {reason?: 'device-limit' | 'forced'}
  ): Promise<void> {
    if (settings.kind !== 'distance-band')
      throw new Error('Streamed point weights support distance-band topology');
    const {device} = this.session;
    const rowCount = this.dataset.rowCount;
    const identifier = `kepler-streamed-weights-${++this.sequence}`;
    const owned: Buffer[] = [];
    const track = (buffer: Buffer) => {
      owned.push(buffer);
      return buffer;
    };
    let streamed: KeplerStreamedPointWeights | undefined;
    let reservation: KeplerMemoryReservation | undefined;
    try {
      // Pack the coordinate columns (and combined null validity) into one buffer each.
      const [firstName, secondName] = getKeplerPointWeightCoordinateColumns(settings);
      const firstCoordinate = track(createStorageBuffer(device, `${identifier}-first`, rowCount));
      const secondCoordinate = track(createStorageBuffer(device, `${identifier}-second`, rowCount));
      const packGraph = new GPUCommandGraph(device, {id: `${identifier}-pack`});
      for (const [index, [name, destination]] of [
        [firstName, firstCoordinate],
        [secondName, secondCoordinate]
      ].entries()) {
        const source = this.dataset.value.columns[name as string];
        if (!source || source.format !== 'float32')
          throw new Error('Point weights require prepared Float32 coordinates');
        addCopyVectorToView(
          packGraph,
          `${identifier}-coordinate-${index}`,
          packGraph.importGPUVector(`source-coordinate-${index}`, source as GPUVector<'float32'>),
          importDestination(
            packGraph,
            `${identifier}-coordinate-destination-${index}`,
            destination as Buffer,
            'float32',
            rowCount
          )
        );
      }
      const validityVectors = [...new Set([firstName, secondName])].flatMap((name, index) => {
        const validity = this.dataset.value.analytics.validity[name];
        return validity
          ? [packGraph.importGPUVector(`coordinate-validity-${index}`, validity)]
          : [];
      });
      let coordinateValidityMask: Buffer | undefined;
      if (validityVectors.length) {
        coordinateValidityMask = track(
          createStorageBuffer(device, `${identifier}-validity`, rowCount)
        );
        const destination = importDestination(
          packGraph,
          `${identifier}-validity-destination`,
          coordinateValidityMask,
          'uint32',
          rowCount
        );
        if (validityVectors.length === 1) {
          addCopyVectorToView(packGraph, `${identifier}-validity`, validityVectors[0], destination);
        } else {
          packGraph.add(
            new GPUMask({
              id: `${identifier}-validity`,
              inputs: validityVectors.map((vector, index) =>
                packAnalysisVector(packGraph, `${identifier}-validity-packed-${index}`, vector)
              ),
              output: destination
            })
          );
        }
      }
      const compiled = packGraph.compile() as CompiledGPUCommandGraph<undefined>;
      try {
        const commandEncoder = device.createCommandEncoder({id: `${identifier}-pack-encoder`});
        compiled.encode(commandEncoder, {parameters: undefined});
        device.submit(commandEncoder.finish());
      } finally {
        compiled.destroy();
      }

      streamed = await buildKeplerStreamedPointWeights({
        device,
        rowCount,
        firstCoordinate,
        secondCoordinate,
        coordinateValidityMask,
        settings,
        maximumSlotsPerChunk: options.maximumSlotsPerChunk
      });
      const chunkBytes = streamed.chunks.reduce(
        (total, chunk) =>
          total + chunk.offsets.byteLength + chunk.neighbors.byteLength + chunk.weights.byteLength,
        0
      );
      reservation = this.session.registry.reserveMemory(
        chunkBytes + rowCount * 4 + 12,
        'gpu',
        'output'
      );
      // Diagnostics in the order the runtime reads them: overflow, required count, invalid count.
      const statusBuffers = [0, streamed.neighborCount, streamed.invalidCount].map(
        (value, index) => {
          const buffer = createStorageBuffer(device, `${identifier}-status-${index}`, 1);
          buffer.write(new Uint32Array([value]));
          return buffer;
        }
      );
      owned.forEach(buffer => buffer.destroy());
      this.entries.set(key, {
        key,
        streamed,
        statusBuffers,
        reservation,
        references: 0,
        retired: false,
        report: {
          path: 'streamed',
          chunkCount: streamed.chunks.length,
          neighborCount: streamed.neighborCount,
          maximumSlotsPerChunk: options.maximumSlotsPerChunk,
          reason: options.reason ?? 'forced'
        }
      });
    } catch (error) {
      streamed?.destroy();
      reservation?.release();
      owned.forEach(buffer => buffer.destroy());
      throw error;
    }
  }

  /** Imports a ready entry into `graph`; throws when it was not prepared. */
  acquire(
    graph: GPUCommandGraph,
    requestedSettings: KeplerPointWeightsSettings
  ): AnalysisPointWeightsLease {
    if (this.destroyed) throw new Error('Streamed point weights are destroyed');
    const settings = parseKeplerPointWeightsSettings(requestedSettings);
    const entry = this.entries.get(this.getKey(settings));
    // Not ready: the runtime awaits prepare() before building a program.
    if (!entry) throw new Error('Streamed point weights are not ready');
    entry.references++;
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      entry.references--;
      if (entry.retired && entry.references === 0) this.destroyEntry(entry);
    };
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ): GraphDataView<Format> =>
      graph.createDataView(
        graph.importBuffer(
          {id: `${entry.key}-${name}`, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        ),
        {format, length}
      );
    try {
      const status = {
        overflow: view('overflow', entry.statusBuffers[0], 'uint32', 1),
        requiredCount: view('required-count', entry.statusBuffers[1], 'uint32', 1),
        invalidCount: view('invalid-count', entry.statusBuffers[2], 'uint32', 1)
      };
      const binding: KeplerPointWeightsBinding = {
        source: {datasetId: this.dataset.id, revision: this.dataset.revision},
        settings: {...settings},
        coordinateMask: view(
          'coordinate-mask',
          entry.streamed.coordinateMask,
          'uint32',
          this.dataset.rowCount
        ),
        status,
        weights: {...importKeplerStreamedPointWeights(graph, entry.streamed), status}
      };
      return {
        binding,
        diagnostics: entry.statusBuffers,
        path: entry.report,
        // The chunks were built and submitted by prepare(); nothing to record.
        encode: () => {},
        markConsumerEncoded: () => {},
        retain: () => {
          entry.references++;
          let finished = false;
          return () => {
            if (finished) return;
            finished = true;
            entry.references--;
            if (entry.retired && entry.references === 0) this.destroyEntry(entry);
          };
        },
        release
      };
    } catch (error) {
      release();
      throw error;
    }
  }

  /** Retires entries that no program uses (the entry for `keepSettings` stays). */
  retireOthers(keepSettings: KeplerPointWeightsSettings): void {
    const keepKey = this.getKey(parseKeplerPointWeightsSettings(keepSettings));
    for (const entry of [...this.entries.values()]) {
      if (entry.key === keepKey) continue;
      this.entries.delete(entry.key);
      entry.retired = true;
      if (entry.references === 0) this.destroyEntry(entry);
    }
  }

  private destroyEntry(entry: StreamedEntry): void {
    entry.streamed.destroy();
    entry.statusBuffers.forEach(buffer => buffer.destroy());
    entry.reservation.release();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const entry of this.entries.values()) this.destroyEntry(entry);
    this.entries.clear();
  }
}
