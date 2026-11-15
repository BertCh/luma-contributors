// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUMask, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';

type AltitudeMask = {
  datasetId: string;
  sourceMask: GPUVector<'uint32'>;
  validity: GPUVector<'uint32'>;
  output: GPUVector<'uint32'>;
  buffers: Buffer[];
  graph: GPUCommandGraph;
  compiled: CompiledGPUCommandGraph;
  lease: KeplerDatasetLease;
  reservation: KeplerMemoryReservation;
};

/** Source rows stay fixed; only layers using the uploaded altitude apply its finite-value mask. */
export class PointAltitudeBindingRuntime {
  private readonly entries = new Map<string, AltitudeMask>();
  private readonly pendingRetirements = new Set<AltitudeMask>();
  private readonly retirements = new Set<Promise<void>>();
  private readonly errors: unknown[] = [];
  private destroyed = false;

  constructor(
    private readonly session: KeplerComputeSession,
    private readonly requestUpdate: () => void
  ) {}

  resolve(
    datasetId: string,
    layerId: string,
    sourceMask: GPUVector<'uint32'>,
    validity: GPUVector<'uint32'> | undefined
  ): GPUVector<'uint32'> {
    if (this.destroyed || this.session.deviceLost)
      throw new Error('Point altitude owner is unavailable');
    const previous = this.entries.get(layerId);
    if (
      previous &&
      previous.datasetId === datasetId &&
      previous.sourceMask === sourceMask &&
      previous.validity === validity &&
      previous.lease.isValid()
    )
      return previous.output;
    if (previous) {
      this.entries.delete(layerId);
      this.pendingRetirements.add(previous);
    }
    if (!validity) return sourceMask;
    if (
      validity.length !== sourceMask.length ||
      validity.data.length !== sourceMask.data.length ||
      validity.data.some((chunk, index) => chunk.length !== sourceMask.data[index].length)
    )
      throw new Error('Altitude validity must preserve source mask chunks');
    const outputBytes = sourceMask.data.reduce(
      (total, chunk) => total + Math.max(4, chunk.length * 4),
      0
    );
    const reservation = this.session.registry.reserveMemory(
      outputBytes + 1024 * 1024,
      'gpu',
      'output'
    );
    const buffers: Buffer[] = [];
    const graph = new GPUCommandGraph(this.session.device);
    let lease: KeplerDatasetLease | undefined;
    let output: GPUVector<'uint32'> | undefined;
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      lease = this.session.registry.acquireDataset(datasetId);
      output = new GPUVector({
        type: 'data',
        name: `${layerId}-altitude-mask`,
        format: 'uint32',
        ownsData: false,
        data: sourceMask.data.map(chunk => {
          const buffer = this.session.device.createBuffer({
            byteLength: Math.max(4, chunk.length * 4),
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          });
          buffers.push(buffer);
          return new GPUData({format: 'uint32', length: chunk.length, buffer, ownsBuffer: false});
        })
      });
      graph.add(
        new GPUMask({
          id: 'finite-altitude-selection',
          inputs: [
            graph.importGPUVector('selection', sourceMask),
            graph.importGPUVector('altitude-validity', validity)
          ],
          output: graph.importGPUVector('output', output)
        })
      );
      // Published luma.gl has no pre-compile resource plan: compile, then admit the measured
      // transient bytes. The catch below destroys the compiled graph if admission fails.
      compiled = graph.compile();
      const transientBytes = compiled.stats.physicalTransientResourceBytes;
      if (transientBytes > 1024 * 1024)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Point altitude mask exceeds admitted working memory'
        });
      reservation.resize(outputBytes + transientBytes);
      this.entries.set(layerId, {
        datasetId,
        sourceMask,
        validity,
        output,
        buffers,
        graph,
        compiled,
        lease,
        reservation
      });
      this.requestUpdate();
      return output;
    } catch (error) {
      compiled?.destroy();
      output?.destroy();
      for (const buffer of buffers) buffer.destroy();
      reservation.release();
      if (lease) this.track(lease.release());
      throw error;
    }
  }

  encode(encoder: CommandEncoder): void {
    for (const entry of this.entries.values())
      if (entry.lease.isValid()) entry.compiled.encode(encoder, {parameters: undefined});
  }

  afterRender(): void {
    if (this.destroyed || !this.pendingRetirements.size) return;
    const fence = this.session.deviceLost ? undefined : this.session.device.createFence();
    const completion = Promise.race([
      fence?.signaled ?? Promise.resolve(),
      this.session.device.lost
    ]);
    for (const entry of this.pendingRetirements) this.retire(entry, completion);
    this.pendingRetirements.clear();
    void completion.finally(() => fence?.destroy()).catch(() => undefined);
  }

  async retainLayers(ids: readonly string[], completion: PromiseLike<unknown>): Promise<void> {
    const retained = new Set(ids);
    for (const [id, entry] of this.entries)
      if (!retained.has(id)) {
        this.entries.delete(id);
        this.retire(entry, completion);
      }
    for (const entry of this.pendingRetirements) {
      this.pendingRetirements.delete(entry);
      this.retire(entry, completion);
    }
    await this.drain();
  }

  async retireDataset(datasetId: string, completion: PromiseLike<unknown>): Promise<void> {
    for (const [id, entry] of this.entries)
      if (entry.datasetId === datasetId) {
        this.entries.delete(id);
        this.retire(entry, completion);
      }
    for (const entry of this.pendingRetirements)
      if (entry.datasetId === datasetId) {
        this.pendingRetirements.delete(entry);
        this.retire(entry, completion);
      }
    await this.drain();
  }

  async destroy(completion: PromiseLike<unknown>): Promise<void> {
    this.destroyed = true;
    for (const entry of this.pendingRetirements) this.retire(entry, completion);
    this.pendingRetirements.clear();
    await this.retainLayers([], completion);
  }

  private retire(entry: AltitudeMask, completion: PromiseLike<unknown>): void {
    this.track(
      (async () => {
        try {
          await Promise.race([Promise.resolve(completion), this.session.device.lost]);
        } finally {
          try {
            entry.compiled.destroy();
            entry.output.destroy();
            for (const buffer of entry.buffers) buffer.destroy();
          } finally {
            entry.reservation.release();
            await entry.lease.release();
          }
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
      throw new AggregateError(this.errors.splice(0), 'Altitude mask cleanup failed');
  }
}
