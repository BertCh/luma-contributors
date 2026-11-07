// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {GPUReadbackRing} from '@luma.gl/graph/gpu-core';

/** The part of an app's resource tracker that {@link SummaryReader} needs. */
export type SummaryReaderResources = {
  device: Device;
  track<T extends {destroy(): void}>(resource: T): T;
  release(resource: {destroy(): void}): void;
};

export type SummaryReaderOptions = {
  /**
   * When set, errors thrown by `onResult` and readbacks that end unexpectedly are logged with this
   * label. When omitted, a failed readback is dropped silently (the ring or device was destroyed).
   */
  logLabel?: string;
};

/**
 * Small summary readback: copies `sources` back to back into a ring ticket and hands the bytes to
 * `onResult`. At most one read is in flight.
 */
export class SummaryReader {
  private pending = false;
  private stale = false;
  private destroyed = false;
  private readonly ring: GPUReadbackRing;
  private readonly resources: SummaryReaderResources;
  private readonly sources: readonly {buffer: Buffer; size: number}[];
  private readonly byteLength: number;
  private readonly onResult: (bytes: ArrayBuffer) => void;
  private readonly logLabel?: string;

  constructor(
    resources: SummaryReaderResources,
    id: string,
    sources: readonly {buffer: Buffer; size: number}[],
    onResult: (bytes: ArrayBuffer) => void,
    options: SummaryReaderOptions = {}
  ) {
    this.resources = resources;
    this.sources = sources;
    this.byteLength = sources.reduce((total, source) => total + source.size, 0);
    this.onResult = onResult;
    this.logLabel = options.logLabel;
    this.ring = resources.track(
      new GPUReadbackRing(resources.device, {id: `summary-${id}`, byteLength: this.byteLength})
    );
  }

  /** True while a read is in flight. */
  get isPending(): boolean {
    return this.pending;
  }

  /** Marks the summary stale without recording a copy; {@link flush} reads it when possible. */
  markStale(): void {
    this.stale = true;
  }

  /** Marks the summary stale and reads it now, or on the next {@link flush} if a read is in flight. */
  request(commandEncoder: CommandEncoder): void {
    this.stale = true;
    this.flush(commandEncoder);
  }

  /**
   * Records the copies into `commandEncoder` when the summary is stale and no read is in flight,
   * and delivers the bytes when they are ready. Call it every frame after the graphs are encoded
   * so a request made while a read was in flight is not lost.
   */
  flush(commandEncoder: CommandEncoder): void {
    if (!this.stale || this.pending || this.destroyed) return;
    const ticket = this.ring.tryAcquire();
    if (!ticket) return;
    let offset = 0;
    for (const {buffer, size} of this.sources) {
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: buffer,
        destinationBuffer: ticket.buffer,
        destinationOffset: offset,
        size
      });
      offset += size;
    }
    ticket.markEncoded({byteOffset: 0, byteLength: this.byteLength});
    this.stale = false;
    this.pending = true;
    // `GPUReadbackRing` requires the encoder to be submitted before `read()` maps the slot. Apps
    // flush inside deck's frame, and deck submits that encoder synchronously before the frame
    // callback returns, so the next task runs after the submit.
    void new Promise<void>(resolve => setTimeout(resolve, 0))
      .then(() => ticket.read())
      .then(bytes => {
        if (this.destroyed) return;
        try {
          this.onResult(bytes.slice().buffer);
        } catch (error) {
          if (!this.logLabel) throw error;
          // A bug in the callback must be visible, not swallowed as a lost read.
          // biome-ignore lint/suspicious/noConsole: surfaces failures that would otherwise be swallowed
          console.error(`[summary-reader ${this.logLabel}] onResult threw`, error);
        }
      })
      .catch(error => {
        // The ring or device was destroyed while the read was in flight.
        // biome-ignore lint/suspicious/noConsole: surfaces failures that would otherwise be swallowed
        if (this.logLabel && !this.destroyed) {
          // biome-ignore lint/suspicious/noConsole: surfaces failures that would otherwise be swallowed
          console.debug(`[summary-reader ${this.logLabel}] readback ended`, error);
        }
      })
      .finally(() => {
        this.pending = false;
      });
  }

  /** Stops delivering results. */
  stop(): void {
    this.destroyed = true;
  }

  /**
   * Stops delivering results and releases the readback ring early, for a reader whose graph was
   * replaced. A read in flight then ends quietly.
   */
  release(): void {
    this.stop();
    this.resources.release(this.ring);
  }
}
