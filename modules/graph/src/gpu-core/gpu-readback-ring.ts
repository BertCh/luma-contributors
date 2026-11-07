// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';

/** Construction properties for {@link GPUReadbackRing}. */
export type GPUReadbackRingProps = {
  /** Prefix for the staging-buffer IDs. */
  id?: string;
  /** Capacity of every staging buffer, in bytes. Must be a positive multiple of four. */
  byteLength: number;
  /** Number of independently reusable staging buffers. Defaults to three. */
  slotCount?: number;
};

type GPUReadbackTicketState = 'reserved' | 'encoded' | 'reading' | 'released';

/**
 * One exclusive reservation in a {@link GPUReadbackRing}.
 *
 * A ticket may record one copy, or expose its buffer to another encoder and call `markEncoded()`.
 * The application submits the encoder before calling `read()`. A slot is returned to the ring only
 * after mapping finishes, so a mapped or in-flight staging buffer is never reused.
 */
export class GPUReadbackTicket {
  /** Caller-visible staging buffer for graph imports and texture-to-buffer copies. */
  readonly buffer: Buffer;

  private readonly ring: GPUReadbackRing;
  private state: GPUReadbackTicketState = 'reserved';
  private copiedByteOffset = 0;
  private copiedByteLength: number;
  private cancelled = false;

  /** @internal */
  constructor(ring: GPUReadbackRing, buffer: Buffer) {
    this.ring = ring;
    this.buffer = buffer;
    this.copiedByteLength = buffer.byteLength;
  }

  /** Records a buffer copy into this ticket without submitting the command encoder. */
  copyFrom(
    commandEncoder: CommandEncoder,
    sourceBuffer: Buffer,
    options: {sourceOffset?: number; destinationOffset?: number; byteLength?: number} = {}
  ): void {
    this.assertReserved();
    const sourceOffset = options.sourceOffset ?? 0;
    const destinationOffset = options.destinationOffset ?? 0;
    const byteLength = options.byteLength ?? sourceBuffer.byteLength - sourceOffset;
    validateCopyRange(sourceBuffer, this.buffer, sourceOffset, destinationOffset, byteLength);
    commandEncoder.copyBufferToBuffer({
      sourceBuffer,
      sourceOffset,
      destinationBuffer: this.buffer,
      destinationOffset,
      size: byteLength
    });
    this.copiedByteOffset = destinationOffset;
    this.copiedByteLength = byteLength;
    this.state = 'encoded';
  }

  /**
   * Marks a copy encoded by another abstraction, such as a command graph texture-copy node.
   */
  markEncoded(options: {byteOffset?: number; byteLength?: number} = {}): void {
    this.assertReserved();
    const byteOffset = options.byteOffset ?? 0;
    const byteLength = options.byteLength ?? this.buffer.byteLength - byteOffset;
    validateAlignedRange(this.buffer.byteLength, byteOffset, byteLength, 'readback');
    this.copiedByteOffset = byteOffset;
    this.copiedByteLength = byteLength;
    this.state = 'encoded';
  }

  /**
   * Maps and copies the encoded bytes after the application submits its command encoder.
   *
   * A ticket must be read or cancelled to return its slot; a caller that throws between `acquire()`
   * and `read()` should call `cancel()` in a `finally` block. Mapping errors, including device loss,
   * destroy and recreate the slot's staging buffer (it may be left mapped or pending) and then
   * rethrow.
   */
  async read(): Promise<Uint8Array> {
    if (this.state !== 'encoded') {
      throw new Error('GPU readback ticket must be encoded before reading');
    }
    this.state = 'reading';
    let readFailed = true;
    try {
      const data = await this.buffer.readAsync(this.copiedByteOffset, this.copiedByteLength);
      readFailed = false;
      if (this.cancelled) {
        throw new Error('GPU readback ticket was cancelled');
      }
      return data;
    } finally {
      this.release(readFailed);
    }
  }

  /**
   * Cancels an unused reservation, or discards a read already in progress.
   *
   * Cancelling an encoded ticket releases its slot immediately, which is how callers recover when
   * submit fails after encoding. If the copy was in fact submitted, queue ordering keeps any later
   * copy or mapping of the slot behind it, so the slot can be reused safely. A ticket whose `read()`
   * is in progress is discarded when the read settles.
   */
  cancel(): void {
    if (this.state === 'reserved' || this.state === 'encoded') {
      this.cancelled = true;
      this.release();
      return;
    }
    if (this.state === 'reading') {
      this.cancelled = true;
    }
  }

  private assertReserved(): void {
    if (this.state !== 'reserved') {
      throw new Error('GPU readback ticket has already been used');
    }
  }

  private release(replaceBuffer = false): void {
    if (this.state !== 'released') {
      this.state = 'released';
      this.ring.release(this.buffer, replaceBuffer);
    }
  }
}

/**
 * Fixed-capacity ring of reusable asynchronous GPU-to-CPU staging buffers.
 *
 * `tryAcquire()` makes drop-on-pressure policy explicit. `acquire()` instead waits for the next
 * completed slot. Neither path submits commands or hides queue synchronization.
 */
export class GPUReadbackRing {
  /** Capacity of every staging buffer, in bytes. */
  readonly byteLength: number;
  /** Total number of staging slots. */
  readonly slotCount: number;

  private readonly device: Device;
  private readonly id: string;
  private readonly availableBuffers: Buffer[];
  private readonly waiters: Array<{
    resolve: (ticket: GPUReadbackTicket) => void;
    reject: (error: Error) => void;
  }> = [];
  private destroyed = false;
  private replacementCount = 0;

  constructor(device: Device, props: GPUReadbackRingProps) {
    const slotCount = props.slotCount ?? 3;
    validateAlignedRange(Number.MAX_SAFE_INTEGER, 0, props.byteLength, 'ring');
    if (!Number.isSafeInteger(slotCount) || slotCount <= 0) {
      throw new Error('GPUReadbackRing slotCount must be a positive integer');
    }
    this.byteLength = props.byteLength;
    this.slotCount = slotCount;
    this.device = device;
    this.id = props.id ?? 'gpu-readback-ring';
    this.availableBuffers = Array.from({length: slotCount}, (_, slotIndex) =>
      this.createSlotBuffer(slotIndex)
    );
    // Hold the ring weakly so a dropped ring is not kept alive for the whole device lifetime.
    const weakRing = new WeakRef(this);
    void device.lost.then(() => weakRing.deref()?.destroy());
  }

  private createSlotBuffer(slotIndex: number): Buffer {
    return this.device.createBuffer({
      id: `${this.id}-slot-${slotIndex}`,
      byteLength: this.byteLength,
      usage: Buffer.COPY_DST | Buffer.MAP_READ
    });
  }

  /** Number of slots that can be reserved immediately. */
  get availableSlotCount(): number {
    return this.availableBuffers.length;
  }

  /** Reserves a slot immediately, or returns `null` when the ring is under backpressure. */
  tryAcquire(): GPUReadbackTicket | null {
    if (this.destroyed) {
      return null;
    }
    const buffer = this.availableBuffers.shift();
    return buffer ? new GPUReadbackTicket(this, buffer) : null;
  }

  /** Waits for and reserves the next available slot. */
  acquire(): Promise<GPUReadbackTicket> {
    const ticket = this.tryAcquire();
    if (ticket) {
      return Promise.resolve(ticket);
    }
    if (this.destroyed) {
      return Promise.reject(new Error('GPUReadbackRing has been destroyed'));
    }
    return new Promise((resolve, reject) => this.waiters.push({resolve, reject}));
  }

  /** Rejects pending waiters and destroys idle slots; active slots are destroyed on release. */
  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(new Error('GPUReadbackRing has been destroyed'));
    }
    for (const buffer of this.availableBuffers.splice(0)) {
      buffer.destroy();
    }
  }

  /** @internal */
  release(buffer: Buffer, replaceBuffer = false): void {
    if (this.destroyed) {
      buffer.destroy();
      return;
    }
    if (replaceBuffer) {
      // A failed map can leave the staging buffer unusable, so swap in a fresh one.
      buffer.destroy();
      buffer = this.createSlotBuffer(this.slotCount + this.replacementCount++);
    }
    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve(new GPUReadbackTicket(this, buffer));
    } else {
      this.availableBuffers.push(buffer);
    }
  }
}

function validateCopyRange(
  sourceBuffer: Buffer,
  destinationBuffer: Buffer,
  sourceOffset: number,
  destinationOffset: number,
  byteLength: number
): void {
  if (sourceBuffer.device !== destinationBuffer.device) {
    throw new Error('GPU readback source and staging buffers must belong to the same device');
  }
  validateAlignedRange(sourceBuffer.byteLength, sourceOffset, byteLength, 'source');
  validateAlignedRange(destinationBuffer.byteLength, destinationOffset, byteLength, 'destination');
}

function validateAlignedRange(
  capacity: number,
  byteOffset: number,
  byteLength: number,
  name: string
): void {
  if (
    !Number.isSafeInteger(byteOffset) ||
    !Number.isSafeInteger(byteLength) ||
    byteOffset < 0 ||
    byteLength <= 0 ||
    byteOffset % 4 !== 0 ||
    byteLength % 4 !== 0 ||
    byteOffset + byteLength > capacity
  ) {
    throw new Error(`${name} byte range must be positive, four-byte aligned, and in bounds`);
  }
}
