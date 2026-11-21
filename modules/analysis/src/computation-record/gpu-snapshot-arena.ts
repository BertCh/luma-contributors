// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import type {
  GPUComputationCapture,
  GPUComputationCaptureElementType
} from './computation-record-types';
import {validateOption, validateRequirement} from '../analysis-core/kernels/gpu-contributor-utils';

/** One packed, four-byte-aligned range in every snapshot frame. */
export type GPUSnapshotSlot = {
  /** Unique name within the arena, also used as the capture name. */
  name: string;
  /** Packed byte count, divisible by four. */
  byteLength: number;
  /** Typed-array interpretation of the captured words. */
  elementType: GPUComputationCaptureElementType;
  /** Number of words per logical value. */
  components?: number;
};

/** Coordinate and description of a snapshot frame. */
export type GPUSnapshotLabel = {
  /** Iteration coordinate; defaults to the frame index. */
  iteration?: number;
  /** Last graph node before this frame's copies. */
  afterNode?: string;
  /** Human-readable description, stored as capture `meaning`. */
  label?: string;
};

type GPUSnapshotReservation = {state: 'reserved' | 'submitted' | 'discarded'};

/** Reserves copied frames in one bounded buffer, then reads and caches only the retained prefix. */
export class GPUSnapshotArena {
  /** Device owning the readback buffer. */
  readonly device: Device;
  /** Prefix for capture identifiers. */
  readonly id: string;
  /** Ordered ranges in every frame. */
  readonly slots: readonly GPUSnapshotSlot[];
  /** Bytes of one snapshot frame (all slots, each four-byte aligned). */
  readonly frameByteLength: number;
  /** Maximum number of frames. */
  readonly capacity: number;

  private readonly buffer: Buffer;
  private readonly labels: GPUSnapshotLabel[] = [];
  private readback?: Promise<Uint8Array>;
  private submittedCount = 0;
  private readonly encoderReservations = new WeakMap<CommandEncoder, GPUSnapshotReservation>();
  private readonly frameReservations: GPUSnapshotReservation[] = [];
  private readonly checkpoints = new WeakMap<object, GPUSnapshotLabel | undefined>();
  private destroyed = false;
  private sealed = false;

  /** Allocates at most `maximumByteLength` bytes (64 MiB by default). */
  constructor(
    device: Device,
    props: {
      id?: string;
      slots: readonly GPUSnapshotSlot[];
      capacity: number;
      maximumByteLength?: number;
    }
  ) {
    this.device = device;
    this.id = props.id ?? 'gpu-snapshot-arena';
    this.capacity = props.capacity;
    this.slots = Object.freeze(props.slots.map(slot => Object.freeze({...slot})));
    const names = new Set<string>();
    for (const slot of this.slots) {
      validateOption(this.id, `slot "${slot.name}" elementType`, slot.elementType, [
        'f32',
        'i32',
        'u32'
      ]);
      this.validateByteRange(slot.byteLength, `slot "${slot.name}" byteLength`);
      if (names.has(slot.name)) throw new Error(`${this.id} duplicate slot "${slot.name}"`);
      names.add(slot.name);
      if (
        slot.components !== undefined &&
        (!Number.isSafeInteger(slot.components) ||
          slot.components < 1 ||
          slot.byteLength % (slot.components * 4) !== 0)
      ) {
        throw new Error(`${this.id} slot "${slot.name}" must contain whole components`);
      }
    }
    this.frameByteLength = this.slots.reduce((total, slot) => total + slot.byteLength, 0);
    const byteLength = this.capacity * this.frameByteLength;
    if (!Number.isSafeInteger(this.capacity) || this.capacity < 0) {
      throw new Error(`${this.id} capacity must be a non-negative integer`);
    }
    const maximumByteLength = props.maximumByteLength ?? 64 * 1024 * 1024;
    if (
      !Number.isSafeInteger(byteLength) ||
      !Number.isSafeInteger(maximumByteLength) ||
      maximumByteLength < 0 ||
      byteLength > maximumByteLength
    ) {
      throw new Error(`${this.id} exceeds maximumByteLength; decimate snapshot frames`);
    }
    this.buffer = device.createBuffer({
      id: this.id,
      byteLength,
      usage: Buffer.MAP_READ | Buffer.COPY_DST
    });
  }

  /** Number of reserved frames, including unsubmitted copies; discard removes an abandoned suffix. */
  get count(): number {
    return this.labels.length;
  }

  /**
   * Marks the start of an unsubmitted suffix, including reservations made by taps. Take this before
   * encoding. The returned token belongs to this arena and is invalid if its prefix is discarded.
   */
  checkpoint(): Readonly<{count: number}> {
    this.validateOpen();
    const checkpoint = Object.freeze({count: this.count});
    this.checkpoints.set(checkpoint, this.labels[this.count - 1]);
    return checkpoint;
  }

  /**
   * Protects a contiguous prefix through `checkpoint` (or all current frames when omitted). Call
   * immediately before submitting that prefix's encoders. Later reserved encoders remain discardable.
   * The boundary cannot split any encoder's reservations; submitted encoders cannot reserve again.
   * Submitted and in-flight indices are never reused. Submission failure leaves the prefix protected.
   */
  markSubmitted(checkpoint?: Readonly<{count: number}>): void {
    this.validateOpen();
    const count = checkpoint === undefined ? this.count : this.validateCheckpoint(checkpoint);
    this.validateSubmissionCount(count);
    this.validateEncoderBoundary(count);
    for (const reservation of this.frameReservations.slice(0, count))
      reservation.state = 'submitted';
    this.submittedCount = count;
  }

  /**
   * Removes only the unsubmitted suffix since `checkpoint`, without splitting any encoder's frames.
   * Every encoder containing a removed copy must be abandoned permanently: never submit or encode
   * into it later. Earlier submitted indices stay unchanged.
   */
  discard(checkpoint: Readonly<{count: number}>): void {
    this.validateOpen();
    const count = this.validateCheckpoint(checkpoint);
    this.validateSubmissionCount(count);
    this.validateEncoderBoundary(count);
    for (const reservation of this.frameReservations.slice(count)) reservation.state = 'discarded';
    this.labels.length = count;
    this.frameReservations.length = count;
  }

  /** Records every slot into the next frame and returns its zero-based index. */
  encodeSnapshot(
    commandEncoder: CommandEncoder,
    sources: Readonly<Record<string, {buffer: Buffer; byteOffset?: number}>>,
    label: GPUSnapshotLabel = {}
  ): number {
    this.validateOpen();
    if (this.count >= this.capacity) throw new Error(`${this.id} snapshot arena is full`);
    const reservation = this.encoderReservations.get(commandEncoder) ?? {
      state: 'reserved' as const
    };
    validateRequirement(
      this.id,
      reservation.state === 'reserved',
      `commandEncoder must be unsubmitted and not discarded; received ${reservation.state}`
    );
    // Capture and validate the entire frame before recording any copies or rereading caller getters.
    const frameSources = this.slots.map(slot => {
      const source = Object.hasOwn(sources, slot.name) ? sources[slot.name] : undefined;
      if (!source) throw new Error(`${this.id} missing source for slot "${slot.name}"`);
      const buffer = source.buffer;
      const byteOffset = source.byteOffset ?? 0;
      this.validateByteRange(byteOffset, `slot "${slot.name}" byteOffset`);
      if (byteOffset + slot.byteLength > buffer.byteLength) {
        throw new Error(`${this.id} source for slot "${slot.name}" is too small`);
      }
      return {buffer, byteOffset};
    });
    const frame = this.count;
    let destinationOffset = frame * this.frameByteLength;
    for (const [index, slot] of this.slots.entries()) {
      const source = frameSources[index];
      if (slot.byteLength > 0) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source.buffer,
          sourceOffset: source.byteOffset ?? 0,
          destinationBuffer: this.buffer,
          destinationOffset,
          size: slot.byteLength
        });
      }
      destinationOffset += slot.byteLength;
    }
    this.encoderReservations.set(commandEncoder, reservation);
    this.frameReservations.push(reservation);
    this.labels.push({...label});
    return frame;
  }

  /**
   * Seals the frame set at the first call; submit all retained encoders before reading. Maps only
   * `(0, count * frameByteLength)`, skips empty ranges, shares an in-flight mapping and caches
   * successful bytes. A rejected mapping may be retried, without reopening capture or discard.
   * Each returned capture owns its values. `runId` belongs to the enclosing capture sidecar.
   */
  async read(
    props: {
      runId?: string;
      graph?: string;
      resources?: Readonly<Record<string, string>>;
      keySpaces?: Readonly<Record<string, string>>;
    } = {}
  ): Promise<GPUComputationCapture[]> {
    if (this.destroyed) throw new Error(`${this.id} has been destroyed`);
    this.sealed = true;
    const byteLength = this.count * this.frameByteLength;
    this.readback ??= byteLength
      ? Promise.resolve()
          .then(() => this.buffer.readAsync(0, byteLength))
          .then(bytes => {
            validateRequirement(
              this.id,
              bytes.byteLength === byteLength,
              `readback byteLength must equal ${byteLength}; received ${bytes.byteLength}`
            );
            return new Uint8Array(bytes);
          })
          .catch(error => {
            this.readback = undefined;
            throw error;
          })
      : Promise.resolve(new Uint8Array(0));
    const bytes = await this.readback;
    const captures: GPUComputationCapture[] = [];
    for (const [frame, label] of this.labels.entries()) {
      let byteOffset = frame * this.frameByteLength;
      for (const slot of this.slots) {
        const copy = bytes.slice(byteOffset, byteOffset + slot.byteLength);
        const Values =
          slot.elementType === 'f32'
            ? Float32Array
            : slot.elementType === 'i32'
              ? Int32Array
              : Uint32Array;
        const resource =
          props.resources && Object.hasOwn(props.resources, slot.name)
            ? props.resources[slot.name]
            : undefined;
        const keySpace =
          props.keySpaces && Object.hasOwn(props.keySpaces, slot.name)
            ? props.keySpaces[slot.name]
            : undefined;
        captures.push({
          id: `${this.id}-${slot.name}-${frame}`,
          name: slot.name,
          iteration: label.iteration ?? frame,
          elementType: slot.elementType,
          values: new Values(copy.buffer, copy.byteOffset, copy.byteLength / 4),
          provenance: 'gpu-readback',
          ...(props.graph !== undefined && {graph: props.graph}),
          ...(resource !== undefined && {resource}),
          ...(keySpace !== undefined && {keySpace}),
          ...(label.afterNode !== undefined && {afterNode: label.afterNode}),
          ...(label.label !== undefined && {meaning: label.label}),
          ...(slot.components !== undefined && {components: slot.components})
        });
        byteOffset += slot.byteLength;
      }
    }
    return captures;
  }

  /** Releases the readback buffer. Already returned values remain valid. */
  destroy(): void {
    if (!this.destroyed) this.buffer.destroy();
    this.destroyed = true;
  }

  private validateCheckpoint(checkpoint: Readonly<{count: number}>): number {
    validateRequirement(
      this.id,
      this.checkpoints.has(checkpoint) &&
        checkpoint.count <= this.count &&
        this.checkpoints.get(checkpoint) === this.labels[checkpoint.count - 1],
      'checkpoint must belong to this arena and retain its original prefix'
    );
    return checkpoint.count;
  }

  private validateSubmissionCount(count: number): void {
    validateRequirement(
      this.id,
      count >= this.submittedCount,
      `checkpoint count ${count} must be at least submitted frame count ${this.submittedCount}`
    );
  }

  private validateEncoderBoundary(count: number): void {
    const prefix = new Set(this.frameReservations.slice(0, count));
    validateRequirement(
      this.id,
      this.frameReservations.slice(count).every(reservation => !prefix.has(reservation)),
      `checkpoint count ${count} must not split an encoder's reservations`
    );
  }

  private validateOpen(): void {
    if (this.destroyed || this.sealed) throw new Error(`${this.id} is closed for snapshots`);
  }

  private validateByteRange(value: number, name: string): void {
    if (!Number.isSafeInteger(value) || value < 0 || value % 4 !== 0) {
      throw new Error(`${this.id} ${name} must be a non-negative multiple of 4`);
    }
  }
}
