// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {vi} from 'vitest';
import {
  GPUSnapshotArena,
  type GPUSnapshotSlot
} from '../../src/computation-record/gpu-snapshot-arena';

/** Copies become visible only on submit, like commands in a real encoder. */
export function createRecordingSnapshotFixture(slots: readonly GPUSnapshotSlot[], capacity = 4) {
  let storage = new Uint8Array(0);
  const readAsync = vi.fn(async (byteOffset = 0, byteLength = storage.byteLength) =>
    storage.slice(byteOffset, byteOffset + byteLength)
  );
  const device = {
    createBuffer: ({byteLength}: {byteLength: number}) => {
      storage = new Uint8Array(byteLength);
      return {byteLength, readAsync, destroy: vi.fn()};
    }
  } as unknown as Device;
  const arena = new GPUSnapshotArena(device, {id: 'reserved', slots, capacity});
  const createSource = (values: number[]) => {
    const data = new Uint8Array(new Uint32Array(values).buffer);
    return {byteLength: data.byteLength, data} as unknown as Buffer;
  };
  const createEncoder = () => {
    const copies: (() => void)[] = [];
    const encoder = {
      copyBufferToBuffer: (props: {
        sourceBuffer: Buffer & {data: Uint8Array};
        sourceOffset: number;
        destinationOffset: number;
        size: number;
      }) =>
        copies.push(() =>
          storage.set(
            props.sourceBuffer.data.subarray(props.sourceOffset, props.sourceOffset + props.size),
            props.destinationOffset
          )
        )
    } as unknown as CommandEncoder;
    return {encoder, submit: () => copies.forEach(copy => copy())};
  };
  return {arena, device, createSource, createEncoder, readAsync};
}
