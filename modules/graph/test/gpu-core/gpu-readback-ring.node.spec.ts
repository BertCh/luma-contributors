// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {Device} from '@luma.gl/core';
import {GPUReadbackRing} from '../../src/gpu-core/gpu-readback-ring';

function createFakeDevice(readAsync?: () => Promise<Uint8Array>) {
  let resolveLost: (info: unknown) => void = () => {};
  const lost = new Promise(resolve => {
    resolveLost = resolve;
  });
  const destroyedBuffers: string[] = [];
  const device = {
    lost,
    createBuffer: (props: {id: string; byteLength: number}) => ({
      id: props.id,
      byteLength: props.byteLength,
      destroy: () => destroyedBuffers.push(props.id),
      readAsync: readAsync ?? (async () => new Uint8Array(props.byteLength))
    })
  } as unknown as Device;
  return {device, resolveLost, destroyedBuffers};
}

describe('GPUReadbackRing', () => {
  it('replaces the staging buffer and keeps the slot available when a read rejects', async () => {
    const {device, destroyedBuffers} = createFakeDevice(async () => {
      throw new Error('map failed');
    });
    const ring = new GPUReadbackRing(device, {byteLength: 16, slotCount: 1, id: 'ring'});
    const ticket = ring.tryAcquire()!;
    ticket.markEncoded();
    await expect(ticket.read()).rejects.toThrow('map failed');
    expect(destroyedBuffers).toEqual(['ring-slot-0']);
    expect(ring.availableSlotCount).toBe(1);
    expect(ring.tryAcquire()).not.toBeNull();
  });

  it('keeps waiters queued until a slot is cancelled', async () => {
    const {device} = createFakeDevice();
    const ring = new GPUReadbackRing(device, {byteLength: 16, slotCount: 1});
    const leaked = ring.tryAcquire()!;
    expect(ring.tryAcquire()).toBeNull();
    const waiting = ring.acquire();
    leaked.cancel();
    await expect(waiting).resolves.toBeDefined();
  });

  it('releases an encoded ticket on cancel so failed submits do not leak slots', async () => {
    const {device} = createFakeDevice();
    const ring = new GPUReadbackRing(device, {byteLength: 16, slotCount: 1});
    const ticket = ring.tryAcquire()!;
    ticket.markEncoded();
    expect(ring.availableSlotCount).toBe(0);
    ticket.cancel();
    expect(ring.availableSlotCount).toBe(1);
    // A waiter must now be satisfiable instead of hanging forever.
    await expect(ring.acquire()).resolves.toBeDefined();
  });

  it('serves a waiting acquire when an encoded ticket is cancelled', async () => {
    const {device} = createFakeDevice();
    const ring = new GPUReadbackRing(device, {byteLength: 16, slotCount: 1});
    const ticket = ring.tryAcquire()!;
    ticket.markEncoded();
    const waiting = ring.acquire();
    ticket.cancel();
    await expect(waiting).resolves.toBeDefined();
  });

  it('destroys the ring when the device is lost', async () => {
    const {device, resolveLost, destroyedBuffers} = createFakeDevice();
    const ring = new GPUReadbackRing(device, {byteLength: 16, slotCount: 2});
    resolveLost({});
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(ring.tryAcquire()).toBeNull();
    expect(destroyedBuffers).toHaveLength(2);
  });

  it('does not retain the ring through the device lost handler', async () => {
    const {device} = createFakeDevice();
    let ring: GPUReadbackRing | null = new GPUReadbackRing(device, {byteLength: 16});
    const weakRing = new WeakRef(ring);
    ring = null;
    const gc = (globalThis as {gc?: () => void}).gc;
    if (!gc) {
      return; // Collection can only be forced when Node runs with --expose-gc.
    }
    await new Promise(resolve => setTimeout(resolve, 0));
    gc();
    expect(weakRing.deref()).toBeUndefined();
  });
});
