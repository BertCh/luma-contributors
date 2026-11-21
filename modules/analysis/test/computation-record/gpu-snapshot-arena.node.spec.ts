// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer as NodeBuffer} from 'node:buffer';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {describe, expect, test, vi} from 'vitest';
import {
  GPUSnapshotArena,
  type GPUSnapshotSlot
} from '../../src/computation-record/gpu-snapshot-arena';

import {createRecordingSnapshotFixture} from './snapshot-recording-fixture';

function createFixture(slots: readonly GPUSnapshotSlot[], capacity = 2) {
  let storage = new Uint8Array(0);
  const readAsync = vi.fn(async (byteOffset = 0, byteLength = storage.byteLength) =>
    storage.slice(byteOffset, byteOffset + byteLength)
  );
  const destroy = vi.fn();
  const createBuffer = vi.fn((props: {byteLength: number}) => {
    storage = new Uint8Array(props.byteLength);
    return {byteLength: props.byteLength, readAsync, destroy};
  });
  const device = {createBuffer} as unknown as Device;
  const arena = new GPUSnapshotArena(device, {id: 'history', slots, capacity});
  const copyBufferToBuffer = vi.fn(
    (props: {
      sourceBuffer: Buffer & {data?: Uint8Array};
      sourceOffset: number;
      destinationOffset: number;
      size: number;
    }) => {
      storage.set(
        props.sourceBuffer.data?.subarray(props.sourceOffset, props.sourceOffset + props.size) ??
          new Uint8Array(props.size).fill(props.sourceOffset),
        props.destinationOffset
      );
    }
  );
  const encoder = {copyBufferToBuffer} as unknown as CommandEncoder;
  const source = {byteLength: 64} as Buffer;
  return {arena, encoder, source, createBuffer, copyBufferToBuffer, readAsync, destroy, device};
}

describe('GPUSnapshotArena', () => {
  test('LIF-05: captured words retain their names after attempted slot reordering', async () => {
    const fixture = createFixture([
      {name: 'x', byteLength: 4, elementType: 'u32'},
      {name: 'y', byteLength: 4, elementType: 'u32'}
    ]);
    const source = {
      byteLength: 8,
      data: new Uint8Array(new Uint32Array([7, 9]).buffer)
    } as unknown as Buffer;
    fixture.arena.encodeSnapshot(fixture.encoder, {
      x: {buffer: source},
      y: {buffer: source, byteOffset: 4}
    });
    expect(() => (fixture.arena.slots as GPUSnapshotSlot[]).reverse()).toThrow();
    expect(Object.isFrozen(fixture.arena.slots[0])).toBe(true);
    // Hand fixture: x is the first word (7), y is the second (9).
    expect(
      (await fixture.arena.read()).map(capture => [capture.name, Array.from(capture.values)])
    ).toEqual([
      ['x', [7]],
      ['y', [9]]
    ]);
    fixture.arena.destroy();
  });

  for (const name of ['toString', 'constructor', 'missing']) {
    test(`EGR-07: ${name} requires own sources and optional metadata`, async () => {
      const {arena, encoder, source, copyBufferToBuffer} = createFixture([
        {name, byteLength: 4, elementType: 'u32'}
      ]);
      expect(() => arena.encodeSnapshot(encoder, {})).toThrow(
        `history missing source for slot "${name}"`
      );
      const inheritedSources = Object.create({[name]: {buffer: source}});
      expect(() => arena.encodeSnapshot(encoder, inheritedSources)).toThrow('missing source');
      const inheritedGetter = Object.create(
        Object.defineProperty({}, name, {
          get: () => {
            throw new Error('inherited getter must not run');
          }
        })
      );
      expect(() => arena.encodeSnapshot(encoder, inheritedGetter)).toThrow('missing source');
      expect(copyBufferToBuffer).not.toHaveBeenCalled();
      arena.encodeSnapshot(encoder, {[name]: {buffer: source}});
      const [capture] = await arena.read({resources: {}, keySpaces: {}});
      expect(capture).not.toHaveProperty('resource');
      expect(capture).not.toHaveProperty('keySpace');
      const [inherited] = await arena.read({
        resources: Object.create({[name]: 'inherited-resource'}),
        keySpaces: Object.create({[name]: 'inherited-keys'})
      });
      expect(inherited).not.toHaveProperty('resource');
      expect(inherited).not.toHaveProperty('keySpace');
      const [own] = await arena.read({
        resources: {[name]: 'own-resource'},
        keySpaces: {[name]: 'own-keys'}
      });
      expect(own).toMatchObject({name, resource: 'own-resource', keySpace: 'own-keys'});
      arena.destroy();
    });
  }

  test('LIF-13: discards only an unsubmitted suffix and preserves submitted frame numbers', async () => {
    const {arena, createSource, createEncoder} = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const initial = arena.checkpoint();
    expect(Object.isFrozen(initial)).toBe(true);
    expect(() => arena.discard({count: 0})).toThrow('checkpoint');
    const first = createEncoder();
    expect(
      arena.encodeSnapshot(first.encoder, {value: {buffer: createSource([5])}}, {label: 'first'})
    ).toBe(0);
    arena.markSubmitted(); // Protect before submission: in-flight frames cannot be reused either.
    expect(() => arena.discard(initial)).toThrow('submitted');
    first.submit();
    const boundary = arena.checkpoint();
    const abandoned = createEncoder();
    arena.encodeSnapshot(
      abandoned.encoder,
      {value: {buffer: createSource([7])}},
      {label: 'discard'}
    );
    const stale = arena.checkpoint();
    arena.discard(boundary); // Never submit abandoned.encoder after this point.
    expect(arena.count).toBe(1);
    const next = createEncoder();
    expect(
      arena.encodeSnapshot(next.encoder, {value: {buffer: createSource([9])}}, {label: 'submit'})
    ).toBe(1);
    expect(() => arena.discard(stale)).toThrow('checkpoint');
    arena.markSubmitted();
    next.submit();
    expect(
      (await arena.read()).map(capture => [
        capture.iteration,
        capture.meaning,
        Array.from(capture.values)
      ])
    ).toEqual([
      [0, 'first', [5]],
      [1, 'submit', [9]]
    ]);
    expect(() => arena.discard(boundary)).toThrow('closed');
    arena.destroy();
  });

  test('EGR-09/EGR-10/LIF-15: concurrent reads map only the live prefix and own cached bytes', async () => {
    const {arena, createSource, createEncoder, readAsync} = createRecordingSnapshotFixture(
      [{name: 'value', byteLength: 4, elementType: 'u32'}],
      1024
    );
    const encoded = createEncoder();
    arena.encodeSnapshot(encoded.encoder, {value: {buffer: createSource([7])}});
    encoded.submit();
    const [first, second] = await Promise.all([arena.read(), arena.read()]);
    expect(readAsync).toHaveBeenCalledTimes(1);
    expect(readAsync).toHaveBeenCalledWith(0, 4); // One uint32 word, independent of 1024-frame capacity.
    expect(Array.from(first[0].values)).toEqual([7]);
    (first[0].values as Uint32Array)[0] = 99;
    expect(Array.from(second[0].values)).toEqual([7]);
    expect(Array.from((await arena.read())[0].values)).toEqual([7]);
    expect(readAsync).toHaveBeenCalledTimes(1);
    arena.destroy();
  });

  test('EGR-09/LIF-15: failed shared mapping permits retry while the frame set stays sealed', async () => {
    const {arena, createSource, createEncoder, readAsync} = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const boundary = arena.checkpoint();
    const encoded = createEncoder();
    arena.encodeSnapshot(encoded.encoder, {value: {buffer: createSource([9])}});
    encoded.submit();
    let rejectMap!: (reason: Error) => void;
    readAsync.mockImplementationOnce(
      () =>
        new Promise<never>((_resolve, reject) => {
          rejectMap = reject;
        })
    );
    const failed = Promise.allSettled([arena.read(), arena.read()]);
    await Promise.resolve();
    expect(readAsync).toHaveBeenCalledTimes(1);
    rejectMap(new Error('temporary mapping failure'));
    const outcomes = await failed;
    expect(outcomes.map(outcome => outcome.status)).toEqual(['rejected', 'rejected']);
    expect(() =>
      arena.encodeSnapshot(encoded.encoder, {value: {buffer: createSource([0])}})
    ).toThrow('closed');
    expect(() => arena.discard(boundary)).toThrow('closed');
    expect(() => arena.checkpoint()).toThrow('closed');
    const [first, second] = await Promise.all([arena.read(), arena.read()]);
    expect(readAsync).toHaveBeenCalledTimes(2);
    expect(Array.from(first[0].values)).toEqual([9]);
    expect(Array.from(second[0].values)).toEqual([9]);
    await arena.read();
    expect(readAsync).toHaveBeenCalledTimes(2);
    arena.destroy();
  });

  test('EGR-09: synchronous mapping failures and truncated bytes can be retried', async () => {
    const {arena, createSource, createEncoder, readAsync} = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const encoded = createEncoder();
    arena.encodeSnapshot(encoded.encoder, {value: {buffer: createSource([11])}});
    encoded.submit();
    readAsync.mockImplementationOnce(() => {
      throw new Error('synchronous mapping failure');
    });
    await expect(arena.read()).rejects.toThrow('synchronous mapping failure');
    readAsync.mockResolvedValueOnce(new Uint8Array(0));
    await expect(arena.read()).rejects.toThrow('readback byteLength');
    expect(Array.from((await arena.read())[0].values)).toEqual([11]);
    expect(readAsync).toHaveBeenCalledTimes(3);
    arena.destroy();
  });

  test('review B3: marks an encoder prefix while a later unsubmitted suffix is reserved', async () => {
    const {arena, createEncoder, createSource} = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const first = createEncoder();
    arena.encodeSnapshot(first.encoder, {value: {buffer: createSource([53])}}, {label: 'A'});
    const throughFirst = arena.checkpoint();
    const later = createEncoder();
    arena.encodeSnapshot(later.encoder, {value: {buffer: createSource([79])}}, {label: 'B'});
    arena.markSubmitted(throughFirst);
    first.submit();
    arena.discard(throughFirst);
    const replacement = createEncoder();
    expect(
      arena.encodeSnapshot(
        replacement.encoder,
        {value: {buffer: createSource([97])}},
        {label: 'replacement'}
      )
    ).toBe(1);
    arena.markSubmitted();
    replacement.submit();
    expect(
      (await arena.read()).map(capture => [
        capture.iteration,
        capture.meaning,
        Array.from(capture.values)
      ])
    ).toEqual([
      [0, 'A', [53]],
      [1, 'replacement', [97]]
    ]);
    arena.destroy();
  });

  test('review B3: neither discard nor submission may split one encoder reservations', async () => {
    const {arena, createEncoder, createSource} = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const start = arena.checkpoint();
    const abandoned = createEncoder();
    arena.encodeSnapshot(
      abandoned.encoder,
      {value: {buffer: createSource([31])}},
      {label: 'first-batch'}
    );
    const split = arena.checkpoint();
    arena.encodeSnapshot(
      abandoned.encoder,
      {value: {buffer: createSource([41])}},
      {label: 'second-batch'}
    );
    expect(() => arena.discard(split)).toThrow('encoder');
    expect(() => arena.markSubmitted(split)).toThrow('encoder');
    expect(arena.count).toBe(2);
    arena.discard(start);
    expect(() =>
      arena.encodeSnapshot(abandoned.encoder, {value: {buffer: createSource([61])}})
    ).toThrow('discarded');
    const valid = createEncoder();
    arena.encodeSnapshot(valid.encoder, {value: {buffer: createSource([83])}}, {label: 'valid'});
    arena.markSubmitted();
    expect(() =>
      arena.encodeSnapshot(valid.encoder, {value: {buffer: createSource([89])}})
    ).toThrow('submitted');
    valid.submit();
    expect(
      (await arena.read()).map(capture => [capture.meaning, Array.from(capture.values)])
    ).toEqual([['valid', [83]]]);
    arena.destroy();
  });

  test('review B6: owns a Node Buffer subview across concurrent and cached readers', async () => {
    const fixture = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const encoded = fixture.createEncoder();
    fixture.arena.encodeSnapshot(encoded.encoder, {value: {buffer: fixture.createSource([73])}});
    encoded.submit();
    const backing = NodeBuffer.alloc(12);
    backing.writeUInt32LE(999, 0);
    backing.writeUInt32LE(73, 4);
    backing.writeUInt32LE(888, 8);
    const mapped = backing.subarray(4, 8);
    fixture.readAsync.mockResolvedValueOnce(mapped);
    const [first, second] = await Promise.all([fixture.arena.read(), fixture.arena.read()]);
    (first[0].values as Uint32Array)[0] = 666;
    expect(Array.from(second[0].values)).toEqual([73]);
    mapped.writeUInt32LE(777);
    expect(Array.from((await fixture.arena.read())[0].values)).toEqual([73]);
    expect(backing.readUInt32LE(0)).toBe(999);
    expect(backing.readUInt32LE(8)).toBe(888);
    fixture.arena.destroy();
  });

  test('review S1: copies the validated source and offset only once', async () => {
    const fixture = createRecordingSnapshotFixture([
      {name: 'value', byteLength: 4, elementType: 'u32'}
    ]);
    const encoded = fixture.createEncoder();
    const sources = Object.create({value: {buffer: fixture.createSource([92])}});
    let bufferReads = 0;
    let offsetReads = 0;
    const source = {
      get buffer() {
        bufferReads++;
        return fixture.createSource([0, 81]);
      },
      get byteOffset() {
        offsetReads++;
        return 4;
      }
    };
    Object.defineProperty(sources, 'value', {
      configurable: true,
      get: () => {
        delete sources.value;
        return source;
      }
    });
    fixture.arena.encodeSnapshot(encoded.encoder, sources);
    encoded.submit();
    expect((await fixture.arena.read()).map(capture => Array.from(capture.values))).toEqual([[81]]);
    expect(bufferReads).toBe(1);
    expect(offsetReads).toBe(1);
    fixture.arena.destroy();
  });

  test('review earlier selector: rejects unsupported slot element types before allocation', () => {
    const createBuffer = vi.fn();
    const device = {createBuffer} as unknown as Device;
    expect(
      () =>
        new GPUSnapshotArena(device, {
          id: 'slot-format',
          capacity: 1,
          slots: [{name: 'value', byteLength: 4, elementType: 's32' as never}]
        })
    ).toThrow('slot-format slot "value" elementType must be f32 or i32 or u32; received s32');
    expect(createBuffer).not.toHaveBeenCalled();
  });

  test('lays out frames, maps once, and returns independent typed copies and metadata', async () => {
    const fixture = createFixture([
      {name: 'position', byteLength: 8, elementType: 'f32', components: 2},
      {name: 'mask', byteLength: 4, elementType: 'u32'},
      {name: 'signed', byteLength: 4, elementType: 'i32'}
    ]);
    const {arena, encoder, source} = fixture;
    expect(arena.frameByteLength).toBe(16);
    expect(fixture.createBuffer).toHaveBeenCalledWith({
      id: 'history',
      byteLength: 32,
      usage: Buffer.MAP_READ | Buffer.COPY_DST
    });
    const sources = {
      position: {buffer: source},
      mask: {buffer: source, byteOffset: 4},
      signed: {buffer: source}
    };
    expect(
      arena.encodeSnapshot(encoder, sources, {iteration: 7, afterNode: 'advance', label: 'ready'})
    ).toBe(0);
    expect(arena.encodeSnapshot(encoder, sources)).toBe(1);
    expect(fixture.copyBufferToBuffer.mock.calls.map(([copy]) => copy.destinationOffset)).toEqual([
      0, 8, 12, 16, 24, 28
    ]);
    const captures = await arena.read({
      graph: 'compiled',
      resources: {mask: 'mask-buffer'},
      keySpaces: {mask: 'rows'}
    });
    expect(captures).toHaveLength(6);
    expect(captures[0].values).toBeInstanceOf(Float32Array);
    expect(captures[1].values).toBeInstanceOf(Uint32Array);
    expect(captures[2].values).toBeInstanceOf(Int32Array);
    expect(captures[1]).toMatchObject({
      id: 'history-mask-0',
      name: 'mask',
      iteration: 7,
      afterNode: 'advance',
      meaning: 'ready',
      graph: 'compiled',
      resource: 'mask-buffer',
      keySpace: 'rows',
      provenance: 'gpu-readback'
    });
    expect(captures[3].iteration).toBe(1);
    expect(Object.values(captures[3])).not.toContain(undefined);
    (captures[1].values as Uint32Array)[0] = 0;
    const repeated = await arena.read();
    expect(repeated[1].values[0]).toBe(0x04040404);
    expect(fixture.readAsync).toHaveBeenCalledTimes(1);
    expect(() => arena.encodeSnapshot(encoder, sources)).toThrow('history');
    arena.destroy();
    arena.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
    expect(repeated[1].values[0]).toBe(0x04040404);
    await expect(arena.read()).rejects.toThrow('history');
  });

  test('rejects missing, misaligned and undersized sources before copying; rejects full arenas', () => {
    const {arena, encoder, source, copyBufferToBuffer} = createFixture(
      [
        {name: 'first', byteLength: 4, elementType: 'u32'},
        {name: 'second', byteLength: 4, elementType: 'u32'}
      ],
      1
    );
    expect(() => arena.encodeSnapshot(encoder, {first: {buffer: source}})).toThrow(
      'history missing source'
    );
    expect(() =>
      arena.encodeSnapshot(encoder, {
        first: {buffer: source},
        second: {buffer: source, byteOffset: 2}
      })
    ).toThrow('multiple of 4');
    expect(() =>
      arena.encodeSnapshot(encoder, {
        first: {buffer: source},
        second: {buffer: source, byteOffset: 64}
      })
    ).toThrow('too small');
    expect(copyBufferToBuffer).not.toHaveBeenCalled();
    expect(arena.count).toBe(0);
    const sources = {first: {buffer: source}, second: {buffer: source}};
    arena.encodeSnapshot(encoder, sources);
    expect(() => arena.encodeSnapshot(encoder, sources)).toThrow('history snapshot arena is full');
    arena.destroy();
  });

  test('validates slot alignment, capacity, component counts and the byte budget before allocation', () => {
    const createBuffer = vi.fn();
    const device = {createBuffer} as unknown as Device;
    const slot: GPUSnapshotSlot = {name: 'value', byteLength: 4, elementType: 'u32'};
    const make = (overrides: Partial<ConstructorParameters<typeof GPUSnapshotArena>[1]>) =>
      new GPUSnapshotArena(device, {id: 'bounded', slots: [slot], capacity: 1, ...overrides});
    expect(() => make({slots: [{...slot, byteLength: 3}]})).toThrow('bounded');
    expect(() => make({slots: [{...slot, byteLength: -4}]})).toThrow('bounded');
    expect(() => make({slots: [{...slot, components: 2}]})).toThrow('whole components');
    expect(() => make({slots: [slot, slot]})).toThrow('duplicate slot');
    expect(() => make({capacity: -1})).toThrow('capacity');
    expect(() => make({capacity: 1.5})).toThrow('capacity');
    expect(() => make({maximumByteLength: 3})).toThrow('bounded exceeds maximumByteLength');
    expect(() => make({capacity: 16 * 1024 * 1024 + 1})).toThrow('maximumByteLength');
    expect(createBuffer).not.toHaveBeenCalled();
  });

  test('empty frames and empty executions need no map', async () => {
    const {arena, encoder, source, readAsync} = createFixture([
      {name: 'empty', byteLength: 0, elementType: 'u32'}
    ]);
    arena.encodeSnapshot(encoder, {empty: {buffer: source}});
    expect((await arena.read())[0].values.length).toBe(0);
    expect(readAsync).not.toHaveBeenCalled();
    arena.destroy();
    const empty = createFixture([], 0);
    expect(await empty.arena.read()).toEqual([]);
    expect(empty.readAsync).not.toHaveBeenCalled();
    empty.arena.destroy();
  });
});
