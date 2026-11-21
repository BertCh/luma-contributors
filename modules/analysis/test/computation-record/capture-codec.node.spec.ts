// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  serializeGPUComputationCaptures,
  deserializeGPUComputationCaptures
} from '../../src/computation-record/index';
import {validateGPUComputationSidecars} from '../../src/computation-record/validate-computation-record';
import {computationCapturesFixture, computationRecordFixture} from './computation-record-fixture';

test('UDF-04 binary capture codec preserves subarray boundaries and float bits in every category', () => {
  // IEEE-754 words: payload NaN, negative zero, 2. The guard words must not be serialized.
  const words = Uint32Array.of(99, 0x7fc12345, 0x80000000, 0x40000000, 99);
  const values = new Float32Array(words.buffer, 4, 3);
  const captures = structuredClone(computationCapturesFixture);
  const capture = {...captures.snapshots[0], components: 1, values};
  captures.snapshots = [capture];
  captures.stamps = [
    {...capture, id: 'stamp', elementType: 'u32', values: Uint32Array.of(0, 4294967295)}
  ];
  captures.replays = [
    {...capture, id: 'replay', elementType: 'i32', values: Int32Array.of(-2147483648, 2147483647)}
  ];
  const text = serializeGPUComputationCaptures(captures);
  // Little-endian bytes independently calculated from the three hexadecimal words.
  expect(JSON.parse(text).snapshots[0].values).toBe('b64:RSPBfwAAAIAAAABA');
  const restored = deserializeGPUComputationCaptures(text);
  const floats = restored.snapshots[0].values as Float32Array;
  expect(Array.from(new Uint32Array(floats.buffer, floats.byteOffset, floats.length))).toEqual([
    0x7fc12345, 0x80000000, 0x40000000
  ]);
  expect(Array.from(restored.stamps[0].values)).toEqual([0, 4294967295]);
  expect(Array.from(restored.replays[0].values)).toEqual([-2147483648, 2147483647]);
  const legacy = deserializeGPUComputationCaptures(JSON.stringify(computationCapturesFixture));
  expect(Array.from(legacy.snapshots[0].values)).toEqual([1, 2, 3, 4]);
  for (const invalid of [null, [null, 0], 'b64:AA==', 'b64:!!!!', 'b64:']) {
    const wire = JSON.parse(JSON.stringify(computationCapturesFixture));
    wire.snapshots[0].values = invalid;
    if (invalid === 'b64:') wire.snapshots[0].valueCount = 4;
    expect(() => deserializeGPUComputationCaptures(JSON.stringify(wire))).toThrow();
  }
});

test('UNI-10 rotated rectangular raster and legacy scalar geometry round trip', () => {
  const captures = structuredClone(computationCapturesFixture);
  const geometry = {
    kind: 'raster' as const,
    shape: [2, 2] as const,
    affine: [10, 3, 100, 2, -20, 200] as const,
    pixelInterpretation: 'area' as const,
    coordinateModel: 'projected' as const,
    unit: 'm',
    frameId: 'survey',
    crs: 'EPSG:3857'
  };
  captures.snapshots[0].geometry = geometry;
  const restored = deserializeGPUComputationCaptures(serializeGPUComputationCaptures(captures));
  expect(restored.snapshots[0].geometry).toEqual(geometry);
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures: restored}).valid).toBe(
    true
  );
  captures.snapshots[0].geometry = {
    kind: 'raster',
    shape: [2, 2],
    origin: [0, 40],
    cellSize: 10,
    rowDirection: 'south'
  };
  expect(
    deserializeGPUComputationCaptures(serializeGPUComputationCaptures(captures)).snapshots[0]
      .geometry
  ).toEqual(captures.snapshots[0].geometry);
  for (const change of [
    {affine: [1, 2]},
    {affine: [1, 0, NaN, 0, 1, 0]},
    {pixelInterpretation: 'unknown'},
    {frameId: ''},
    {crs: 4}
  ]) {
    Object.assign(captures.snapshots[0], {geometry: {...geometry, ...change}});
    expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
    expect(() => deserializeGPUComputationCaptures(JSON.stringify(captures))).toThrow();
  }
});

import {runInNewContext} from 'node:vm';

test('B1: a four MiB capture round trips exact length and boundary words', () => {
  const words = new Uint32Array(1024 * 1024);
  words[0] = 0x7f800001;
  words[words.length - 1] = 0x80000000;
  const captures = structuredClone(computationCapturesFixture);
  captures.snapshots[0].values = new Float32Array(words.buffer);
  const restored = deserializeGPUComputationCaptures(serializeGPUComputationCaptures(captures));
  const values = restored.snapshots[0].values as Float32Array;
  const actual = new Uint32Array(values.buffer, values.byteOffset, values.length);
  expect(actual.length).toBe(1048576);
  expect(actual[0]).toBe(0x7f800001);
  expect(actual[actual.length - 1]).toBe(0x80000000);
});

test('B2: foreign realm float subarrays retain signaling NaN bits and their byte window', () => {
  const values = runInNewContext(
    'new Float32Array(Uint32Array.of(99, 0x7f800001, 0x80000000, 99).buffer, 4, 2)'
  );
  const captures = structuredClone(computationCapturesFixture);
  captures.snapshots[0].values = values;
  const restored = deserializeGPUComputationCaptures(serializeGPUComputationCaptures(captures));
  const actual = restored.snapshots[0].values as Float32Array;
  expect(Array.from(new Uint32Array(actual.buffer, actual.byteOffset, actual.length))).toEqual([
    0x7f800001, 0x80000000
  ]);
});

test.each([
  NaN,
  Infinity,
  null,
  0,
  1.5
])('B3: invalid components %s cannot become scalar through JSON', components => {
  const captures = structuredClone(computationCapturesFixture);
  Object.assign(captures.snapshots[0], {components});
  expect(() => serializeGPUComputationCaptures(captures)).toThrow();
  expect(() => deserializeGPUComputationCaptures(JSON.stringify(captures))).toThrow();
});

test.each(['id', 'name', 'provenance'])('B3: capture %s is required on save and load', field => {
  const captures = structuredClone(computationCapturesFixture);
  delete (captures.snapshots[0] as unknown as Record<string, unknown>)[field];
  expect(() => serializeGPUComputationCaptures(captures)).toThrow();
  expect(() => deserializeGPUComputationCaptures(JSON.stringify(captures))).toThrow();
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
});

test('B3: unknown provenance cannot masquerade as readback', () => {
  const captures = structuredClone(computationCapturesFixture);
  Object.assign(captures.snapshots[0], {provenance: 'unknown'});
  expect(() => serializeGPUComputationCaptures(captures)).toThrow();
  expect(() => deserializeGPUComputationCaptures(JSON.stringify(captures))).toThrow();
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
});

test('B4: established nonfinite float tokens load, but null and integer tokens fail', () => {
  const captures = structuredClone(computationCapturesFixture);
  captures.snapshots[0].values = ['NaN', 'Infinity', '-Infinity', 2] as never;
  const actual = deserializeGPUComputationCaptures(JSON.stringify(captures)).snapshots[0].values;
  expect(Number.isNaN(actual[0])).toBe(true);
  expect(Array.from(actual).slice(1)).toEqual([Infinity, -Infinity, 2]);
  captures.snapshots[0].elementType = 'u32';
  expect(() => deserializeGPUComputationCaptures(JSON.stringify(captures))).toThrow();
});

test('S5: sparse affine tuples fail before JSON conversion', () => {
  const captures = structuredClone(computationCapturesFixture);
  captures.snapshots[0].geometry = {kind: 'raster', shape: [1, 2], affine: Array(6) as never};
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
});

test('S1: matching typed readbacks do not visit each lane during sidecar validation', () => {
  let lengthReads = 0;
  class Readback extends Float32Array {
    override get length() {
      lengthReads++;
      return this.byteLength / Float32Array.BYTES_PER_ELEMENT;
    }
  }
  const captures = structuredClone(computationCapturesFixture);
  captures.snapshots[0].values = new Readback(1024);
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(true);
  expect(lengthReads).toBeLessThanOrEqual(2);
});
