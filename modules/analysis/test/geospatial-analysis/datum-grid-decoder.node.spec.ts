// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {runInNewContext} from 'node:vm';
import {
  decodeGPUDatumGridTransformPositions,
  decodeGPUDatumGridTransformStatus
} from '../../src/geospatial-analysis';

it('EGR-14 reconstructs longitude residuals in binary64 and preserves row statuses', () => {
  const positions = Float32Array.of(120, 40, 3);
  const positionsLow = Float32Array.of(0.000001, 0, 0);
  // 1e-6 rounded to binary32 is 0.0000009999999974752427; binary64 sum rounds to this literal.
  expect(positions.map((value, index) => value + positionsLow[index])[0]).toBe(120);
  const decoded = decodeGPUDatumGridTransformPositions({
    positions,
    positionsLow,
    status: Uint32Array.of(0)
  });
  expect(decoded.positions).toBeInstanceOf(Float64Array);
  expect([...decoded.positions]).toEqual([120.000001, 40, 3]);
  expect([...decoded.status]).toEqual([0]);
  expect(
    decodeGPUDatumGridTransformPositions({positions, status: Uint32Array.of(4)}).status[0]
  ).toBe(4);
});

it('EGR-14 honors byte windows and checks row and chunk alignment', () => {
  const bytes = new Uint8Array(20);
  bytes.set(new Uint8Array(Float32Array.of(120, 40, 3).buffer), 4);
  const statusBytes = new Uint8Array(12);
  statusBytes.set(new Uint8Array(Uint32Array.of(8).buffer), 4);
  const output = decodeGPUDatumGridTransformPositions({
    positions: bytes.subarray(4, 16),
    status: statusBytes.subarray(4, 8)
  });
  expect([...output.positions]).toEqual([120, 40, 3]);
  expect([...output.status]).toEqual([8]);
  const chunked = decodeGPUDatumGridTransformPositions({
    positions: [Float32Array.of(120, 40, 3), Float32Array.of(-70, 20, 5)],
    positionsLow: [Float32Array.of(0.000001, 0, 0), Float32Array.of(0, 0, 0.5)],
    status: [Uint32Array.of(0), Uint32Array.of(4)]
  });
  expect([...chunked.positions]).toEqual([120.000001, 40, 3, -70, 20, 5.5]);
  expect([...chunked.status]).toEqual([0, 4]);
  for (const positions of [new Uint8Array(11), new Float32Array(2)]) {
    expect(() =>
      decodeGPUDatumGridTransformPositions({positions, status: Uint32Array.of(0)})
    ).toThrow();
  }
  expect(() =>
    decodeGPUDatumGridTransformPositions({
      positions: new Float32Array(3),
      status: new Uint32Array()
    })
  ).toThrow();
  expect(() =>
    decodeGPUDatumGridTransformPositions({
      positions: [new Float32Array(3), new Float32Array(3)],
      positionsLow: [new Float32Array(6)],
      status: [Uint32Array.of(0), Uint32Array.of(0)]
    })
  ).toThrow();
});

it('EGR-14 decodes named status flags and rejects missing status words', () => {
  expect(decodeGPUDatumGridTransformStatus(Uint32Array.of(0))).toEqual({
    complete: true,
    invalidInput: false,
    noCoverage: false,
    nodata: false,
    iterationLimit: false
  });
  for (const [code, field] of [
    [1, 'invalidInput'],
    [2, 'noCoverage'],
    [4, 'nodata'],
    [8, 'iterationLimit']
  ] as const) {
    expect(decodeGPUDatumGridTransformStatus(Uint32Array.of(code).buffer)).toEqual({
      complete: false,
      invalidInput: false,
      noCoverage: false,
      nodata: false,
      iterationLimit: false,
      [field]: true
    });
  }
  expect(() => decodeGPUDatumGridTransformStatus(new Uint8Array(3))).toThrow();
  expect(() => decodeGPUDatumGridTransformStatus(new Uint32Array())).toThrow();
  const unaligned = new Uint8Array(13);
  unaligned.set(new Uint8Array(Float32Array.of(120, 40, 3).buffer), 1);
  expect([
    ...decodeGPUDatumGridTransformPositions({
      positions: unaligned.subarray(1),
      status: Uint32Array.of(0)
    }).positions
  ]).toEqual([120, 40, 3]);
});

it('B3 accepts foreign realm ArrayBuffer readbacks through both public decoders', () => {
  expect(decodeGPUDatumGridTransformStatus(runInNewContext('Uint32Array.of(2).buffer'))).toEqual({
    complete: false,
    invalidInput: false,
    noCoverage: true,
    nodata: false,
    iterationLimit: false
  });
  const decoded = decodeGPUDatumGridTransformPositions({
    positions: runInNewContext('Float32Array.of(120,40,3).buffer'),
    positionsLow: runInNewContext('Float32Array.of(0.000001,0,0).buffer'),
    status: runInNewContext('Uint32Array.of(0).buffer')
  });
  expect([...decoded.positions]).toEqual([120.000001, 40, 3]);
  expect([...decoded.status]).toEqual([0]);
});
