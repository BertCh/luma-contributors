// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {diagnoseFloat32Narrowing} from '../../../examples/deck/kepler-compute/src/analysis-numeric-diagnostics';
import {aggregateInteger64OnGpu} from '@luma-contributors/analysis/dataframe-analysis';
import {scanWideNumericColumnOnGpu} from '@luma-contributors/analysis/dataframe-analysis';
import {probeWideNumericHelpersOnGpu} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-probe';
import {
  addWords,
  convertWordsToFloat32,
  countLeadingZeroWords,
  divideSignedWordsBySmall,
  divideWordsBySmall,
  isSignedLess,
  isUnsignedLess,
  multiplyWords128,
  multiplyWordsLow,
  negateWords,
  shiftWordsLeft,
  shiftWordsRight,
  splitBigIntToWords,
  splitDoubleToWords,
  subtractWords,
  summarizeInteger64Reference,
  summarizeNarrowingReference
} from '@luma-contributors/analysis/dataframe-analysis';
import {makeWideNumericTestValues} from './wide-numeric-test-values';

function makeWordVector(device: Device, chunks: readonly (readonly number[][])[]) {
  return new GPUVector({
    name: 'words',
    type: 'data',
    format: 'uint32x2',
    ownsData: true,
    data: chunks.map(
      chunk =>
        new GPUData({
          format: 'uint32x2',
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: Uint32Array.from(chunk.length ? chunk.flat() : [0, 0]),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
}

function makeValidityVector(device: Device, chunks: readonly (readonly boolean[])[]) {
  return new GPUVector({
    name: 'validity',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: chunks.map(
      chunk =>
        new GPUData({
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: Uint32Array.from(chunk.length ? chunk.map(valid => (valid ? 1 : 0)) : [0]),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
}

function splitIntoChunks<T>(values: readonly T[], sizes: readonly number[]): T[][] {
  const chunks: T[][] = [];
  let offset = 0;
  for (const size of sizes) {
    chunks.push(values.slice(offset, offset + size));
    offset += size;
  }
  chunks.push(values.slice(offset));
  return chunks;
}

describe('wide numeric GPU kernels', () => {
  it('WGSL helpers equal the TypeScript emulation word for word', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values = makeWideNumericTestValues();
    const cases = values.map((value, index) => [value, values[(index * 7 + 3) % values.length]]);
    const rows = cases.flatMap(([a, b]) => [splitBigIntToWords(a), splitBigIntToWords(b)]);
    const words = makeWordVector(device, [rows]);
    try {
      const results = await probeWideNumericHelpersOnGpu({device, words, caseCount: cases.length});
      for (const [index, [a, b]] of cases.entries()) {
        const pa = splitBigIntToWords(a);
        const pb = splitBigIntToWords(b);
        const result = results[index];
        const label = `${a} ${b}`;
        expect(result.addition, label).toEqual(addWords(pa, pb));
        expect(result.subtraction, label).toEqual(subtractWords(pa, pb));
        expect(result.negation, label).toEqual(negateWords(pa));
        expect(result.lessUnsigned, label).toBe(isUnsignedLess(pa, pb));
        expect(result.lessSigned, label).toBe(isSignedLess(pa, pb));
        expect(result.multiplyLow, label).toEqual(multiplyWordsLow(pa, pb));
        expect(result.multiply128, label).toEqual(multiplyWords128(pa, pb));
        const divided = divideWordsBySmall(pa, pb[0]);
        expect(result.divideUnsigned, label).toEqual(divided);
        const dividedSigned = divideSignedWordsBySmall(pa, pb[0]);
        expect(result.divideSigned, label).toEqual(dividedSigned);
        const unsignedFloat = convertWordsToFloat32(pa, false);
        expect(new Float32Array(Uint32Array.of(result.unsignedFloat32.bits).buffer)[0], label).toBe(
          unsignedFloat.value
        );
        expect(result.unsignedFloat32.exact, label).toBe(unsignedFloat.exact);
        const signedFloat = convertWordsToFloat32(pa, true);
        expect(new Float32Array(Uint32Array.of(result.signedFloat32.bits).buffer)[0], label).toBe(
          signedFloat.value
        );
        expect(result.signedFloat32.exact, label).toBe(signedFloat.exact);
        expect(result.leadingZeros, label).toBe(countLeadingZeroWords(pa));
        expect(result.shiftRight, label).toEqual(shiftWordsRight(pa, pb[0] & 63));
        expect(result.shiftLeft, label).toEqual(shiftWordsLeft(pa, pb[0] & 63));
      }
    } finally {
      words.destroy();
    }
  });

  for (const signed of [true, false])
    it(`exact ${signed ? 'Int64' : 'Uint64'} sum/min/max/count/nullCount across chunks equals BigInt`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const raw = makeWideNumericTestValues();
      const values: (bigint | null)[] = raw.map((value, index) =>
        index % 11 === 5 ? null : signed ? BigInt.asIntN(64, value) : BigInt.asUintN(64, value)
      );
      const sizes = [0, 3, 1, 40, 0, 77];
      const valueChunks = splitIntoChunks(values, sizes);
      const words = makeWordVector(
        device,
        valueChunks.map(chunk => chunk.map(value => splitBigIntToWords(value ?? 0n)))
      );
      const validity = makeValidityVector(
        device,
        valueChunks.map(chunk => chunk.map(value => value !== null))
      );
      try {
        const actual = await aggregateInteger64OnGpu({device, words, signed, validity});
        expect(actual).toEqual(summarizeInteger64Reference(values, signed));
        // Without validity every row is valid (nulls become zero words here).
        const allValid = await aggregateInteger64OnGpu({device, words, signed});
        expect(allValid.count).toBe(values.length);
        expect(allValid.nullCount).toBe(0);
        const empty = makeWordVector(device, [[]]);
        const emptyValidity = makeValidityVector(device, [[]]);
        try {
          expect(
            await aggregateInteger64OnGpu({device, words: empty, signed, validity: emptyValidity})
          ).toEqual({count: 0, nullCount: 0, minimum: null, maximum: null, sum: 0n});
        } finally {
          empty.destroy();
          emptyValidity.destroy();
        }
      } finally {
        words.destroy();
        validity.destroy();
      }
    });

  it('sums many maximal values with carries beyond 64 bits', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rowCount = 70000;
    const words = makeWordVector(device, [
      Array.from({length: rowCount}, () => splitBigIntToWords((1n << 64n) - 1n))
    ]);
    try {
      const unsigned = await aggregateInteger64OnGpu({device, words, signed: false});
      expect(unsigned.sum).toBe(((1n << 64n) - 1n) * BigInt(rowCount));
      const signed = await aggregateInteger64OnGpu({device, words, signed: true});
      expect(signed.sum).toBe(-BigInt(rowCount));
      expect(signed.minimum).toBe(-1n);
    } finally {
      words.destroy();
    }
  });

  for (const format of ['int64', 'uint64'] as const)
    it(`${format} narrowing scan equals the BigInt reference`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const values: (bigint | null)[] = makeWideNumericTestValues().map((value, index) =>
        index % 13 === 4
          ? null
          : format === 'int64'
            ? BigInt.asIntN(64, value)
            : BigInt.asUintN(64, value)
      );
      const chunks = splitIntoChunks(values, [10, 0, 90]);
      const words = makeWordVector(
        device,
        chunks.map(chunk => chunk.map(value => splitBigIntToWords(value ?? 0n)))
      );
      const validity = makeValidityVector(
        device,
        chunks.map(chunk => chunk.map(value => value !== null))
      );
      try {
        const actual = await scanWideNumericColumnOnGpu({device, words, format, validity});
        const expected = summarizeNarrowingReference(format, values);
        expect(actual).toMatchObject({
          format,
          rowCount: expected.rowCount,
          nullCount: expected.nullCount,
          inexactCount: expected.inexactCount,
          overflowCount: 0,
          nonFiniteCount: 0,
          verdict: expected.verdict
        });
        expect(actual.maximumAbsoluteError).toBe(expected.maximumAbsoluteError);
      } finally {
        words.destroy();
        validity.destroy();
      }
    });

  it('Float64 narrowing scan equals diagnoseFloat32Narrowing for exact, lossy, overflow and non-finite rows', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const doubles: (number | null)[] = [
      0,
      1,
      0.5,
      16777216,
      0.1,
      1 / 3,
      1e10,
      3.4028234663852886e38,
      3.4028235677973366e38,
      1e300,
      -1e300,
      1e-40,
      7e-46,
      5e-324,
      Infinity,
      -Infinity,
      NaN,
      null,
      Math.PI,
      Math.fround(Math.PI),
      9007199254740993,
      -123456.789
    ];
    let state = 99;
    for (let index = 0; index < 400; index++) {
      state = (Math.imul(state, 1103515245) + 12345) >>> 0;
      doubles.push((state / 0xffffffff - 0.5) * 10 ** (((state >>> 7) % 90) - 45));
    }
    const chunks = splitIntoChunks(doubles, [5, 100]);
    const words = makeWordVector(
      device,
      chunks.map(chunk => chunk.map(value => splitDoubleToWords(value ?? 0)))
    );
    const validity = makeValidityVector(
      device,
      chunks.map(chunk => chunk.map(value => value !== null))
    );
    try {
      const actual = await scanWideNumericColumnOnGpu({device, words, format: 'float64', validity});
      const expected = diagnoseFloat32Narrowing('float64', doubles);
      expect(actual).toMatchObject({
        rowCount: expected.rowCount,
        nullCount: expected.nullCount,
        inexactCount: expected.inexactCount,
        overflowCount: expected.overflowCount,
        nonFiniteCount: expected.nonFiniteCount,
        verdict: expected.verdict
      });
      expect(actual.maximumAbsoluteError).toBeLessThanOrEqual(
        Number(expected.maximumAbsoluteError) * (1 + 1e-12)
      );
      expect(actual.maximumAbsoluteError).toBeGreaterThan(
        Number(expected.maximumAbsoluteError) * (1 - 1e-8)
      );
      // A clean column reports exact.
      const clean = makeWordVector(device, [[0, 1, 0.5, 2 ** 24, -3].map(splitDoubleToWords)]);
      try {
        expect(
          (await scanWideNumericColumnOnGpu({device, words: clean, format: 'float64'})).verdict
        ).toBe('exact');
      } finally {
        clean.destroy();
      }
    } finally {
      words.destroy();
      validity.destroy();
    }
  });
});
