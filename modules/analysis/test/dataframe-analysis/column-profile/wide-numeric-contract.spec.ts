// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  aggregateInteger64OnGpu,
  aggregateDecimal128OnGpu,
  aggregateDecimal128Column,
  scanWideNumericColumnOnGpu,
  splitBigIntToWords,
  splitDecimal128ToWords,
  summarizeInteger64Reference,
  summarizeDecimal128Reference,
  diagnoseFloat32Narrowing
} from '../../../src/dataframe-analysis/column-profile/index';
import {requireTestDevice} from '../../utils/require-test-device';

function makeIntegerVectors(device: Device, chunks: readonly (readonly (bigint | null)[])[]) {
  const words = new GPUVector({
    name: 'integer-words',
    type: 'data',
    format: 'uint32x2',
    ownsData: true,
    data: chunks.map(
      values =>
        new GPUData({
          format: 'uint32x2',
          length: values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: Uint32Array.from(
              values.length ? values.flatMap(value => splitBigIntToWords(value ?? 0n)) : [0, 0]
            ),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
  const validity = new GPUVector({
    name: 'validity',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: chunks.map(
      values =>
        new GPUData({
          format: 'uint32',
          length: values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: Uint32Array.from(
              values.length ? values.map(value => (value === null ? 0 : 1)) : [0]
            ),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
  return {words, validity};
}

test('shared wide summaries and exact bigint narrowing errors agree on chunked and all-null columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const halfwayPlusOne = (1n << 63n) + (1n << 39n) + 1n;
  for (const chunks of [
    [[0n, null], [], [(1n << 64n) - 1n, halfwayPlusOne]],
    [[null], [], [null]]
  ] as const) {
    const values = chunks.flat();
    const vectors = makeIntegerVectors(device, chunks);
    try {
      for (const signed of [false, true]) {
        const normalized = values.map(value =>
          value === null ? null : signed ? BigInt.asIntN(64, value) : value
        );
        expect(await aggregateInteger64OnGpu({device, ...vectors, signed})).toEqual(
          summarizeInteger64Reference(normalized, signed)
        );
        const format = signed ? 'int64' : 'uint64';
        expect(await scanWideNumericColumnOnGpu({device, ...vectors, format})).toEqual(
          diagnoseFloat32Narrowing(format, normalized)
        );
      }
    } finally {
      vectors.words.destroy();
      vectors.validity.destroy();
    }
  }
});

test('shared Decimal128 summary retains cross-word carry sums and scale-normalized filtering', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = [(1n << 127n) - 1n, -(1n << 127n), 1234n, 1235n];
  const words = new GPUVector({
    name: 'decimal-words',
    type: 'data',
    format: 'uint32x4',
    ownsData: true,
    data: [
      new GPUData({
        format: 'uint32x4',
        length: values.length,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: Uint32Array.from(values.flatMap(splitDecimal128ToWords)),
          usage: Buffer.STORAGE | Buffer.COPY_DST
        })
      })
    ]
  });
  try {
    expect(await aggregateDecimal128OnGpu({device, words, scale: 2})).toEqual(
      summarizeDecimal128Reference(values, 2)
    );
    expect(
      await aggregateDecimal128OnGpu({
        device,
        words,
        scale: 2,
        comparison: {operator: '>=', constant: 12345n, constantScale: 3}
      })
    ).toEqual(summarizeDecimal128Reference(values, 2, value => value >= 1235n));
  } finally {
    words.destroy();
  }
});

test('Arrow Decimal128 aggregation preserves sliced chunk values and nullable bitmap offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = [999n, 11n, null, (1n << 96n) + 7n, -22n, 888n];
  const type = new arrow.Decimal(2, 38, 128);
  const original = new arrow.Vector([
    arrow.makeData({
      type,
      length: values.length,
      data: Uint32Array.from(values.flatMap(value => splitDecimal128ToWords(value ?? 0n))),
      nullBitmap: new Uint8Array([0b111011]),
      nullCount: 1
    })
  ]);
  const column = new arrow.Vector([...original.slice(1, 4).data, ...original.slice(4, 5).data]);
  expect(await aggregateDecimal128Column(device, column)).toEqual(
    summarizeDecimal128Reference(values.slice(1, 5), 2)
  );
});
