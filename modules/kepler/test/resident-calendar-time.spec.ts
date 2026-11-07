// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUKeplerTimeUnit} from '@luma-contributors/kepler';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {makeResidentCalendarTimeView} from '../../../examples/deck/kepler-compute/src/resident-calendar-time';

const INT64_MINIMUM = -(1n << 63n);
const INT64_MAXIMUM = (1n << 63n) - 1n;
const DATE_LIMIT_MILLISECONDS = 8640000000000000n;
const NANOSECONDS_PER_UNIT: Record<GPUKeplerTimeUnit, bigint> = {
  second: 1000000000n,
  millisecond: 1000000n,
  microsecond: 1000n,
  nanosecond: 1n
};

function normalizeOnCPU(value: bigint, unit: GPUKeplerTimeUnit): bigint {
  // Milliseconds are deliberately borrowed, with clipping left to the calendar consumer.
  if (unit === 'millisecond') return value;
  const nanoseconds = value * NANOSECONDS_PER_UNIT[unit];
  if (
    nanoseconds < -DATE_LIMIT_MILLISECONDS * 1000000n ||
    nanoseconds > DATE_LIMIT_MILLISECONDS * 1000000n
  )
    return INT64_MAXIMUM;
  return nanoseconds / 1000000n;
}

function makeValues(unit: GPUKeplerTimeUnit): bigint[] {
  const values = [
    0n,
    1n,
    -1n,
    999n,
    -999n,
    1000n,
    -1000n,
    1001n,
    -1001n,
    999999n,
    -999999n,
    1000000n,
    -1000000n,
    1000001n,
    -1000001n,
    65535n,
    -65535n,
    65536n,
    -65536n,
    0xffffffffn,
    -0xffffffffn,
    0x100000000n,
    -0x100000000n,
    INT64_MINIMUM,
    INT64_MAXIMUM
  ];
  const ticksPerSecond = 1000000000n / NANOSECONDS_PER_UNIT[unit];
  const monthStart = (BigInt(Date.parse('2024-03-01T00:00:00Z')) * ticksPerSecond) / 1000n;
  const dateLimit = (DATE_LIMIT_MILLISECONDS * ticksPerSecond) / 1000n;
  for (const boundary of [monthStart, -monthStart, dateLimit, -dateLimit])
    for (const offset of [-1001n, -1000n, -1n, 0n, 1n, 1000n, 1001n]) {
      const value = boundary + offset;
      if (value >= INT64_MINIMUM && value <= INT64_MAXIMUM) values.push(value);
    }
  let state = 0x2f39c147;
  const randomWord = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state >>> 0;
  };
  for (let index = 0; index < 128; index++) {
    const value = BigInt.asIntN(64, (BigInt(randomWord()) << 32n) | BigInt(randomWord()));
    values.push(value);
    // Also exercise non-overflow multiplication instead of spending every random seconds row on clipping.
    values.push(value % (unit === 'second' ? 8640000000000n : INT64_MAXIMUM));
  }
  return values;
}

function makeSourceWords(values: readonly bigint[], byteOffset: number): Uint32Array {
  const words = new Uint32Array(values.length * 2 + byteOffset / 4);
  words.fill(0xdecafbad);
  const view = new DataView(words.buffer);
  values.forEach((value, index) => view.setBigInt64(byteOffset + index * 8, value, true));
  return words;
}

test('exact calendar time units preserve chunks and match BigInt conversion at word, Date and fractional boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const unit of ['second', 'millisecond', 'microsecond', 'nanosecond'] as const) {
    const graph = new GPUCommandGraph(device, {id: `calendar-time-${unit}`});
    const values = makeValues(unit);
    const chunkValues = [values.slice(0, 3), values.slice(3, 29), values.slice(29)];
    const buffers: Buffer[] = [];
    const inputs: {buffer: Buffer; values: bigint[]; byteOffset: number}[] = [];
    const outputs: Buffer[] = [];
    const conversionViews = [];
    for (const [chunkIndex, chunk] of chunkValues.entries()) {
      // Four-byte (not eight-byte) view alignment exercises component addressing of packed Int64 words.
      const byteOffset = [4, 16, 0][chunkIndex];
      const input = device.createBuffer({
        id: `input-${chunkIndex}`,
        data: makeSourceWords(chunk, byteOffset),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(input);
      inputs.push({buffer: input, values: chunk, byteOffset});
      const source = graph.createDataView(
        graph.importBuffer(
          {id: `input-${chunkIndex}`, byteLength: input.byteLength, usage: input.usage},
          input
        ),
        {format: 'uint32x2', length: chunk.length, byteOffset}
      );
      const converted = makeResidentCalendarTimeView(graph, {
        id: `chunk-${chunkIndex}`,
        timestamps: source,
        sourceType: 'Timestamp',
        unit
      });
      conversionViews.push(converted);
      if (unit === 'millisecond') expect(converted).toBe(source);
      else expect(converted.buffer).not.toBe(source.buffer);
      const output = device.createBuffer({
        id: `output-${chunkIndex}`,
        byteLength: chunk.length * 8,
        usage: Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(output);
      outputs.push(output);
      const destination = graph.importBuffer(
        {id: `output-${chunkIndex}`, byteLength: output.byteLength, usage: output.usage},
        output
      );
      graph.addCopyPass({
        id: `copy-${chunkIndex}`,
        resources: [
          {buffer: converted, usage: 'copy-source'},
          {buffer: destination, usage: 'copy-destination'}
        ],
        compile: () => ({
          encode: ({commandEncoder, getBuffer}) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: getBuffer(converted.buffer),
              sourceOffset: converted.byteOffset,
              destinationBuffer: getBuffer(destination),
              size: chunk.length * 8
            })
        })
      });
    }
    expect(conversionViews.map(view => view.length)).toEqual(
      chunkValues.map(chunk => chunk.length)
    );
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      compiled = graph.compile();
      for (let iteration = 0; iteration < 2; iteration++) {
        if (iteration === 1) {
          for (const input of inputs) {
            input.values.reverse();
            input.buffer.write(makeSourceWords(input.values, input.byteOffset));
          }
        }
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const actual = await Promise.all(outputs.map(output => output.readAsync()));
        for (const [chunkIndex, bytes] of actual.entries()) {
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
          const result = inputs[chunkIndex].values.map((_, index) =>
            view.getBigInt64(index * 8, true)
          );
          expect(result).toEqual(
            inputs[chunkIndex].values.map(value => normalizeOnCPU(value, unit))
          );
          const sourceBytes = await inputs[chunkIndex].buffer.readAsync();
          expect(
            new Uint32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4)
          ).toEqual(makeSourceWords(inputs[chunkIndex].values, inputs[chunkIndex].byteOffset));
        }
      }
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
