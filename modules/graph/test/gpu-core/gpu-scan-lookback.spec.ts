// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUScan, type GPUScanProps} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makePseudoRandomUint32, withDeviceCapabilities} from './gpu-primitive-test-utils';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
/** Words the look-back scan uses as check constants; sums equal to them must not confuse readers. */
const AGGREGATE_CHECK = 0x9e3779b9;
const INCLUSIVE_CHECK = 0x85ebca6b;

type ScanRun = {output: number[]; total?: number};

it('GPUScan single-workgroup path matches a CPU reference for 257 to 4096 rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const length of [257, 300, 512, 1000, 2049, 4095, 4096]) {
    const values = makePseudoRandomUint32(length, length).map(value => value >>> 24);
    for (const mode of ['exclusive', 'inclusive'] as const) {
      for (const disableSubgroups of [false, true]) {
        const result = await runScan(device, values, {mode}, {disableSubgroups, withTotal: true});
        const expected = getCpuScan(values, mode);
        expect(result.output, `${length} rows ${mode} subgroups off: ${disableSubgroups}`).toEqual(
          expected.output
        );
        expect(result.total, `${length} rows total`).toBe(expected.total);
      }
    }
  }
});

it('GPUScan total is zero for empty input and equals the sum otherwise', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const length of [0, 1, 255, 256, 70001]) {
    const values = makePseudoRandomUint32(length, 7).map(value => value >>> 20);
    const result = await runScan(device, values, {}, {withTotal: true});
    expect(result.total, `${length} rows`).toBe(getCpuScan(values, 'exclusive').total);
  }
});

it('GPUScan decoupled look-back matches the multi-pass scan across sizes and modes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const length of [1, 255, 1023, 1024, 1025, 4097, 100000, 1048577]) {
    const values = makePseudoRandomUint32(length, length + 3);
    for (const mode of ['exclusive', 'inclusive'] as const) {
      const expected = getCpuScan(values, mode);
      const result = await runScan(
        device,
        values,
        {mode, algorithm: 'decoupled-lookback'},
        {withTotal: true}
      );
      expect(result.output, `${length} rows ${mode}`).toEqual(expected.output);
      expect(result.total, `${length} rows total`).toBe(expected.total);
    }
  }
}, 120000);

it('GPUScan decoupled look-back survives adversarial inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const tileSize = 1024;
  const length = tileSize * 40 + 17;
  const cases: Record<string, Uint32Array> = {
    zeros: new Uint32Array(length),
    ones: new Uint32Array(length).fill(1),
    // Sums wrap modulo 2^32 on every tile.
    maximum: new Uint32Array(length).fill(0xffffffff),
    // Tile aggregates equal the check constants, so a stale zero check word would validate.
    aggregateCheck: makeTileSumValues(length, tileSize, AGGREGATE_CHECK),
    inclusiveCheck: makeTileSumValues(length, tileSize, INCLUSIVE_CHECK),
    // One huge tile followed by empty ones makes late tiles wait on a slow predecessor.
    heavyFirstTile: Uint32Array.from({length}, (_, index) => (index < tileSize ? 0xfffffff : 0)),
    alternating: Uint32Array.from({length}, (_, index) => (index % 2 ? 0xffffffff : 1))
  };
  for (const [name, values] of Object.entries(cases)) {
    for (const mode of ['exclusive', 'inclusive'] as const) {
      const expected = getCpuScan(values, mode);
      for (const lookbackMaximumSpins of [undefined, 0, 1]) {
        const result = await runScan(
          device,
          values,
          {mode, algorithm: 'decoupled-lookback', lookbackMaximumSpins},
          {withTotal: true}
        );
        expect(result.output, `${name} ${mode} spins ${lookbackMaximumSpins ?? 'default'}`).toEqual(
          expected.output
        );
        expect(result.total).toBe(expected.total);
      }
    }
  }
}, 120000);

it('GPUScan decoupled look-back works under bounded dispatch limits and nonzero counting', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const values = makePseudoRandomUint32(50000, 11).map(value => (value % 3 === 0 ? 0 : value));
  const expected = getCpuScan(
    values.map(value => (value === 0 ? 0 : 1)),
    'exclusive'
  );
  // 49 tiles with a limit of 4 per dimension need a three-dimensional dispatch.
  const result = await runScan(
    device,
    values,
    {algorithm: 'decoupled-lookback', countNonzeroInput: true},
    {maxComputeWorkgroupsPerDimension: 4, withTotal: true}
  );
  expect(result.output).toEqual(expected.output);
  expect(result.total).toBe(expected.total);
});

it('GPUScan segmented decoupled look-back matches a CPU reference across sizes and densities', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const length of [1, 255, 1024, 1025, 4097, 70001]) {
    const values = makePseudoRandomUint32(length, length + 5).map(value => value >>> 24);
    for (const [density, modulus] of [
      ['sparse', 997],
      ['dense', 3],
      ['every row', 1]
    ] as const) {
      const flags = makePseudoRandomUint32(length, length + 9).map(value =>
        value % modulus === 0 ? 1 : 0
      );
      for (const mode of ['exclusive', 'inclusive'] as const) {
        const expected = getCpuSegmentedScan(values, flags, mode);
        const result = await runScan(
          device,
          values,
          {mode, algorithm: 'decoupled-lookback'},
          {segmentFlags: flags}
        );
        expect(result.output, `${length} rows ${density} ${mode}`).toEqual(expected);
      }
    }
  }
}, 120000);

it('GPUScan segmented decoupled look-back survives heads at tile edges and forced fallbacks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const tileSize = 1024;
  const length = tileSize * 40 + 17;
  const values = makePseudoRandomUint32(length, 21);
  const cases: Record<string, Uint32Array> = {
    // No head after row zero: one segment spans every tile.
    singleSegment: new Uint32Array(length),
    firstRowOfEveryTile: Uint32Array.from({length}, (_, index) => (index % tileSize === 0 ? 1 : 0)),
    lastRowOfEveryTile: Uint32Array.from({length}, (_, index) =>
      index % tileSize === tileSize - 1 ? 1 : 0
    ),
    // Heads only in the first and last tiles make middle tiles forward long carries.
    outerTiles: Uint32Array.from({length}, (_, index) =>
      index === 5 || index === length - 3 ? 1 : 0
    ),
    // Tile aggregates equal the check constants, so a stale zero check word would validate.
    aggregateCheck: new Uint32Array(length)
  };
  const checkValues = Uint32Array.from({length}, (_, index) =>
    index % tileSize === 0 ? AGGREGATE_CHECK : 0
  );
  for (const [name, flags] of Object.entries(cases)) {
    const scanned = name === 'aggregateCheck' ? checkValues : values;
    for (const mode of ['exclusive', 'inclusive'] as const) {
      const expected = getCpuSegmentedScan(scanned, flags, mode);
      for (const lookbackMaximumSpins of [undefined, 0, 1]) {
        const result = await runScan(
          device,
          scanned,
          {mode, algorithm: 'decoupled-lookback', lookbackMaximumSpins},
          {segmentFlags: flags}
        );
        expect(result.output, `${name} ${mode} spins ${lookbackMaximumSpins ?? 'default'}`).toEqual(
          expected
        );
      }
    }
  }
}, 120000);

function getCpuSegmentedScan(
  values: Uint32Array,
  flags: Uint32Array,
  mode: 'exclusive' | 'inclusive'
): number[] {
  const output: number[] = [];
  let sum = 0;
  for (const [index, value] of values.entries()) {
    if (index === 0 || flags[index]) sum = 0;
    output.push(mode === 'inclusive' ? (sum + value) >>> 0 : sum);
    sum = (sum + value) >>> 0;
  }
  return output;
}

function makeTileSumValues(length: number, tileSize: number, tileSum: number): Uint32Array {
  const values = new Uint32Array(length);
  for (let tileStart = 0; tileStart < length; tileStart += tileSize) values[tileStart] = tileSum;
  return values;
}

function getCpuScan(
  values: Uint32Array,
  mode: 'exclusive' | 'inclusive'
): {output: number[]; total: number} {
  const output: number[] = [];
  let sum = 0;
  for (const value of values) {
    output.push(mode === 'inclusive' ? (sum + value) >>> 0 : sum);
    sum = (sum + value) >>> 0;
  }
  return {output, total: sum};
}

async function runScan(
  device: Device,
  values: Uint32Array,
  scanProps: Partial<GPUScanProps>,
  options: {
    disableSubgroups?: boolean;
    maxComputeWorkgroupsPerDimension?: number;
    withTotal?: boolean;
    segmentFlags?: Uint32Array;
  } = {}
): Promise<ScanRun> {
  return withDeviceCapabilities(device, options, async () => {
    const length = values.length;
    const inputBuffer = device.createBuffer({
      data: length ? values : new Uint32Array(1),
      usage: STORAGE_USAGE
    });
    const outputBuffer = device.createBuffer({
      byteLength: Math.max(length, 1) * 4,
      usage: STORAGE_USAGE
    });
    const totalBuffer = device.createBuffer({
      data: new Uint32Array([0xdeadbeef]),
      usage: STORAGE_USAGE
    });
    const flagsBuffer = options.segmentFlags
      ? device.createBuffer({data: options.segmentFlags, usage: STORAGE_USAGE})
      : undefined;
    const graph = new GPUCommandGraph(device);
    const importView = (id: string, buffer: Buffer, viewLength: number) =>
      graph.createDataView(
        graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
        {format: 'uint32', length: viewLength}
      );
    graph.add(
      new GPUScan({
        ...scanProps,
        input: importView('input', inputBuffer, length),
        output: importView('output', outputBuffer, length),
        ...(flagsBuffer && {segmentFlags: importView('flags', flagsBuffer, length)}),
        ...(options.withTotal && {total: importView('total', totalBuffer, 1)})
      })
    );
    const compiled = graph.compile();
    try {
      const commandEncoder = device.createCommandEncoder({id: 'scan-lookback-test'});
      compiled.encode(commandEncoder, {parameters: undefined});
      device.submit(commandEncoder.finish());
      const outputBytes = await outputBuffer.readAsync();
      const totalBytes = await totalBuffer.readAsync();
      return {
        output: Array.from(new Uint32Array(outputBytes.buffer, outputBytes.byteOffset, length)),
        total: options.withTotal
          ? new Uint32Array(totalBytes.buffer, totalBytes.byteOffset, 1)[0]
          : undefined
      };
    } finally {
      compiled.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
      totalBuffer.destroy();
      flagsBuffer?.destroy();
    }
  });
}
