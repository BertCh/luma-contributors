// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGroupStatistics,
  type GPUGroupStatistic,
  type GPUGroupStatisticsColumnOutput
} from '../../../src/dataframe-analysis/group-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  canonicalizeValue,
  computeGroupStatisticsOnCPU,
  createRandom,
  type GroupStatisticsOracleResult
} from './group-statistics-oracle';
import {ORDER_STATISTIC_EDGE_POLICY} from '../dataframe-edge-policy';
import {requireTestDevice} from '../../utils/require-test-device';
import {withGroupStatisticsTestBindingLimit} from './group-statistics-test-device';

const SUM_SCALE = 65536;
const ALL_STATISTICS: GPUGroupStatistic[] = [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'variance',
  'standardDeviation',
  'skewness',
  'kurtosis',
  'median',
  'percentiles',
  'mode',
  'uniqueCount',
  'zScore'
];

/**
 * Stated float32 tolerances of the moments against the float64 oracle. Measured worst relative
 * errors on this data are about 1e-7 (mean, variance), 2e-6 (skewness, kurtosis) and 7e-6
 * (z-score); the tolerances leave an order of magnitude of headroom for other GPUs.
 */
const TOLERANCE = {
  mean: {relative: 1e-6, absolute: 1e-6},
  variance: {relative: 2e-5, absolute: 1e-4},
  skewness: {relative: 1e-4, absolute: 1e-4},
  kurtosis: {relative: 1e-4, absolute: 1e-4},
  zScore: {relative: 1e-4, absolute: 1e-4}
};

const FIELDS: Record<GPUGroupStatistic, (keyof GPUGroupStatisticsColumnOutput)[]> = {
  count: ['counts'],
  sum: ['sums', 'sumValues'],
  mean: ['means'],
  minimum: ['minimums'],
  maximum: ['maximums'],
  variance: ['variances'],
  standardDeviation: ['standardDeviations'],
  skewness: ['skewness'],
  kurtosis: ['kurtosis'],
  median: ['medians'],
  percentiles: ['quantiles'],
  mode: ['modes'],
  uniqueCount: ['uniqueCounts'],
  zScore: ['zScores']
};

type ColumnScene = {values: Float32Array; statistics: GPUGroupStatistic[]};

type Scene = {
  scratchTileSize?: number;
  keyBits: 32 | 64;
  keys: bigint[];
  mask?: Uint32Array;
  columns: ColumnScene[];
  variance?: 'sample' | 'population';
  fractions: number[];
  capacity: number;
  /** Dense mode: `uint32` keys in `[0, keyCount)` index the output rows; `capacity` equals it. */
  keyCount?: number;
};

type ActualColumn = Partial<Record<keyof GPUGroupStatisticsColumnOutput, number[]>>;

type Actual = {
  count: number;
  overflow: number;
  total: number;
  keys: bigint[];
  counts: number[];
  columns: ActualColumn[];
  /** Raw words of every output buffer, in a fixed order. */
  words: number[];
};

class Harness {
  readonly buffers: Buffer[] = [];
  constructor(readonly device: Device) {}

  input(values: Float32Array | Uint32Array): Buffer {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return buffer;
  }

  output(words: number): Buffer {
    const buffer = createOutputBuffer(this.device, words);
    this.buffers.push(buffer);
    return buffer;
  }

  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

function packKeys(keys: readonly bigint[], keyBits: 32 | 64): Uint32Array {
  const words = new Uint32Array(keys.length * (keyBits === 64 ? 2 : 1));
  for (const [row, key] of keys.entries()) {
    if (keyBits === 64) {
      words[2 * row] = Number(key & 0xffffffffn);
      words[2 * row + 1] = Number(key >> 32n);
    } else {
      words[row] = Number(key);
    }
  }
  return words;
}

async function readWords(buffer: Buffer, length: number): Promise<Uint32Array> {
  const bytes = await buffer.readAsync();
  return new Uint32Array(bytes.slice(0, length * 4).buffer.slice(0, length * 4));
}

type InputChunking = {
  keys: number[];
  values: number[][];
  /** Separate physical buffers; otherwise use offset views of the rewritable input buffers. */
  separateBuffers?: boolean;
};

/** One compiled graph with rewritable keys, mask, values and percentile fractions. */
async function createRun(harness: Harness, scene: Scene, chunking?: InputChunking) {
  const {device} = harness;
  const rows = scene.keys.length;
  const {capacity} = scene;
  const fractionCount = scene.fractions.length;
  const graph = new GPUCommandGraph(device, {id: 'group-statistics-graph'});
  const inputWrites = new Map<
    string,
    {buffer: Buffer; byteOffset: number; rowOffset: number; length: number; stride: number}[]
  >();
  function rewriteInput(name: string, data: Float32Array | Uint32Array): void {
    for (const segment of inputWrites.get(name) ?? []) {
      if (segment.length > 0)
        segment.buffer.write(
          data.subarray(
            segment.rowOffset * segment.stride,
            (segment.rowOffset + segment.length) * segment.stride
          ),
          segment.byteOffset
        );
    }
  }
  function inputView<Format extends 'float32' | 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    data: Float32Array | Uint32Array,
    lengths?: number[]
  ): GraphDataView<Format> | GraphVectorView<Format> {
    const packed = importGraphBuffer(graph, name, buffer, format, rows);
    const stride = format === 'uint32x2' ? 2 : 1;
    const segments: {
      buffer: Buffer;
      byteOffset: number;
      rowOffset: number;
      length: number;
      stride: number;
    }[] = [];
    inputWrites.set(name, segments);
    if (!lengths) {
      segments.push({buffer, byteOffset: 0, rowOffset: 0, length: rows, stride});
      return packed;
    }
    let rowOffset = 0;
    const chunks = lengths.map((length, chunkIndex) => {
      const byteOffset = rowOffset * stride * 4;
      const values = data.subarray(rowOffset * stride, (rowOffset + length) * stride);
      rowOffset += length;
      // Record the buffers actually bound, including physically separate chunks.
      const chunkBuffer = chunking?.separateBuffers
        ? harness.input(values.length ? values : new Uint32Array(stride))
        : buffer;
      const chunkByteOffset = chunking?.separateBuffers ? 0 : byteOffset;
      segments.push({
        buffer: chunkBuffer,
        byteOffset: chunkByteOffset,
        rowOffset: rowOffset - length,
        length,
        stride
      });
      return chunking?.separateBuffers
        ? importGraphBuffer(graph, `${name}-chunk-${chunkIndex}`, chunkBuffer, format, length)
        : graph.createDataView(packed.buffer, {format, length, byteOffset});
    });
    expect(rowOffset).toBe(rows);
    return new GraphVectorView({
      id: name,
      name,
      format,
      length: rows,
      valueLength: rows * stride,
      stride,
      byteStride: stride * 4,
      rowByteLength: stride * 4,
      data: chunks
    });
  }
  const keyWordCount = scene.keyBits === 64 ? 2 : 1;
  const keyFormat = scene.keyBits === 64 ? 'uint32x2' : 'uint32';
  const keysBuffer = harness.input(packKeys(scene.keys, scene.keyBits));
  const maskBuffer = harness.input(scene.mask ?? new Uint32Array(rows).fill(1));
  const valueBuffers = scene.columns.map(column => harness.input(column.values));
  const parameters = fractionCount
    ? new GPUParameterBuffer(device, {
        id: 'fractions',
        format: 'float32',
        length: fractionCount,
        values: Float32Array.from(scene.fractions)
      })
    : undefined;
  const outputBuffers = {
    keys: harness.output(capacity * keyWordCount),
    counts: harness.output(capacity),
    count: harness.output(1),
    overflow: harness.output(1),
    total: harness.output(1)
  };
  const columnOutputs: {
    views: GPUGroupStatisticsColumnOutput;
    buffers: Partial<Record<keyof GPUGroupStatisticsColumnOutput, Buffer>>;
    lengths: Partial<Record<keyof GPUGroupStatisticsColumnOutput, number>>;
  }[] = [];
  const columns = scene.columns.map((column, columnIndex) => {
    const views: Record<string, GraphDataView> = {};
    const buffers: Partial<Record<keyof GPUGroupStatisticsColumnOutput, Buffer>> = {};
    const lengths: Partial<Record<keyof GPUGroupStatisticsColumnOutput, number>> = {};
    for (const statistic of column.statistics) {
      for (const field of FIELDS[statistic]) {
        const length =
          field === 'zScores' ? rows : field === 'quantiles' ? capacity * fractionCount : capacity;
        const format =
          field === 'sums'
            ? 'uint32x2'
            : field === 'counts' || field === 'uniqueCounts'
              ? 'uint32'
              : 'float32';
        const buffer = harness.output(length * (field === 'sums' ? 2 : 1));
        buffers[field] = buffer;
        lengths[field] = length * (field === 'sums' ? 2 : 1);
        views[field] = importGraphBuffer(graph, `c${columnIndex}-${field}`, buffer, format, length);
      }
    }
    columnOutputs.push({
      views: views as GPUGroupStatisticsColumnOutput,
      buffers,
      lengths
    });
    return {
      values: inputView(
        `values-${columnIndex}`,
        valueBuffers[columnIndex],
        'float32',
        column.values,
        chunking?.values[columnIndex]
      ),
      statistics: column.statistics,
      output: views as GPUGroupStatisticsColumnOutput
    };
  });
  const contributor = new GPUGroupStatistics({
    scratchTileSize: scene.scratchTileSize,
    keys: inputView(
      'keys',
      keysBuffer,
      keyFormat,
      packKeys(scene.keys, scene.keyBits),
      chunking?.keys
    ) as never,
    mask: inputView(
      'mask',
      maskBuffer,
      'uint32',
      scene.mask ?? new Uint32Array(rows).fill(1),
      chunking?.keys
    ),
    columns,
    variance: scene.variance,
    keyCount: scene.keyCount,
    probabilities: parameters?.importToGraph(graph),
    sumScale: SUM_SCALE,
    output: {
      keys: importGraphBuffer(graph, 'out-keys', outputBuffers.keys, keyFormat, capacity) as never,
      counts: importGraphBuffer(graph, 'out-counts', outputBuffers.counts, 'uint32', capacity),
      status: {
        count: importGraphBuffer(graph, 'out-count', outputBuffers.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'out-overflow', outputBuffers.overflow, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'out-total', outputBuffers.total, 'uint32', 1)
      }
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  const allOutputs: [Buffer, number][] = [
    [outputBuffers.keys, capacity * keyWordCount],
    [outputBuffers.counts, capacity],
    [outputBuffers.count, 1],
    [outputBuffers.overflow, 1],
    [outputBuffers.total, 1]
  ];
  for (const output of columnOutputs) {
    for (const [field, buffer] of Object.entries(output.buffers)) {
      allOutputs.push([buffer, output.lengths[field as keyof GPUGroupStatisticsColumnOutput]!]);
    }
  }
  return {
    compiled,
    parameters,
    keysBuffer,
    maskBuffer,
    valueBuffers,
    rewriteColumn(columnIndex: number, values: Float32Array): void {
      rewriteInput(`values-${columnIndex}`, values);
    },
    rewriteKeys(keys: bigint[]): void {
      rewriteInput('keys', packKeys(keys, scene.keyBits));
    },
    poison(): void {
      for (const [buffer, length] of allOutputs) {
        buffer.write(new Uint32Array(length).fill(0xa5a5a5a5));
      }
    },
    async run(): Promise<Actual> {
      submitGraph(device, compiled, undefined);
      const words: number[] = [];
      for (const [buffer, length] of allOutputs) {
        words.push(...(await readWords(buffer, length)));
      }
      const [count] = await readWords(outputBuffers.count, 1);
      const [overflow] = await readWords(outputBuffers.overflow, 1);
      const [total] = await readWords(outputBuffers.total, 1);
      const keyWords = await readWords(outputBuffers.keys, capacity * keyWordCount);
      const keys: bigint[] = [];
      for (let row = 0; row < capacity; row++) {
        keys.push(
          scene.keyBits === 64
            ? (BigInt(keyWords[2 * row + 1]) << 32n) | BigInt(keyWords[2 * row])
            : BigInt(keyWords[row])
        );
      }
      const counts = Array.from(await readWords(outputBuffers.counts, capacity));
      const actualColumns: ActualColumn[] = [];
      for (const output of columnOutputs) {
        const column: ActualColumn = {};
        for (const [field, buffer] of Object.entries(output.buffers)) {
          const name = field as keyof GPUGroupStatisticsColumnOutput;
          const raw = await readWords(buffer, output.lengths[name]!);
          column[name] =
            name === 'counts' || name === 'uniqueCounts' || name === 'sums'
              ? Array.from(raw)
              : Array.from(new Float32Array(raw.buffer, raw.byteOffset, raw.length));
        }
        actualColumns.push(column);
      }
      return {
        count,
        overflow,
        total,
        keys,
        counts,
        columns: actualColumns,
        words
      };
    },
    destroy(): void {
      compiled.destroy();
      parameters?.destroy();
    }
  };
}

function expectClose(
  actual: number,
  expected: number,
  tolerance: {relative: number; absolute: number},
  label: string
): void {
  if (Number.isNaN(expected)) {
    expect(Number.isNaN(actual), `${label} should be NaN, got ${actual}`).toBe(true);
    return;
  }
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance.relative * Math.abs(expected) + tolerance.absolute
  );
}

function expectExactFloat(actual: number, expected: number, label: string): void {
  const wanted = Math.fround(expected);
  expect(
    Object.is(actual, wanted) || (Number.isNaN(actual) && Number.isNaN(wanted)),
    `${label}: ${actual} vs ${wanted}`
  ).toBe(true);
}

function expectMatchesOracle(
  actual: Actual,
  scene: Scene,
  label: string
): GroupStatisticsOracleResult {
  const expected = computeGroupStatisticsOnCPU({
    // Dense mode skips keys outside [0, keyCount) like reserved keys.
    keys:
      scene.keyCount === undefined
        ? scene.keys
        : scene.keys.map(key => (key >= BigInt(scene.keyCount!) ? 0xffffffffn : key)),
    keyBits: scene.keyBits,
    mask: scene.mask,
    columns: scene.columns.map(column => column.values),
    variance: scene.variance ?? 'sample',
    fractions: scene.fractions,
    sumScale: SUM_SCALE,
    capacity: scene.capacity
  });
  const groupCount = expected.groups.length;
  const dense = scene.keyCount !== undefined;
  const groupByKey = new Map(expected.groups.map(group => [Number(group.key), group]));
  const emptyKey = scene.keyBits === 64 ? 0xffffffffffffffffn : 0xffffffffn;
  if (dense) {
    expect(actual.total, `${label} total`).toBe(scene.keyCount);
    expect(actual.count, `${label} count`).toBe(scene.keyCount);
    expect(actual.overflow, `${label} overflow`).toBe(0);
    for (let key = 0; key < scene.capacity; key++) {
      expect(actual.keys[key], `${label} key ${key}`).toBe(BigInt(key));
      expect(actual.counts[key], `${label} count of key ${key}`).toBe(
        groupByKey.get(key)?.count ?? 0
      );
    }
  } else {
    expect(actual.total, `${label} total`).toBe(expected.requiredCount);
    expect(actual.count, `${label} count`).toBe(groupCount);
    expect(actual.overflow, `${label} overflow`).toBe(
      expected.requiredCount > scene.capacity ? 1 : 0
    );
    expect(actual.keys.slice(0, groupCount), `${label} keys`).toEqual(
      expected.groups.map(group => group.key)
    );
    expect(actual.counts.slice(0, groupCount), `${label} counts`).toEqual(
      expected.groups.map(group => group.count)
    );
    for (let row = groupCount; row < scene.capacity; row++) {
      expect(actual.keys[row], `${label} tail key ${row}`).toBe(emptyKey);
      expect(actual.counts[row], `${label} tail count ${row}`).toBe(0);
    }
  }
  const fractionCount = scene.fractions.length;
  for (const [columnIndex, column] of scene.columns.entries()) {
    const wanted = new Set(column.statistics);
    const out = actual.columns[columnIndex];
    for (let group = 0; group < scene.capacity; group++) {
      const stats = dense
        ? groupByKey.get(group)?.columns[columnIndex]
        : group >= groupCount
          ? undefined
          : expected.groups[group].columns[columnIndex];
      const where = `${label} column ${columnIndex} group ${group}`;
      if (out.counts) {
        expect(out.counts[group], `${where} counts`).toBe(stats?.count ?? 0);
      }
      if (out.sums) {
        const sum = BigInt.asIntN(
          64,
          (BigInt(out.sums[2 * group + 1]) << 32n) | BigInt(out.sums[2 * group])
        );
        expect(sum, `${where} sums`).toBe(stats?.sum ?? 0n);
      }
      if (out.sumValues) {
        expectClose(
          out.sumValues[group],
          stats?.sumValue ?? 0,
          {relative: 1e-6, absolute: 1e-6},
          `${where} sumValues`
        );
      }
      const checks: [
        keyof GPUGroupStatisticsColumnOutput,
        number | undefined,
        'exact' | keyof typeof TOLERANCE
      ][] = [
        ['means', stats?.mean, 'mean'],
        ['minimums', stats?.minimum, 'exact'],
        ['maximums', stats?.maximum, 'exact'],
        ['variances', stats?.variance, 'variance'],
        [
          'standardDeviations',
          stats === undefined ? undefined : Math.sqrt(stats.variance),
          'variance'
        ],
        ['skewness', stats?.skewness, 'skewness'],
        ['kurtosis', stats?.kurtosis, 'kurtosis'],
        ['medians', stats?.median, 'exact'],
        ['modes', stats?.mode, 'exact']
      ];
      for (const [field, value, kind] of checks) {
        const values = out[field];
        if (!values) {
          continue;
        }
        const wantedValue = value ?? NaN;
        if (kind === 'exact') {
          expectExactFloat(values[group], wantedValue, `${where} ${field}`);
        } else if (field === 'standardDeviations') {
          // sd tolerance follows the variance tolerance in the square root domain.
          expectClose(
            values[group],
            wantedValue,
            {relative: TOLERANCE.variance.relative, absolute: 1e-3},
            `${where} ${field}`
          );
        } else {
          expectClose(values[group], wantedValue, TOLERANCE[kind], `${where} ${field}`);
        }
      }
      if (out.uniqueCounts) {
        expect(out.uniqueCounts[group], `${where} unique`).toBe(stats?.uniqueCount ?? 0);
      }
      if (out.quantiles) {
        for (let fraction = 0; fraction < fractionCount; fraction++) {
          expectExactFloat(
            out.quantiles[group * fractionCount + fraction],
            stats?.percentiles[fraction] ?? NaN,
            `${where} percentile ${scene.fractions[fraction]}`
          );
        }
      }
    }
    if (out.zScores) {
      for (let row = 0; row < scene.keys.length; row++) {
        const wantedValue = expected.zScores[columnIndex][row];
        expectClose(
          out.zScores[row],
          wantedValue,
          TOLERANCE.zScore,
          `${label} column ${columnIndex} z ${row}`
        );
      }
    }
    expect(wanted.size).toBeGreaterThan(0);
  }
  return expected;
}

/** Quantized value: a multiple of 1/64 so fixed-point sums are exact. */
function quantize(value: number): number {
  return Math.round(value * 64) / 64;
}

type SceneOptions = {
  seed: number;
  keyBits: 32 | 64;
  columns: GPUGroupStatistic[][];
  capacity?: number;
  mask?: boolean;
  variance?: 'sample' | 'population';
  fractions?: number[];
  bigGroupSize?: number;
  smallGroups?: number;
};

/**
 * Groups of size 1, 2, 3, ~300 and a big one, plus many small ones. Keys of the 64-bit scenes
 * differ in the high word only for some groups. Column 0 has quantized values with NaN, +-Infinity
 * and -0; column 1 has small integers (mode ties); column 2 has arbitrary f32 values.
 */
function createScene(options: SceneOptions): Scene {
  const random = createRandom(options.seed);
  const keys: bigint[] = [];
  const reserved = options.keyBits === 64 ? 0xffffffffffffffffn : 0xffffffffn;
  const groupKeys: bigint[] = [];
  const sizes = [1, 2, 3, 300, options.bigGroupSize ?? 5000];
  for (let group = 0; group < (options.smallGroups ?? 40); group++) {
    sizes.push(4 + Math.floor(random() * 40));
  }
  for (let group = 0; group < sizes.length; group++) {
    if (options.keyBits === 64) {
      // Some groups share the low word and differ in the high word only.
      const low = BigInt(group % 3 === 0 ? 7 : Math.floor(random() * 0xffffffff));
      groupKeys.push((BigInt(group * 5 + 1) << 32n) | low);
    } else {
      groupKeys.push(BigInt(group * 977 + 13 + (group === 1 ? 0xf0000000 : 0)));
    }
  }
  for (const [group, size] of sizes.entries()) {
    for (let row = 0; row < size; row++) {
      keys.push(groupKeys[group]);
    }
  }
  // Invalid keys.
  for (let index = 0; index < 25; index++) {
    keys.push(reserved);
  }
  // Fisher-Yates shuffle.
  for (let row = keys.length - 1; row > 0; row--) {
    const other = Math.floor(random() * (row + 1));
    [keys[row], keys[other]] = [keys[other], keys[row]];
  }
  const rows = keys.length;
  const columns: ColumnScene[] = options.columns.map((statistics, columnIndex) => {
    const values = new Float32Array(rows);
    for (let row = 0; row < rows; row++) {
      const roll = random();
      if (columnIndex === 1) {
        values[row] = roll < 0.03 ? NaN : 1 + Math.floor(random() * 5);
      } else if (columnIndex === 2) {
        values[row] = roll < 0.02 ? NaN : Math.fround((random() - 0.3) * 1000);
      } else {
        values[row] =
          roll < 0.03
            ? NaN
            : roll < 0.04
              ? Infinity
              : roll < 0.05
                ? -0
                : roll < 0.06
                  ? -Infinity
                  : quantize((random() - 0.4) * 600);
      }
    }
    return {values, statistics};
  });
  const mask = options.mask
    ? // The three smallest groups keep every row so sizes 1, 2 and 3 are exercised.
      Uint32Array.from({length: rows}, (_, row) =>
        groupKeys.slice(0, 3).includes(keys[row]) || random() >= 0.15 ? 1 : 0
      )
    : undefined;
  return {
    keyBits: options.keyBits,
    keys,
    mask,
    columns,
    variance: options.variance,
    fractions: options.fractions ?? [],
    capacity: options.capacity ?? sizes.length + 8
  };
}

it('GPUGroupStatistics computes every statistic for u32 keys, a mask and NaN values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const scene = createScene({
    seed: 11,
    keyBits: 32,
    mask: true,
    fractions: [0, 0.1, 0.5, 0.9, 1],
    columns: [ALL_STATISTICS]
  });
  const run = await createRun(harness, scene);
  const actual = await run.run();
  const expected = expectMatchesOracle(actual, scene, 'u32');
  // A few groups are tiny and the biggest holds more than 4096 rows.
  expect(Math.max(...expected.groups.map(group => group.count))).toBeGreaterThan(4096);
  expect(expected.groups.some(group => group.count === 1)).toBe(true);
  expect(expected.groups.some(group => group.count === 2)).toBe(true);
  expect(expected.groups.some(group => group.columns[0].count === 0)).toBe(false);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics matches the oracle for u64 keys that differ in the high word, 3 columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const scene = createScene({
    seed: 23,
    keyBits: 64,
    mask: true,
    variance: 'population',
    fractions: [0.25, 0.5, 0.75],
    columns: [
      ['count', 'sum', 'mean', 'minimum', 'maximum', 'standardDeviation', 'median', 'zScore'],
      ['mode', 'uniqueCount', 'percentiles', 'variance', 'count'],
      ['kurtosis', 'skewness', 'sum', 'percentiles', 'zScore']
    ]
  });
  const run = await createRun(harness, scene);
  const actual = await run.run();
  const expected = expectMatchesOracle(actual, scene, 'u64');
  // Keys that share a low word and differ in the high word are separate groups.
  const lowWords = new Map<bigint, number>();
  for (const group of expected.groups) {
    lowWords.set(group.key & 0xffffffffn, (lowWords.get(group.key & 0xffffffffn) ?? 0) + 1);
  }
  expect(Math.max(...lowWords.values())).toBeGreaterThan(5);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics keeps the smallest keys on capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const keyBits of [32, 64] as const) {
    const harness = new Harness(device);
    const scene = createScene({
      seed: 5 + keyBits,
      keyBits,
      bigGroupSize: 600,
      capacity: 13,
      fractions: [0.5],
      columns: [
        [
          'count',
          'mean',
          'minimum',
          'median',
          'percentiles',
          'mode',
          'uniqueCount',
          'variance',
          'zScore'
        ]
      ]
    });
    const run = await createRun(harness, scene);
    const actual = await run.run();
    const expected = expectMatchesOracle(actual, scene, `overflow ${keyBits}`);
    expect(actual.total).toBeGreaterThan(13);
    expect(actual.overflow).toBe(1);
    expect(actual.count).toBe(13);
    expect(expected.groups.length).toBe(13);
    // Rows of dropped groups have NaN z-scores.
    const zScores = actual.columns[0].zScores!;
    expect(zScores.some(Number.isNaN)).toBe(true);
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics percentile fractions, values and masks change without a rebuild', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const scene = createScene({
    seed: 77,
    keyBits: 32,
    bigGroupSize: 800,
    smallGroups: 20,
    fractions: [0.1, 0.5, 0.9],
    columns: [['median', 'percentiles', 'count', 'mean', 'zScore']]
  });
  const run = await createRun(harness, scene);
  const compiled = run.compiled;
  const first = await run.run();
  expectMatchesOracle(first, scene, 'fractions A');
  // Rewrite only the percentile parameter view and re-encode the SAME compiled graph.
  const rewritten = {...scene, fractions: [0.25, 0, 1]};
  run.parameters!.write(Float32Array.from(rewritten.fractions));
  const second = await run.run();
  expectMatchesOracle(second, rewritten, 'fractions B');
  expect(run.compiled).toBe(compiled);
  expect(second.columns[0].quantiles).not.toEqual(first.columns[0].quantiles);
  // Out-of-range and non-finite fractions give NaN (the shared order-statistic policy).
  const odd = {...scene, fractions: [-3, 7, NaN]};
  run.parameters!.write(Float32Array.from(odd.fractions));
  expectMatchesOracle(await run.run(), odd, 'fractions odd');
  // Rewrite values and the mask too.
  const random = createRandom(5);
  const values = Float32Array.from(scene.columns[0].values, () => quantize((random() - 0.5) * 80));
  const mask = Uint32Array.from(scene.keys, () => (random() < 0.3 ? 0 : 1));
  run.valueBuffers[0].write(values);
  run.maskBuffer.write(mask);
  const changed = {
    ...scene,
    fractions: [0.1, 0.5, 0.9],
    mask,
    columns: [{...scene.columns[0], values}]
  };
  run.parameters!.write(Float32Array.from(changed.fractions));
  expectMatchesOracle(await run.run(), changed, 'values and mask');
  expect(run.compiled).toBe(compiled);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics encodes bitwise identically twice and handles an all-masked input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const scene = createScene({
    seed: 91,
    keyBits: 64,
    bigGroupSize: 1500,
    smallGroups: 10,
    fractions: [0.3, 0.7],
    columns: [ALL_STATISTICS, ['mode', 'median', 'uniqueCount']]
  });
  const run = await createRun(harness, scene);
  const first = await run.run();
  run.poison();
  const second = await run.run();
  expect(second.words.length).toBe(first.words.length);
  expect(second.words.every((word, index) => word === first.words[index])).toBe(true);
  expectMatchesOracle(second, scene, 'determinism');
  // Everything masked off: no groups, an empty tail and NaN z-scores.
  run.maskBuffer.write(new Uint32Array(scene.keys.length));
  run.poison();
  const empty = await run.run();
  expect([empty.count, empty.total, empty.overflow]).toEqual([0, 0, 0]);
  expect(empty.columns[0].zScores!.every(Number.isNaN)).toBe(true);
  expect(empty.counts.every(count => count === 0)).toBe(true);
  expect(empty.columns[1].uniqueCounts!.every(count => count === 0)).toBe(true);
  expect(canonicalizeValue(-0)).toBe(0);
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics handles every statistic alone and a contributor with no columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const sets: GPUGroupStatistic[][] = [
    ...ALL_STATISTICS.map(statistic => [statistic]),
    ['mode', 'uniqueCount'],
    ['minimum', 'maximum'],
    ['median', 'mode']
  ];
  for (const statistics of sets) {
    const scene = createScene({
      seed: 3,
      keyBits: 32,
      bigGroupSize: 300,
      smallGroups: 8,
      mask: true,
      fractions: statistics.includes('percentiles') ? [0.2, 0.8] : [],
      columns: [statistics]
    });
    const run = await createRun(harness, scene);
    expectMatchesOracle(await run.run(), scene, statistics.join('+'));
    run.destroy();
  }
  // No value columns: only keys, counts, count, overflow and total.
  const scene = createScene({seed: 4, keyBits: 64, bigGroupSize: 100, smallGroups: 5, columns: []});
  const run = await createRun(harness, scene);
  expectMatchesOracle(await run.run(), scene, 'no columns');
  run.destroy();
  harness.destroy();
});

it('GPUGroupStatistics treats a large constant group as zero variance despite mean rounding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const rows = 6000;
  const keys = Array.from({length: rows}, (_, row) => (row < 5000 ? 9n : BigInt(row % 3)));
  const values = Float32Array.from({length: rows}, (_, row) =>
    row < 5000 ? Math.fround(100.1) : Math.fround(row * 0.37)
  );
  const scene: Scene = {
    keyBits: 32,
    keys,
    columns: [
      {
        values,
        statistics: ['mean', 'variance', 'standardDeviation', 'skewness', 'kurtosis', 'zScore']
      }
    ],
    fractions: [],
    capacity: 8
  };
  const run = await createRun(harness, scene);
  const actual = await run.run();
  expectMatchesOracle(actual, scene, 'constant group');
  const group = actual.keys.indexOf(9n);
  expect(actual.columns[0].skewness![group]).toBeNaN();
  expect(actual.columns[0].kurtosis![group]).toBeNaN();
  expect(actual.columns[0].zScores!.slice(0, 5000).every(z => z === 0)).toBe(true);
  run.destroy();
  harness.destroy();
});

// DPC-109. Four float32 values of 1e38 sum past the float32 maximum (3.4028235e38), and the
// mean used to be the infinite rounded sum divided by 4. The mean of equal values is the value:
// Math.fround(1e38) = 9.999999680285692e37.
it('GPUGroupStatistics keeps a finite mean when the group sum is past the float32 range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const large = Math.fround(1e38);
  const scene: Scene = {
    keyBits: 32,
    keys: [5n, 5n, 5n, 5n, 6n, 6n],
    columns: [
      {
        // The control group must keep its bits in the column's fixed-point unit, which follows
        // the largest value: 2^35 here (biased exponent 253 - 127 + 1 + 3 headroom bits - 95),
        // so 1 and 3 would quantize to 0. 2^120 and 3 * 2^120 are exact multiples of it.
        values: Float32Array.from([large, large, large, large, 2 ** 120, 3 * 2 ** 120]),
        statistics: ['sum', 'mean', 'variance', 'zScore']
      }
    ],
    fractions: [],
    capacity: 4
  };
  const run = await createRun(harness, scene);
  try {
    const actual = await run.run();
    const column = actual.columns[0];
    const group = actual.keys.indexOf(5n);
    const smallGroup = actual.keys.indexOf(6n);
    expect(column.sumValues![group]).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isFinite(column.means![group]), `mean ${column.means![group]}`).toBe(true);
    expect(column.means![group]).toBe(9.999999680285692e37);
    expect(column.variances![group]).toBe(0);
    expect(column.zScores!.slice(0, 4)).toEqual([0, 0, 0, 0]);
    // The ordinary finite path is unchanged: (2^120 + 3 * 2^120) / 2 = 2^121 exactly.
    expect(column.sumValues![smallGroup]).toBe(2 ** 122);
    expect(column.means![smallGroup]).toBe(2 ** 121);
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics sums, means and z-scores of small-magnitude columns are right to float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  // Group 1 and group 2 hold values far below 1 / 65536, the unit of the fixed-point `sums`
  // words; group 3 shares the column with values ten orders of magnitude larger.
  const groups: [bigint, number[]][] = [
    [1n, [1.0e-4, 1.2e-4, 1.4e-4]],
    [2n, [3e-6, 4e-6, 5e-6, 6e-6]],
    [3n, [1e6, 2e6]]
  ];
  const keys: bigint[] = [];
  const values: number[] = [];
  for (const [key, groupValues] of groups) {
    for (const value of groupValues) {
      keys.push(key);
      values.push(value);
    }
  }
  const scene: Scene = {
    keyBits: 32,
    keys,
    columns: [
      {
        values: Float32Array.from(values),
        statistics: ['sum', 'mean', 'variance', 'standardDeviation', 'zScore']
      }
    ],
    fractions: [],
    capacity: 4
  };
  const run = await createRun(harness, scene);
  try {
    const actual = await run.run();
    const column = actual.columns[0];
    const tight = {relative: 2e-6, absolute: 0};
    let row = 0;
    for (const [group, [, groupValues]] of groups.entries()) {
      // Float64 definitions on the float32 inputs the kernel sees.
      const stored = groupValues.map(Math.fround);
      const count = stored.length;
      const sum = stored.reduce((total, value) => total + value, 0);
      const mean = sum / count;
      const variance =
        stored.reduce((total, value) => total + (value - mean) ** 2, 0) / (count - 1);
      expectClose(column.sumValues![group], sum, tight, `group ${group} sum`);
      expectClose(column.means![group], mean, tight, `group ${group} mean`);
      expectClose(column.variances![group], variance, {relative: 2e-5, absolute: 0}, 'variance');
      for (const value of stored) {
        const expectedScore = (value - mean) / Math.sqrt(variance);
        expectClose(
          column.zScores![row],
          expectedScore,
          {relative: 1e-4, absolute: 1e-4},
          `group ${group} z-score of ${value}`
        );
        row++;
      }
    }
    // Hand values: the first group is 1.0e-4, 1.2e-4, 1.4e-4, mean 1.2e-4, z-scores -1, 0, 1.
    expect(column.means![0]).toBeCloseTo(1.2e-4, 9);
    // Rounded to three decimals; adding 0 turns a rounded -0 into +0.
    expect(column.zScores!.slice(0, 3).map(value => Math.round(value * 1000) / 1000 + 0)).toEqual([
      -1, 0, 1
    ]);
    // The fixed-point words stay in units of 1 / sumScale: round(1e6 * 65536) + round(2e6 * 65536).
    const largeSum = (BigInt(column.sums![2 * 2 + 1]) << 32n) | BigInt(column.sums![2 * 2]);
    expect(largeSum).toBe(3_000_000n * 65536n);
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics z-scores keep the sub-unit part of a large mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  // Values 1e7 .. 1e7 + 3: the mean is 1e7 + 1.5, which float32 cannot hold (its step there is
  // 1), so the shift mean is 1e7 + 2 and the correction S1 / n is -0.5. The sample deviation is
  // sqrt(5 / 3), and the z-scores are (-1.5, -0.5, 0.5, 1.5) / sqrt(5 / 3). Without the
  // correction every z-score is 0.5 / sqrt(5 / 3) = 0.39 too low. A second group one unit apart
  // (mean 2e7 + 0.5) and a group of small values share the column.
  const groups: [bigint, number[]][] = [
    [1n, [1e7, 1e7 + 1, 1e7 + 2, 1e7 + 3]],
    [2n, [2e7, 2e7 + 2, 2e7 - 2, 2e7 + 2]],
    [3n, [1, 2, 4]]
  ];
  const keys: bigint[] = [];
  const values: number[] = [];
  for (const [key, groupValues] of groups) {
    for (const value of groupValues) {
      keys.push(key);
      values.push(value);
    }
  }
  const scene: Scene = {
    keyBits: 32,
    keys,
    columns: [{values: Float32Array.from(values), statistics: ['variance', 'zScore']}],
    fractions: [],
    capacity: 4
  };
  const run = await createRun(harness, scene);
  try {
    const actual = await run.run();
    const column = actual.columns[0];
    let row = 0;
    for (const [group, [, groupValues]] of groups.entries()) {
      // Float64 definitions; every value here is exact in float32.
      const count = groupValues.length;
      const mean = groupValues.reduce((total, value) => total + value, 0) / count;
      const variance =
        groupValues.reduce((total, value) => total + (value - mean) ** 2, 0) / (count - 1);
      expectClose(column.variances![group], variance, {relative: 1e-5, absolute: 0}, 'variance');
      for (const value of groupValues) {
        expectClose(
          column.zScores![row],
          (value - mean) / Math.sqrt(variance),
          {relative: 1e-5, absolute: 1e-5},
          `group ${group} z-score of ${value}`
        );
        row++;
      }
    }
    // Hand values of the first group.
    const deviation = Math.sqrt(5 / 3);
    expect(column.zScores![0]).toBeCloseTo(-1.5 / deviation, 4);
    expect(column.zScores![3]).toBeCloseTo(1.5 / deviation, 4);
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics dense mode returns one row per key with empty keys at count 0 and NaN', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const keyCount = 60;
  const random = createRandom(77);
  const emptyKeys = new Set([3, 17, 59]);
  const keys: bigint[] = [];
  for (let row = 0; row < 3000; row++) {
    let key = Math.floor(random() * keyCount);
    if (emptyKeys.has(key)) {
      key = (key + 1) % keyCount;
    }
    // Out-of-range and reserved keys are skipped like masked rows.
    if (random() < 0.05) {
      key = random() < 0.5 ? keyCount + Math.floor(random() * 40) : 0xffffffff;
    }
    keys.push(BigInt(key));
  }
  const rows = keys.length;
  const mask = Uint32Array.from({length: rows}, () => (random() < 0.9 ? 1 : 0));
  const first = new Float32Array(rows);
  const second = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    first[row] =
      Number(keys[row]) === 5 ? NaN : random() < 0.05 ? NaN : quantize(random() * 50 - 10);
    second[row] = 1 + Math.floor(random() * 5);
  }
  const scene: Scene = {
    keyBits: 32,
    keys,
    mask,
    columns: [
      {
        values: first,
        statistics: ['count', 'sum', 'mean', 'minimum', 'maximum', 'variance', 'zScore']
      },
      {values: second, statistics: ['median', 'percentiles', 'mode', 'uniqueCount']}
    ],
    fractions: [0.25, 0.75],
    capacity: keyCount,
    keyCount
  };
  const harness = new Harness(device);
  const run = await createRun(harness, scene);
  try {
    run.poison();
    const actual = await run.run();
    expectMatchesOracle(actual, scene, 'dense');
    // Keys 3, 17 and 59 have no rows, key 5 has rows with no finite value.
    for (const key of [3, 17, 59]) {
      expect(actual.counts[key]).toBe(0);
      expect(Number.isNaN(actual.columns[0].means![key])).toBe(true);
      expect(actual.columns[0].counts![key]).toBe(0);
    }
    expect(actual.counts[5]).toBeGreaterThan(0);
    expect(actual.columns[0].counts![5]).toBe(0);
    expect(Number.isNaN(actual.columns[0].minimums![5])).toBe(true);
    expect(actual.counts.reduce((total, count) => total + count, 0)).toBeGreaterThan(2000);
    // New contents, same graph.
    const nextKeys = Uint32Array.from(keys, key => (Number(key) + 1) % keyCount);
    run.keysBuffer.write(nextKeys);
    const next = await run.run();
    const shifted = {...scene, keys: Array.from(nextKeys, key => BigInt(key))};
    expectMatchesOracle(next, shifted, 'dense shifted');
  } finally {
    run.destroy();
    harness.destroy();
  }
}, 120000);

it('GPUGroupStatistics dense mode validates its options', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const graph = new GPUCommandGraph(device, {id: 'dense-validation'});
  const view = (words: number) =>
    importGraphBuffer(graph, `v${harness.buffers.length}`, harness.output(words), 'uint32', words);
  const pairView = (rows: number) =>
    importGraphBuffer(
      graph,
      `v${harness.buffers.length}`,
      harness.output(rows * 2),
      'uint32x2',
      rows
    );
  const output = (rows: number) => ({
    keys: view(rows),
    counts: view(rows),
    status: {count: view(1), overflow: view(1)}
  });
  expect(
    () => new GPUGroupStatistics({keys: view(8), keyCount: 4, columns: [], output: output(5)})
  ).toThrow(/exactly keyCount/);
  expect(
    () =>
      new GPUGroupStatistics({
        keys: pairView(8),
        keyCount: 4,
        columns: [],
        output: {...output(4), keys: pairView(4)}
      })
  ).toThrow(/uint32 keys/);
  expect(
    () => new GPUGroupStatistics({keys: view(8), keyCount: 0, columns: [], output: output(4)})
  ).toThrow(/keyCount/);
  harness.destroy();
});

it('GPUGroupStatistics percentiles follow the shared order-statistic edge policy (B-14)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One group per policy row (key = row index); every distinct probability is one fraction.
  const fractions = [...new Set(ORDER_STATISTIC_EDGE_POLICY.map(row => row.probability))];
  const keys: bigint[] = [];
  const values: number[] = [];
  for (const [index, row] of ORDER_STATISTIC_EDGE_POLICY.entries()) {
    for (const value of row.values) {
      keys.push(BigInt(index));
      values.push(value);
    }
  }
  const harness = new Harness(device);
  const run = await createRun(harness, {
    keyBits: 32,
    keys,
    columns: [{values: Float32Array.from(values), statistics: ['percentiles']}],
    fractions,
    capacity: ORDER_STATISTIC_EDGE_POLICY.length
  });
  try {
    const actual = await run.run();
    const percentiles = actual.columns[0].quantiles!;
    for (const [index, row] of ORDER_STATISTIC_EDGE_POLICY.entries()) {
      const fraction = fractions.findIndex(value => Object.is(value, row.probability));
      const value = percentiles[index * fractions.length + fraction];
      expect(Object.is(value, row.expected), `${row.name}: ${value}`).toBe(true);
    }
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics keeps small moment terms in a dominant group and clears empty tails', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Symmetric data have exactly zero mean and skewness. In the previous strided workgroup
  // reduction the two lanes starting at +/-4096 each lost 4096 additions of 1 to 2^24. The
  // resulting variance error exceeds TOLERANCE.variance, even without GPU fast math.
  const finiteCount = 2 ** 20 + 2;
  const values = Float32Array.from({length: finiteCount + 5}, (_, row) => (row % 2 ? -1 : 1));
  values[0] = 4096;
  values[1] = -4096;
  values[finiteCount] = NaN;
  values[finiteCount + 1] = Infinity;
  values[finiteCount + 2] = NaN;
  const keys = Array<bigint>(values.length).fill(2n);
  keys[finiteCount + 2] = 4n; // A populated group with no finite values.
  keys[finiteCount + 3] = 9n; // A singleton group.
  keys[finiteCount + 4] = 0xffffffffn;
  const scene: Scene = {
    keyBits: 32,
    keys,
    columns: [{values, statistics: ['count', 'mean', 'variance', 'skewness', 'kurtosis']}],
    fractions: [],
    variance: 'population',
    capacity: 5
  };
  const expectedVariance = (2 * 4096 ** 2 + finiteCount - 2) / finiteCount;
  const expectedKurtosis =
    (2 * 4096 ** 4 + finiteCount - 2) / finiteCount / expectedVariance ** 2 - 3;
  const harness = new Harness(device);
  const run = await createRun(harness, scene);
  try {
    run.poison();
    const actual = await run.run();
    expect(actual.count).toBe(3);
    expect(actual.columns[0].counts).toEqual([finiteCount, 0, 1, 0, 0]);
    expect(actual.columns[0].means![0]).toBe(0);
    expectClose(
      actual.columns[0].variances![0],
      expectedVariance,
      TOLERANCE.variance,
      'dominant variance'
    );
    expectClose(actual.columns[0].skewness![0], 0, TOLERANCE.skewness, 'dominant skewness');
    expectClose(
      actual.columns[0].kurtosis![0],
      expectedKurtosis,
      TOLERANCE.kurtosis,
      'dominant kurtosis'
    );
    for (const group of [1, 3, 4]) {
      expect(Number.isNaN(actual.columns[0].variances![group])).toBe(true);
    }
    expect((await run.run()).words).toEqual(actual.words);
    // Reuse scratch with all rows excluded. Every output must describe the new empty table.
    run.maskBuffer.write(new Uint32Array(values.length));
    const empty = await run.run();
    expect(empty.count).toBe(0);
    expect(empty.columns[0].counts).toEqual([0, 0, 0, 0, 0]);
    expect(empty.columns[0].variances!.every(Number.isNaN)).toBe(true);
  } finally {
    run.destroy();
    harness.destroy();
  }
}, 120000);

it('GPUGroupStatistics merges across unaligned group boundaries with dense gaps and overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const keys = Array.from({length: 1031}, (_, row) => BigInt(row < 3 ? 0 : row < 1027 ? 2 : 4));
  const values = Float32Array.from(keys, (_, row) => (row % 7) - 3);
  const harness = new Harness(device);
  try {
    for (const dense of [false, true]) {
      const scene: Scene = {
        keyBits: 32,
        keys,
        columns: [
          {values, statistics: ['count', 'mean', 'variance', 'skewness', 'kurtosis', 'zScore']}
        ],
        fractions: [],
        capacity: dense ? 6 : 2,
        keyCount: dense ? 6 : undefined
      };
      const run = await createRun(harness, scene);
      try {
        run.poison();
        expectMatchesOracle(await run.run(), scene, dense ? 'dense tree' : 'overflow tree');
      } finally {
        run.destroy();
      }
    }
  } finally {
    harness.destroy();
  }
});

it.each(
  [40, 41, 42].flatMap(exponent =>
    [undefined, 128, 256, 512].map(scratchTileSize => ({exponent, scratchTileSize}))
  )
)('GPUGroupStatistics cancels cubic branches at 2^$exponent with tile $scratchTileSize', async ({
  exponent,
  scratchTileSize
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const amplitude = 2 ** exponent;
  const rowCount = 1024;
  // Exactly symmetric input: mean and third central moment are zero, variance is A^2,
  // and its skewness denominator A^3 is finite. The unscaled tree overflows the two
  // same-sign cubic branches, producing Infinity + -Infinity instead of zero.
  const values = Float32Array.from({length: rowCount}, (_, row) =>
    row < rowCount / 2 ? amplitude : -amplitude
  );
  const harness = new Harness(device);
  const run = await createRun(harness, {
    scratchTileSize,
    keyBits: 32,
    keys: Array<bigint>(rowCount).fill(0n),
    columns: [{values, statistics: ['skewness']}],
    fractions: [],
    capacity: 1
  });
  try {
    run.poison();
    const actual = await run.run();
    expect(actual.count).toBe(1);
    expect(actual.columns[0].skewness![0]).toBe(0);
    expect((await run.run()).words).toEqual(actual.words);
    // Scale must update from the new contents, and must preserve small cubic terms.
    // A Bernoulli population with p=1/4 has skewness (1-2p)/sqrt(p(1-p)) = 2/sqrt(3).
    run.valueBuffers[0].write(
      Float32Array.from({length: rowCount}, (_, row) => (row < rowCount / 4 ? 2 ** -35 : 0))
    );
    const small = await run.run();
    expectClose(
      small.columns[0].skewness![0],
      2 / Math.sqrt(3),
      TOLERANCE.skewness,
      'small skewness'
    );
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics chooses signed moment scales independently for each group', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 2048;
  const scene: Scene = {
    keyBits: 32,
    keys: Array.from({length: rowCount}, (_, row) => (row < 1024 ? 0n : 1n)),
    columns: [
      {
        values: Float32Array.from({length: rowCount}, (_, row) =>
          row < 512 ? 2 ** 40 : row < 1024 ? -(2 ** 40) : row < 1280 ? 2 ** -35 : 0
        ),
        statistics: ['skewness']
      }
    ],
    fractions: [],
    capacity: 3
  };
  const harness = new Harness(device);
  const run = await createRun(harness, scene);
  try {
    const actual = await run.run();
    expect(actual.columns[0].skewness![0]).toBe(0);
    expectClose(
      actual.columns[0].skewness![1],
      2 / Math.sqrt(3),
      TOLERANCE.skewness,
      'small group'
    );
    expect(Number.isNaN(actual.columns[0].skewness![2])).toBe(true);
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics chunked keys and independent value chunks match packed bits and the CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  try {
    // Both atomic strategies, 32/64-bit keys, dense gaps and sparse capacity overflow.
    for (const configuration of [
      {keyBits: 64 as const, capacity: 5, keyCount: undefined},
      {keyBits: 32 as const, capacity: 512, keyCount: 512}
    ]) {
      const rows = 1031;
      const scene: Scene = {
        ...configuration,
        keys: Array.from({length: rows}, (_, row) =>
          row % 37 === 0
            ? configuration.keyBits === 64
              ? 0xffffffffffffffffn
              : 0xffffffffn
            : BigInt((row * 17) % 11) + (configuration.keyBits === 64 ? 1n << 40n : 0n)
        ),
        mask: Uint32Array.from({length: rows}, (_, row) => Number(row % 13 !== 0)),
        columns: [
          {
            values: Float32Array.from({length: rows}, (_, row) =>
              row % 23 === 0
                ? NaN
                : row % 29 === 0
                  ? Infinity
                  : row % 7 === 0
                    ? -0
                    : ((row % 31) - 15) / 8
            ),
            statistics: ALL_STATISTICS
          },
          {
            values: Float32Array.from({length: rows}, (_, row) =>
              row < 19 ? 1e-4 : ((row % 5) - 2) * 1e6
            ),
            statistics: ['sum', 'mean', 'variance', 'zScore']
          }
        ],
        fractions: [0, 0.25, 0.5, 1, NaN]
      };
      const packed = await createRun(harness, scene);
      try {
        const expected = await packed.run();
        expectMatchesOracle(expected, scene, 'packed chunk oracle');
        for (const separateBuffers of [false, true]) {
          const chunked = await createRun(harness, scene, {
            keys: [0, 1, 256, 0, rows - 257, 0],
            values: [
              [2, 0, 17, rows - 19, 0],
              [19, 1, 0, rows - 20]
            ],
            separateBuffers
          });
          try {
            const actual = await chunked.run();
            expect(actual.words).toEqual(expected.words);
            expectMatchesOracle(actual, scene, 'chunked oracle');
            chunked.poison();
            expect((await chunked.run()).words).toEqual(expected.words);
          } finally {
            chunked.destroy();
          }
        }
      } finally {
        packed.destroy();
      }
    }
  } finally {
    harness.destroy();
  }
});

it('GPUGroupStatistics streams over a million rows from chunked keys and values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const rows = 2 ** 20 + 17;
  const scene: Scene = {
    keyBits: 32,
    keys: Array.from({length: rows}, (_, row) => BigInt(row % 4)),
    keyCount: 4,
    capacity: 4,
    fractions: [],
    columns: [
      {
        values: Float32Array.from({length: rows}, (_, row) => (row % 4) + 1),
        statistics: ['count', 'sum', 'mean', 'minimum', 'maximum']
      }
    ]
  };
  const run = await createRun(harness, scene, {
    keys: [1, 2 ** 19, rows - 2 ** 19 - 1],
    values: [[2 ** 18 + 3, rows - 2 ** 18 - 3]],
    separateBuffers: true
  });
  try {
    const actual = await run.run();
    for (let group = 0; group < 4; group++) {
      const count = Math.floor(rows / 4) + Number(group < rows % 4);
      expect(actual.counts[group]).toBe(count);
      expect(actual.columns[0].counts![group]).toBe(count);
      expect(actual.columns[0].sumValues![group]).toBe(count * (group + 1));
      expect(actual.columns[0].means![group]).toBe(group + 1);
      expect(actual.columns[0].minimums![group]).toBe(group + 1);
      expect(actual.columns[0].maximums![group]).toBe(group + 1);
    }
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics skips empty tails at allocation end and retains explicit group values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  try {
    for (const keyBits of [32, 64] as const) {
      for (const scratchTileSize of [undefined, 2]) {
        const scene: Scene = {
          keyBits,
          keys: [1n, 0n, 1n],
          capacity: 2,
          fractions: [],
          scratchTileSize,
          columns: [
            {values: Float32Array.of(1, 4, 3), statistics: ['count', 'mean', 'minimum', 'maximum']}
          ]
        };
        const run = await createRun(harness, scene, {
          keys: [0, 1, 2, 0],
          values: [[1, 0, 2, 0]]
        });
        try {
          const actual = await run.run();
          expect(actual.keys).toEqual([0n, 1n]);
          expect(actual.counts).toEqual([1, 2]);
          expect(actual.columns[0].means).toEqual([4, 2]);
          expect(actual.columns[0].minimums).toEqual([4, 1]);
          expect(actual.columns[0].maximums).toEqual([4, 3]);
          expectMatchesOracle(actual, scene, 'empty trailing input chunks');
          scene.columns[0].values.set([2, 6, 4]);
          run.rewriteColumn(0, scene.columns[0].values);
          expectMatchesOracle(await run.run(), scene, 'rewritten empty trailing chunks');
        } finally {
          run.destroy();
        }
      }
    }
  } finally {
    harness.destroy();
  }
});

it('GPUGroupStatistics resets the shared scale from huge to tiny across actual separate chunk buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const rows = 257;
  const scene: Scene = {
    keyBits: 32,
    keys: new Array<bigint>(rows).fill(0n),
    capacity: 1,
    fractions: [],
    columns: [{values: new Float32Array(rows).fill(2 ** 60), statistics: ['sum', 'mean']}]
  };
  const run = await createRun(harness, scene, {
    keys: [1, 0, 255, 1],
    values: [[0, 129, 0, 128]],
    separateBuffers: true
  });
  try {
    for (const magnitude of [2 ** 60, 2 ** -60, 2 ** 60, 2 ** -60]) {
      run.rewriteColumn(0, new Float32Array(rows).fill(magnitude));
      const actual = await run.run();
      expect(actual.counts).toEqual([rows]);
      expect(actual.columns[0].sumValues).toEqual([rows * magnitude]);
      expectClose(
        actual.columns[0].means![0],
        magnitude,
        {relative: 1e-6, absolute: 0},
        'explicit magnitude mean'
      );
    }
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics handles chunk and workgroup boundaries at both accumulator capacities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  try {
    for (const rows of [1, 255, 256, 257]) {
      for (const capacity of [256, 257]) {
        const scene: Scene = {
          keyBits: 32,
          keyCount: capacity,
          capacity,
          fractions: [0, 0.5, 1],
          keys: Array.from({length: rows}, (_, row) => BigInt(row % 3)),
          columns: [
            {
              values: Float32Array.from({length: rows}, (_, row) => (row % 8) + 1),
              statistics: ALL_STATISTICS
            }
          ]
        };
        const valueFirstLength = Math.ceil(rows / 2);
        const run = await createRun(harness, scene, {
          keys: [0, 1, 0, rows - 1, 0],
          values: [[0, valueFirstLength, 0, rows - valueFirstLength, 0]],
          separateBuffers: true
        });
        try {
          const actual = await run.run();
          expectMatchesOracle(actual, scene, `${rows} rows ${capacity} capacity`);
          expect(actual.count).toBe(capacity);
          expect(actual.overflow).toBe(0);
          expect(actual.counts[0]).toBe(Math.ceil(rows / 3));
        } finally {
          run.destroy();
        }
      }
    }
  } finally {
    harness.destroy();
  }
});

it('GPUGroupStatistics preserves every moment of a cancellation-heavy dominant group across chunks and tree levels', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const rows = 2053;
  const scene: Scene = {
    keyBits: 64,
    capacity: 257,
    fractions: [],
    keys: Array.from({length: rows}, (_, row) => (1n << 40n) + BigInt(row % 127 === 0 ? 1 : 0)),
    columns: [
      {
        values: Float32Array.from({length: rows}, (_, row) => [2 ** 20, 1, -(2 ** 20), 2][row % 4]),
        statistics: [
          'count',
          'sum',
          'mean',
          'minimum',
          'maximum',
          'variance',
          'standardDeviation',
          'skewness',
          'kurtosis',
          'zScore'
        ]
      }
    ]
  };
  const packed = await createRun(harness, scene);
  const chunked = await createRun(harness, scene, {
    keys: [1, 254, 2, 257, 0, rows - 514],
    values: [[0, 129, 128, 513, rows - 770, 0]],
    separateBuffers: true
  });
  try {
    const expected = await packed.run();
    const actual = await chunked.run();
    expect(actual.words).toEqual(expected.words);
    expectMatchesOracle(actual, scene, 'dominant cancellation-heavy group');
  } finally {
    packed.destroy();
    chunked.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics orders chunked 64-bit keys larger than a binding and refreshes overflow on later frames', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const originalLimits = device.limits;
  await withGroupStatisticsTestBindingLimit(device, async limitedDevice => {
    // Keep the original device identity: imported buffers and the graph must have the same owner.
    expect(limitedDevice).toBe(device);
    const limits = limitedDevice.limits;
    const harness = new Harness(limitedDevice);
    const rows = 10001;
    const scene: Scene = {
      keyBits: 64,
      capacity: 2,
      fractions: [],
      keys: Array.from({length: rows}, (_, row) => (1n << 40n) + BigInt([7, 2, 5][row % 3])),
      columns: [{values: new Float32Array(rows).fill(2), statistics: ['count', 'sum', 'mean']}]
    };
    let run: Awaited<ReturnType<typeof createRun>> | undefined;
    try {
      expect(rows * 8).toBeGreaterThan(limits.maxStorageBufferBindingSize);
      expect(rows * 4).toBeLessThan(limits.maxStorageBufferBindingSize);
      run = await createRun(harness, scene, {
        keys: [5000, 0, 5001],
        values: [[1, 4000, 6000]],
        separateBuffers: true
      });
      const initial = await run.run();
      expectMatchesOracle(initial, scene, '64-bit chunk binding overflow');
      expect(initial.keys).toEqual([(1n << 40n) + 2n, (1n << 40n) + 5n]);
      expect([initial.count, initial.total, initial.overflow]).toEqual([2, 3, 1]);
      const nextKeys = Array.from({length: rows}, (_, row) => (1n << 41n) + BigInt(row % 2));
      run.rewriteKeys(nextKeys);
      run.rewriteColumn(0, new Float32Array(rows).fill(4));
      const updated = await run.run();
      expectMatchesOracle(
        updated,
        {
          ...scene,
          keys: nextKeys,
          columns: [{...scene.columns[0], values: new Float32Array(rows).fill(4)}]
        },
        '64-bit refreshed chunks'
      );
      expect(updated.keys).toEqual([1n << 41n, (1n << 41n) + 1n]);
      expect([updated.count, updated.total, updated.overflow]).toEqual([2, 2, 0]);
      expect(updated.columns[0].means).toEqual([4, 4]);
      expect((await run.run()).words).toEqual(updated.words);
    } finally {
      run?.destroy();
      harness.destroy();
    }
  });
  expect(device.limits).toBe(originalLimits);
});

it('GPUGroupStatistics rejects co-bound alignment overlap and computes explicit results with separated output bindings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const graph = new GPUCommandGraph(device);
  const columnBuffer = harness.output(65);
  const keyBuffer = harness.output(1);
  const groupCountsBuffer = harness.output(1);
  const statusBuffer = harness.output(1);
  const overflowBuffer = harness.output(1);
  const counts = importGraphBuffer(graph, 'aligned-column-counts', columnBuffer, 'uint32', 1);
  const values = importGraphBuffer(
    graph,
    'aligned-values',
    harness.input(new Float32Array([1, 2, 3])),
    'float32',
    3
  );
  const createContributor = (byteOffset: number) =>
    new GPUGroupStatistics({
      keys: importGraphBuffer(
        graph,
        `aligned-keys-${byteOffset}`,
        harness.input(new Uint32Array([7, 7, 7])),
        'uint32',
        3
      ),
      columns: [
        {
          values: new GraphVectorView({
            id: 'aligned-value-chunks',
            name: 'aligned-value-chunks',
            format: 'float32',
            length: 3,
            valueLength: 3,
            stride: 1,
            byteStride: 4,
            rowByteLength: 4,
            data: [
              graph.createDataView(values.buffer, {format: 'float32', length: 1}),
              graph.createDataView(values.buffer, {format: 'float32', length: 2, byteOffset: 4})
            ]
          }),
          statistics: ['count', 'mean'],
          output: {
            counts,
            means: graph.createDataView(counts.buffer, {format: 'float32', length: 1, byteOffset})
          }
        }
      ],
      output: {
        keys: importGraphBuffer(graph, `aligned-output-keys-${byteOffset}`, keyBuffer, 'uint32', 1),
        counts: importGraphBuffer(
          graph,
          `aligned-output-counts-${byteOffset}`,
          groupCountsBuffer,
          'uint32',
          1
        ),
        status: {
          count: importGraphBuffer(
            graph,
            `aligned-status-${byteOffset}`,
            statusBuffer,
            'uint32',
            1
          ),
          overflow: importGraphBuffer(
            graph,
            `aligned-overflow-${byteOffset}`,
            overflowBuffer,
            'uint32',
            1
          )
        }
      }
    });
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    expect(() => createContributor(4)).toThrow(/aligned binding byte ranges must not overlap/);
    graph.add(createContributor(256));
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const words = await readWords(columnBuffer, 65);
    expect(words[0]).toBe(3);
    expect(new Float32Array(words.buffer)[64]).toBe(2);
    expect(Array.from(await readWords(keyBuffer, 1))).toEqual([7]);
    expect(Array.from(await readWords(groupCountsBuffer, 1))).toEqual([3]);
    expect(Array.from(await readWords(statusBuffer, 1))).toEqual([1]);
    expect(Array.from(await readWords(overflowBuffer, 1))).toEqual([0]);
  } finally {
    compiled?.destroy();
    harness.destroy();
  }
});

it('GPUGroupStatistics keeps non-finite percentile fractions in runtime data and returns explicit missing values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const scene: Scene = {
    keyBits: 32,
    keys: [7n, 7n, 7n, 7n],
    capacity: 1,
    fractions: [0, 0.5, 1, Infinity, -Infinity, NaN],
    columns: [
      {values: new Float32Array([-1, 1, 3, 5]), statistics: ['count', 'mean', 'percentiles']}
    ]
  };
  const run = await createRun(harness, scene, {
    keys: [0, 1, 3],
    values: [[2, 0, 2]],
    separateBuffers: true
  });
  try {
    const actual = await run.run();
    expectMatchesOracle(actual, scene, 'non-finite runtime fractions');
    expect(actual.counts).toEqual([4]);
    expect(actual.columns[0].means).toEqual([2]);
    expect(actual.columns[0].quantiles).toEqual([-1, 2, 5, NaN, NaN, NaN]);
  } finally {
    run.destroy();
    harness.destroy();
  }
});

for (const keyBits of [32, 64] as const)
  it(`fixed group tiles match the independent CPU oracle for ${keyBits}-bit keys and moments`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new Harness(device);
    const statistics: GPUGroupStatistic[] = [
      'count',
      'sum',
      'mean',
      'minimum',
      'maximum',
      'variance',
      'standardDeviation',
      'skewness',
      'kurtosis'
    ];
    const scene = createScene({
      seed: 91,
      keyBits,
      mask: true,
      columns: [statistics, statistics],
      bigGroupSize: 137,
      smallGroups: 3,
      capacity: 5
    });
    scene.scratchTileSize = 64;
    const run = await createRun(harness, scene, {
      keys: [3, 0, 127, scene.keys.length - 130],
      values: [
        [7, scene.keys.length - 7],
        [89, 0, scene.keys.length - 89]
      ],
      separateBuffers: true
    });
    try {
      run.poison();
      expectMatchesOracle(await run.run(), scene, 'fixed tile moments and overflow');
      const changed = Float32Array.from(scene.columns[0].values, value =>
        Math.fround(value * 0.5 + 8)
      );
      run.rewriteColumn(0, changed);
      expectMatchesOracle(
        await run.run(),
        {...scene, columns: [{...scene.columns[0], values: changed}, scene.columns[1]]},
        'rewritten tile values'
      );
      const missing = new Array<bigint>(scene.keys.length).fill(
        keyBits === 64 ? 0xffffffffffffffffn : 0xffffffffn
      );
      run.rewriteKeys(missing);
      expectMatchesOracle(
        await run.run(),
        {
          ...scene,
          keys: missing,
          columns: [{...scene.columns[0], values: changed}, scene.columns[1]]
        },
        'all invalid tiles'
      );
    } finally {
      run.destroy();
      harness.destroy();
    }
  });

it('fixed dense group tiles keep empty groups and ignore out-of-domain keys', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const rows = 259;
  const scene: Scene = {
    keyBits: 32,
    keys: Array.from({length: rows}, (_, row) => BigInt([0, 2, 2, 8, 0xffffffff][row % 5])),
    capacity: 4,
    keyCount: 4,
    scratchTileSize: 32,
    fractions: [],
    columns: [
      {
        values: Float32Array.from({length: rows}, (_, row) => (row % 17) - 8),
        statistics: [
          'count',
          'sum',
          'mean',
          'minimum',
          'maximum',
          'variance',
          'skewness',
          'kurtosis'
        ]
      }
    ]
  };
  const run = await createRun(harness, scene);
  try {
    expectMatchesOracle(await run.run(), scene, 'dense fixed tiles');
  } finally {
    run.destroy();
    harness.destroy();
  }
});

it('group statistics automatically tiles scalar scratch beyond a 64 KiB binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new Harness(device);
  const scene = createScene({
    seed: 97,
    keyBits: 64,
    mask: true,
    bigGroupSize: 20003,
    smallGroups: 3,
    capacity: 5,
    columns: [['count', 'sum', 'mean', 'minimum', 'maximum', 'variance', 'skewness', 'kurtosis']]
  });
  const lengths = (first: number, width: number) => {
    const chunks = [first, 0];
    let remaining = scene.keys.length - first;
    while (remaining > 0) {
      const length = Math.min(remaining, width);
      chunks.push(length);
      remaining -= length;
    }
    return chunks;
  };
  const run = await withGroupStatisticsTestBindingLimit(device, () =>
    createRun(harness, scene, {
      keys: lengths(3, 4096),
      values: [lengths(17, 4000)],
      separateBuffers: true
    })
  );
  try {
    expectMatchesOracle(await run.run(), scene, 'beyond one binding');
  } finally {
    run.destroy();
    harness.destroy();
  }
});
