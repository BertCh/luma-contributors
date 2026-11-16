// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';
import {
  createRasterTimeSeriesChunkCase,
  STANDARDIZED_CASE_QUANTILES,
  getStandardizedCaseFit,
  getStandardizedCaseValue,
  getStandardizedCaseMask,
  getStandardizedCaseStatus,
  RASTER_TIME_SERIES_CHUNK_CASE_NAMES,
  type ChunkCaseHarness,
  type RasterTimeSeriesChunkCaseName
} from './raster-time-series-chunks-b-cases';
import {
  getGPURasterFixedPointValues,
  getGPURasterStreamingAccumulatorParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {GPURasterStandardizedIndex} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {GPURasterTemporal} from '../../../src/raster-analysis/remote-sensing/gpu-raster-temporal';
import {GPURasterRollingMean} from '../../../src/raster-analysis/raster-time-series/gpu-raster-rolling-mean';
import {GPURasterStreamingAccumulator} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {
  GPURasterStandardizedIndexCalibration,
  getGPURasterStandardizedIndexCalibrationParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {GPUChangeDetection} from '../../../src/raster-analysis/change-detection/gpu-change-detection';
import {GPURasterSeasonalBreak} from '../../../src/raster-analysis/remote-sensing/gpu-raster-seasonal-break';

const FLOAT_SENTINEL = -4321;
const UINT_SENTINEL = 0xa5a5a5a5;
function initializeOutput<View extends GraphDataView>(view: View): View {
  const values =
    view.format === 'float32'
      ? new Float32Array(view.length).fill(FLOAT_SENTINEL)
      : new Uint32Array(view.length).fill(UINT_SENTINEL);
  (view.buffer.defaultBuffer as Buffer).write(values);
  return view;
}

function makeVector<Format extends 'float32' | 'uint32'>(
  name: string,
  chunks: GraphDataView<Format>[]
): GraphVectorView<Format> {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  return new GraphVectorView({
    id: name,
    name,
    format: chunks[0].format,
    length,
    valueLength: length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
}

async function runCase(name: RasterTimeSeriesChunkCaseName, chunked: boolean) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const cellCounts = chunked ? [63, 2, 65] : [130];
  let fieldOrdinal = 0;
  const outputChunks = new Map<
    string,
    {key: string; prefix: number; length: number; sentinel: number}[]
  >();
  const fixture: ChunkCaseHarness = {
    cellCount: 130,
    field(fieldName, format, entriesPerCell, value) {
      let cellBase = 0;
      const ordinal = fieldOrdinal++;
      const chunks = cellCounts.map((cellCount, chunkIndex) => {
        const prefix = [1, 65, 3][chunkIndex] + ((ordinal * 7) % 31);
        const length = cellCount * entriesPerCell;
        const key = `${fieldName}-${chunkIndex}`;
        const backing = value
          ? (format === 'float32' ? harness.floats : harness.integers)(
              Array.from({length: prefix + length}, (_, index) =>
                index < prefix
                  ? 0
                  : value(
                      cellBase + Math.floor((index - prefix) / entriesPerCell),
                      (index - prefix) % entriesPerCell
                    )
              )
            )
          : harness.output(key, format, prefix + length + 7);
        if (!value) initializeOutput(backing);
        cellBase += cellCount;
        if (!value) {
          const fields = outputChunks.get(fieldName) ?? [];
          fields.push({
            key,
            prefix,
            length,
            sentinel: format === 'float32' ? FLOAT_SENTINEL : UINT_SENTINEL
          });
          outputChunks.set(fieldName, fields);
        }
        return harness.graph.createDataView(backing.buffer, {
          format,
          byteOffset: prefix * 4,
          length
        });
      });
      return chunked ? makeVector(fieldName, chunks) : chunks[0];
    },
    shared: values => harness.floats(values),
    dates: () => ({
      acquisitionTimes: harness.words(
        BigInt64Array.from([0, 1, 3, 4, 8], year => BigInt(year) * 31556952n)
      ),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    })
  };
  const testCase = createRasterTimeSeriesChunkCase(name, fixture);
  const raw = await harness.run(testCase.contributor);
  for (const chunks of outputChunks.values())
    for (const chunk of chunks) {
      const sentinel = chunk.sentinel;
      expect(
        raw[chunk.key].slice(0, chunk.prefix).every(value => value === sentinel),
        `${chunk.key} prefix`
      ).toBe(true);
      expect(
        raw[chunk.key].slice(chunk.prefix + chunk.length).every(value => value === sentinel),
        `${chunk.key} tail`
      ).toBe(true);
    }
  const actual = Object.fromEntries(
    [...outputChunks].map(([fieldName, chunks]) => [
      fieldName,
      chunks.flatMap(chunk => raw[chunk.key].slice(chunk.prefix, chunk.prefix + chunk.length))
    ])
  );
  return {actual, expected: testCase.expected, cellCount: fixture.cellCount};
}

function expectValues(actual: number[], expected: number[], label: string): void {
  expect(actual.length, label).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(actual[index], `${label}[${index}]`).toBeNaN();
    else if (Number.isInteger(value))
      expect(Math.abs(actual[index] - value), `${label}[${index}]`).toBeLessThanOrEqual(2e-3);
    else expect(actual[index], `${label}[${index}]`).toBeCloseTo(value, 3);
  });
}

describe('SC3-RTS-B chunked raster series GPU oracles', () => {
  for (const name of RASTER_TIME_SERIES_CHUNK_CASE_NAMES) {
    it(`${name}: three unequal chunks match closed form and packed input with nonaligned prefixes`, async () => {
      const chunked = await runCase(name, true);
      const packed = await runCase(name, false);
      for (const [field, expected] of Object.entries(chunked.expected))
        expectValues(chunked.actual[field], expected, `${name}.${field}`);
      for (const [field, expected] of Object.entries(packed.actual))
        expectValues(chunked.actual[field], expected, `${name}.${field} packed parity`);
      if (name === 'streaming') {
        const parameters = getGPURasterStreamingAccumulatorParameterValues(
          {reset: true, maximumValueMagnitude: chunked.cellCount + 10},
          5
        );
        expectValues(
          Array.from(getGPURasterFixedPointValues(chunked.actual['sums'], parameters[2])),
          Array.from({length: chunked.cellCount}, (_, cell) => 15 + 5 * cell),
          'stream sums'
        );
        expectValues(
          Array.from(getGPURasterFixedPointValues(chunked.actual['moments'], parameters[3])),
          Array(chunked.cellCount).fill(10),
          'stream moments'
        );
      }
    });
  }

  it('evaluates all 513 slices with cell-specific fits, signed quantiles, QA, missing values and an inactive tail', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const cellCounts = [2, 3, 4];
    const field = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      entries: number,
      value?: (cell: number, entry: number) => number
    ) => {
      let base = 0;
      const chunks = cellCounts.map((count, chunkIndex) => {
        const view = value
          ? (format === 'float32' ? harness.floats : harness.integers)(
              Array.from({length: count * entries}, (_, index) =>
                value(base + Math.floor(index / entries), index % entries)
              )
            )
          : harness.output(`${name}-${chunkIndex}`, format, count * entries);
        base += count;
        if (!value) initializeOutput(view);
        return harness.graph.createDataView(view.buffer, {format, length: count * entries});
      });
      return makeVector(name, chunks);
    };
    const contributor = new GPURasterStandardizedIndex({
      cellCount: 9,
      chunkSliceCount: 513,
      distribution: 'fisk',
      values: field('values', 'float32', 513, getStandardizedCaseValue),
      mask: field('mask', 'uint32', 513, getStandardizedCaseMask),
      fit: field('fit', 'float32', 3, getStandardizedCaseFit),
      parameters: harness.floats([505, 0]),
      output: {index: field('index', 'float32', 513), status: field('status', 'uint32', 513)}
    });
    const result = await harness.run(contributor);
    let base = 0;
    cellCounts.forEach((count, chunkIndex) => {
      const codes = Array.from({length: count * 513}, (_, index) =>
        getStandardizedCaseStatus(base + Math.floor(index / 513), index % 513, 505)
      );
      const quantiles = codes.map((code, index) =>
        code === 0
          ? STANDARDIZED_CASE_QUANTILES[(base + Math.floor(index / 513) + (index % 513)) % 4]
          : NaN
      );
      expectValues(result[`index-${chunkIndex}`], quantiles, '513 quantiles');
      expect(result[`status-${chunkIndex}`]).toEqual(codes);
      base += count;
    });
  });

  it('dispatches one chunk beyond 65535 workgroups and unequal smaller chunks', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const cellCounts = [4194241, 63, 65];
    const values = makeVector(
      'large-values',
      cellCounts.map((count, chunkIndex) =>
        harness.floats(new Float32Array(count * 2).fill(chunkIndex + 2))
      )
    );
    const output = {
      values: makeVector(
        'mean',
        cellCounts.map((count, index) =>
          initializeOutput(harness.output(`mean-${index}`, 'float32', count))
        )
      ),
      counts: makeVector(
        'counts',
        cellCounts.map((count, index) =>
          initializeOutput(harness.output(`counts-${index}`, 'uint32', count))
        )
      ),
      chosenSlice: makeVector(
        'chosen',
        cellCounts.map((count, index) =>
          initializeOutput(harness.output(`chosen-${index}`, 'uint32', count))
        )
      ),
      status: makeVector(
        'status',
        cellCounts.map((count, index) =>
          initializeOutput(harness.output(`status-${index}`, 'uint32', count))
        )
      )
    };
    const result = await harness.run(
      new GPURasterTemporal({
        cellCount: cellCounts.reduce((total, count) => total + count, 0),
        sliceCount: 2,
        operation: 'mean',
        values,
        output
      })
    );
    cellCounts.forEach((count, chunkIndex) => {
      // Check all rows against a constant CPU oracle without allocating another multi-million-row array.
      expect(result[`mean-${chunkIndex}`].length).toBe(count);
      expect(result[`mean-${chunkIndex}`].every(value => value === chunkIndex + 2)).toBe(true);
      expect(result[`counts-${chunkIndex}`].every(value => value === 2)).toBe(true);
      expect(result[`chosen-${chunkIndex}`].every(value => value === 0xffffffff)).toBe(true);
      expect(result[`status-${chunkIndex}`].every(value => value === 0)).toBe(true);
    });
  }, 60000);
});

/** Distinct prefixes on every field and chunk, with sentinels outside every writable logical view. */
function createGuardedFields(
  harness: ReturnType<typeof createRasterTimeSeriesHarness>,
  cellCounts: number[]
) {
  let ordinal = 0;
  const outputs = new Map<
    string,
    {key: string; prefix: number; length: number; sentinel: number}[]
  >();
  return {
    field<Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      entriesPerCell: number,
      value?: (cell: number, entry: number) => number
    ): GraphVectorView<Format> {
      const fieldOrdinal = ordinal++;
      let cellBase = 0;
      const chunks = cellCounts.map((cellCount, chunkIndex) => {
        const prefix = 1 + ((fieldOrdinal * 11 + chunkIndex * 23) % 79);
        const length = cellCount * entriesPerCell;
        const key = `${name}-${chunkIndex}`;
        const sentinel = format === 'float32' ? FLOAT_SENTINEL : UINT_SENTINEL;
        const backing = value
          ? (format === 'float32' ? harness.floats : harness.integers)(
              Array.from({length: prefix + length + 7}, (_, index) =>
                index >= prefix && index < prefix + length
                  ? value(
                      cellBase + Math.floor((index - prefix) / entriesPerCell),
                      (index - prefix) % entriesPerCell
                    )
                  : sentinel
              )
            )
          : harness.output(key, format, prefix + length + 7);
        if (!value) {
          initializeOutput(backing);
          const entries = outputs.get(name) ?? [];
          entries.push({key, prefix, length, sentinel});
          outputs.set(name, entries);
        }
        cellBase += cellCount;
        return harness.graph.createDataView(backing.buffer, {
          format,
          byteOffset: prefix * 4,
          length
        });
      });
      return makeVector(name, chunks);
    },
    read(result: Record<string, number[]>): Record<string, number[]> {
      return Object.fromEntries(
        [...outputs].map(([name, chunks]) => [
          name,
          chunks.flatMap(chunk => {
            expect(
              result[chunk.key].slice(0, chunk.prefix).every(value => value === chunk.sentinel),
              `${chunk.key} prefix`
            ).toBe(true);
            expect(
              result[chunk.key]
                .slice(chunk.prefix + chunk.length)
                .every(value => value === chunk.sentinel),
              `${chunk.key} tail`
            ).toBe(true);
            return result[chunk.key].slice(chunk.prefix, chunk.prefix + chunk.length);
          })
        ])
      );
    }
  };
}

function calculateMoments(values: number[]) {
  const sum = values.reduce((total, value) => total + value, 0);
  const mean = sum / values.length;
  return {sum, mean, moment: values.reduce((total, value) => total + (value - mean) ** 2, 0)};
}

const UPDATE_CELL_COUNTS = [63, 2, 65];
const UPDATE_CELL_COUNT = UPDATE_CELL_COUNTS.reduce((total, count) => total + count, 0);
function getUpdateValue(update: number, cell: number, slice: number): number {
  return cell % 3 === 2 && update === 1 && slice === 0 ? NaN : cell / 4 + update * 3 + slice + 1;
}
function getUpdateMask(update: number, cell: number, slice: number): number {
  return cell % 3 === 1 && update === 0 && slice === 1 ? 0 : 1;
}
function getAcceptedUpdateValues(cell: number): number[] {
  return [0, 1]
    .flatMap(update =>
      Array.from({length: 3}, (_, slice) =>
        getUpdateMask(update, cell, slice) ? getUpdateValue(update, cell, slice) : NaN
      )
    )
    .filter(Number.isFinite);
}

describe('SC3-RTS-B reviewer headless regressions', () => {
  it('carries rolling rings and seen counts through reset=false with independent window expectations', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fields = createGuardedFields(harness, UPDATE_CELL_COUNTS);
    const ring = fields.field('ring', 'float32', 3);
    const seenCount = fields.field('seen', 'uint32', 1);
    const contributors = [0, 1].map(
      update =>
        new GPURasterRollingMean({
          id: `rolling-update-${update}`,
          cellCount: UPDATE_CELL_COUNT,
          chunkSliceCount: 3,
          windowLength: 3,
          values: fields.field(`values-${update}`, 'float32', 3, (cell, slice) =>
            getUpdateValue(update, cell, slice)
          ),
          mask: fields.field(`mask-${update}`, 'uint32', 3, (cell, slice) =>
            getUpdateMask(update, cell, slice)
          ),
          parameters: harness.floats([update === 0 ? 1 : 0, 3, 0, 0, 128]),
          ring,
          seenCount,
          output: fields.field(`rolling-${update}`, 'float32', 3)
        })
    );
    const result = fields.read(await harness.run(contributors));
    for (const update of [0, 1]) {
      const expected = Array.from({length: UPDATE_CELL_COUNT}, (_, cell) => {
        const samples = [0, 1].flatMap(round =>
          Array.from({length: 3}, (_, slice) =>
            getUpdateMask(round, cell, slice) ? getUpdateValue(round, cell, slice) : NaN
          )
        );
        return Array.from({length: 3}, (_, slice) => {
          const end = update * 3 + slice + 1;
          const window = samples.slice(Math.max(0, end - 3), end);
          return window.length === 3 && window.every(Number.isFinite)
            ? window.reduce((sum, value) => sum + value, 0) / 3
            : NaN;
        });
      }).flat();
      expectValues(result[`rolling-${update}`], expected, `rolling update ${update}`);
    }
    expect(result['seen']).toEqual(Array(UPDATE_CELL_COUNT).fill(6));
    expectValues(
      result['ring'],
      Array.from({length: UPDATE_CELL_COUNT}, (_, cell) =>
        Array.from({length: 3}, (_, slice) => getUpdateValue(1, cell, slice))
      ).flat(),
      'carried ring'
    );
  });

  it('carries streamed counts, exact sums, moments and extrema through reset=false', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fields = createGuardedFields(harness, UPDATE_CELL_COUNTS);
    const state = {
      counts: fields.field('counts', 'uint32', 1),
      sums: fields.field('sums', 'uint32', 3),
      moments: fields.field('moments', 'uint32', 3),
      extrema: fields.field('extrema', 'float32', 2)
    };
    const settings = {reset: true, maximumValueMagnitude: 128};
    const parameters = getGPURasterStreamingAccumulatorParameterValues(settings, 3);
    const contributors = [0, 1].map(
      update =>
        new GPURasterStreamingAccumulator({
          id: `stream-update-${update}`,
          cellCount: UPDATE_CELL_COUNT,
          chunkSliceCount: 3,
          values: fields.field(`values-${update}`, 'float32', 3, (cell, slice) =>
            getUpdateValue(update, cell, slice)
          ),
          mask: fields.field(`mask-${update}`, 'uint32', 3, (cell, slice) =>
            getUpdateMask(update, cell, slice)
          ),
          parameters: harness.floats(
            getGPURasterStreamingAccumulatorParameterValues({...settings, reset: update === 0}, 3)
          ),
          state
        })
    );
    const result = fields.read(await harness.run(contributors));
    const values = Array.from({length: UPDATE_CELL_COUNT}, (_, cell) =>
      getAcceptedUpdateValues(cell)
    );
    expect(result['counts']).toEqual(values.map(samples => samples.length));
    expectValues(
      Array.from(getGPURasterFixedPointValues(result['sums'], parameters[2])),
      values.map(samples => calculateMoments(samples).sum),
      'streamed sums'
    );
    expectValues(
      Array.from(getGPURasterFixedPointValues(result['moments'], parameters[3])),
      values.map(samples => calculateMoments(samples).moment),
      'streamed moments'
    );
    expectValues(
      result['extrema'],
      values.flatMap(samples => [Math.min(...samples), Math.max(...samples)]),
      'streamed extrema'
    );
  });

  it('carries two-slot calibration counts, all moment words and fits through reset=false', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fields = createGuardedFields(harness, [2, 1, 3]);
    const value = (update: number, cell: number, slice: number) => {
      if (cell === 1 && update === 1 && slice === 1) return NaN;
      if (cell === 2 && update === 0 && slice === 0) return 0;
      return (1 + cell / 4) * (update * 2 + Math.floor(slice / 2) + 1) * ((slice % 2) + 1);
    };
    const mask = (update: number, cell: number, slice: number) =>
      cell === 0 && update === 1 && slice === 2 ? 0 : 1;
    const state = {
      counts: fields.field('counts', 'uint32', 6),
      moments: fields.field('moments', 'uint32', 18)
    };
    const fit = fields.field('fit', 'float32', 6);
    const settings = {reset: true, maximumValueMagnitude: 128};
    const parameters = getGPURasterStandardizedIndexCalibrationParameterValues(settings, 4);
    const slotIndices = harness.integers([0, 1, 0, 1]);
    const contributors = [0, 1].map(
      update =>
        new GPURasterStandardizedIndexCalibration({
          id: `calibration-update-${update}`,
          cellCount: 6,
          chunkSliceCount: 4,
          slotCount: 2,
          distribution: 'fisk',
          zeroInflated: true,
          values: fields.field(`values-${update}`, 'float32', 4, (cell, slice) =>
            value(update, cell, slice)
          ),
          mask: fields.field(`mask-${update}`, 'uint32', 4, (cell, slice) =>
            mask(update, cell, slice)
          ),
          parameters: harness.floats(
            getGPURasterStandardizedIndexCalibrationParameterValues(
              {...settings, reset: update === 0},
              4
            )
          ),
          slotIndices,
          state,
          fit
        })
    );
    const result = fields.read(await harness.run(contributors));
    const counts: number[] = [];
    const fits: number[] = [];
    for (let cell = 0; cell < 6; cell++)
      for (let slot = 0; slot < 2; slot++) {
        const samples = [0, 1]
          .flatMap(update =>
            [slot, slot + 2]
              .filter(slice => mask(update, cell, slice))
              .map(slice => value(update, cell, slice))
          )
          .filter(Number.isFinite);
        const positive = samples.filter(sample => sample > 0);
        const zeros = samples.filter(sample => sample === 0).length;
        const moments = calculateMoments(positive);
        const variance = moments.moment / positive.length;
        counts.push(samples.length, zeros, positive.length);
        fits.push(
          (Math.PI * moments.mean) / Math.sqrt(3 * variance),
          (2 * moments.mean ** 3) / (variance + 2 * moments.mean ** 2),
          zeros / samples.length
        );
        const base = (cell * 2 + slot) * 9;
        const expectedMoments = [
          moments.sum,
          positive.reduce((sum, sample) => sum + Math.log(sample), 0),
          moments.moment
        ];
        for (let component = 0; component < 3; component++) {
          const decoded = getGPURasterFixedPointValues(
            result['moments'].slice(base + 3 * component, base + 3 * component + 3),
            parameters[3 + component]
          );
          expectValues(
            Array.from(decoded),
            [expectedMoments[component]],
            `calibration moments ${cell}/${slot}/${component}`
          );
        }
      }
    expect(result['counts']).toEqual(counts);
    expectValues(result['fit'], fits, 'carried fits');
  });

  it('checks every multiband change output with cell masks, observation QA and a missing band', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fields = createGuardedFields(harness, [63, 2, 65]);
    const sample = (cell: number, slice: number, band: number) =>
      cell % 5 === 3 && slice === 0 && band === 1
        ? NaN
        : cell + (slice === 0 ? [2, 4, 8] : [4, 2, 9])[band];
    const output = {
      difference: fields.field('difference', 'float32', 3),
      logRatio: fields.field('logRatio', 'float32', 3),
      percentChange: fields.field('percentChange', 'float32', 3),
      changeMagnitude: fields.field('magnitude', 'float32', 1),
      changeDirection: fields.field('direction', 'uint32', 1)
    };
    const contributor = new GPUChangeDetection({
      cellCount: UPDATE_CELL_COUNT,
      sliceCount: 2,
      bandCount: 3,
      slices: fields.field('slices', 'float32', 6, (cell, entry) =>
        sample(cell, Math.floor(entry / 3), entry % 3)
      ),
      mask: fields.field('mask', 'uint32', 1, cell => (cell % 5 === 1 ? 0 : 1)),
      observationMask: fields.field('observationMask', 'uint32', 2, (cell, slice) =>
        cell % 5 === 2 && slice === 1 ? 0 : 1
      ),
      parameters: harness.floats([0, 1, 0, 0.05, 1]),
      output
    });
    const result = fields.read(await harness.run(contributor));
    const difference: number[] = [];
    const logarithm: number[] = [];
    const percentage: number[] = [];
    const magnitude: number[] = [];
    const direction: number[] = [];
    for (let cell = 0; cell < UPDATE_CELL_COUNT; cell++) {
      let complete = true;
      for (let band = 0; band < 3; band++) {
        const before = sample(cell, 0, band);
        const after = sample(cell, 1, band);
        const valid =
          cell % 5 !== 1 && cell % 5 !== 2 && Number.isFinite(before) && Number.isFinite(after);
        complete = complete && valid;
        difference.push(valid ? after - before : NaN);
        logarithm.push(valid ? Math.log(after / before) : NaN);
        percentage.push(valid ? (100 * (after - before)) / Math.abs(before) : NaN);
      }
      magnitude.push(complete ? 3 : NaN);
      direction.push(complete ? 131077 : 0xffffffff);
    }
    expectValues(result['difference'], difference, 'band differences');
    expectValues(result['logRatio'], logarithm, 'band logarithms');
    expectValues(result['percentChange'], percentage, 'band percentages');
    expectValues(result['magnitude'], magnitude, 'change magnitude');
    expect(result['direction']).toEqual(direction);
  });

  it('publishes iterative seasonal fits and every diagnostic when the iteration limit is reached', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fields = createGuardedFields(harness, [2, 1, 3]);
    const sample = (cell: number, slice: number) =>
      10 +
      cell +
      0.5 * slice +
      Math.cos((Math.PI * slice) / 2) +
      2 * Math.sin((Math.PI * slice) / 2);
    const output = {
      coefficients: fields.field('coefficients', 'float32', 5),
      components: fields.field('components', 'float32', 36),
      breakIndex: fields.field('breakIndex', 'uint32', 1),
      improvement: fields.field('improvement', 'float32', 1),
      status: fields.field('status', 'uint32', 1),
      diagnostics: fields.field('diagnostics', 'float32', 6)
    };
    const contributor = new GPURasterSeasonalBreak({
      cellCount: 6,
      sliceCount: 12,
      values: fields.field('values', 'float32', 12, sample),
      period: 4,
      minimumSegmentLength: 3,
      fitMethod: 'iterative',
      maximumIterations: 1,
      tolerance: 1e-20,
      minimumBICGain: 1e6,
      output
    });
    const result = fields.read(await harness.run(contributor));
    const coefficients: number[] = [];
    const components: number[] = [];
    const diagnostics: number[] = [];
    for (let cell = 0; cell < 6; cell++) {
      // Independent one-sweep coordinate descent oracle; the large BIC threshold keeps the baseline model.
      const values = Array.from({length: 12}, (_, slice) => Math.fround(sample(cell, slice)));
      const columns = [
        Array(12).fill(1),
        Array.from({length: 12}, (_, slice) => slice / 12),
        Array.from({length: 12}, (_, slice) => Math.cos((Math.PI * slice) / 2)),
        Array.from({length: 12}, (_, slice) => Math.sin((Math.PI * slice) / 2))
      ];
      const residual = [...values];
      const fitted = columns.map(column => {
        const coefficient =
          column.reduce((sum, value, index) => sum + value * residual[index], 0) /
          column.reduce((sum, value) => sum + value * value, 0);
        column.forEach((value, index) => {
          residual[index] -= value * coefficient;
        });
        return coefficient;
      });
      coefficients.push(...fitted, 0);
      for (let slice = 0; slice < 12; slice++) {
        const trend = fitted[0] + (fitted[1] * slice) / 12;
        const seasonal = fitted[2] * columns[2][slice] + fitted[3] * columns[3][slice];
        components.push(trend, seasonal, values[slice] - trend - seasonal);
      }
      diagnostics.push(
        1,
        0,
        12,
        Math.sqrt(residual.reduce((sum, value) => sum + value * value, 0) / 8),
        -1,
        0
      );
    }
    expectValues(result['coefficients'], coefficients, 'iteration-limited coefficients');
    expectValues(result['components'], components, 'iteration-limited components');
    expectValues(result['diagnostics'], diagnostics, 'iteration-limit diagnostics');
    expect(result['status']).toEqual(Array(6).fill(8));
    expect(result['breakIndex']).toEqual(Array(6).fill(0xffffffff));
    expectValues(result['improvement'], Array(6).fill(0), 'iteration-limit gain');
  });
});

it('writes independent packed difference and Welch statistics in one alignment window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const backing = initializeOutput(harness.output('statistics', 'float32', 8));
  const difference = harness.graph.createDataView(backing.buffer, {format: 'float32', length: 3});
  const tStatistic = harness.graph.createDataView(backing.buffer, {
    format: 'float32',
    byteOffset: 12,
    length: 3
  });
  const result = await harness.run(
    new GPUChangeDetection({
      cellCount: 3,
      sliceCount: 4,
      slices: harness.floats([1, 3, 4, 6, 4, 6, 2, 4, -3, -1, 1, 3]),
      parameters: harness.floats([0, 3, 0, 0.05, 2]),
      output: {difference, tStatistic}
    })
  );
  expectValues(result['statistics'].slice(0, 3), [5, 0, 6], 'differences');
  // Both groups have two observations and sample variance 2, so the standard error is sqrt(2).
  expectValues(
    result['statistics'].slice(3, 6),
    [3, -2, 4].map(value => value / Math.sqrt(2)),
    'Welch statistics'
  );
  expect(result['statistics'].slice(6)).toEqual([FLOAT_SENTINEL, FLOAT_SENTINEL]);
});

for (const extended of [false, true]) {
  it(`compiles ${extended ? 'extended' : 'default'} seasonal fits with minimumBICGain 1e20 and selects no break`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fields = createGuardedFields(harness, [1, 1, 1]);
    const sample = (cell: number, slice: number) =>
      10 +
      cell +
      0.5 * slice +
      Math.cos((Math.PI * slice) / 2) +
      2 * Math.sin((Math.PI * slice) / 2);
    const contributor = new GPURasterSeasonalBreak({
      cellCount: 3,
      sliceCount: 12,
      period: 4,
      minimumSegmentLength: 3,
      minimumBICGain: 1e20,
      values: fields.field('values', 'float32', 12, sample),
      ...(extended ? {fitMethod: 'joint' as const} : {}),
      output: {
        coefficients: fields.field('coefficients', 'float32', 5),
        components: fields.field('components', 'float32', 36),
        breakIndex: fields.field('breakIndex', 'uint32', 1),
        improvement: fields.field('improvement', 'float32', 1),
        status: fields.field('status', 'uint32', 1),
        ...(extended ? {diagnostics: fields.field('diagnostics', 'float32', 6)} : {})
      }
    });
    const result = fields.read(await harness.run(contributor));
    expect(result['breakIndex']).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    expect(result['status']).toEqual([0, 0, 0]);
    expect(result['improvement']).toEqual([0, 0, 0]);
    expectValues(
      result['coefficients'],
      [0, 1, 2].flatMap(cell => [10 + cell, 6, 1, 2, 0]),
      'seasonal coefficients'
    );
    expectValues(
      result['components'],
      [0, 1, 2].flatMap(cell =>
        Array.from({length: 12}, (_, slice) => [
          10 + cell + 0.5 * slice,
          Math.cos((Math.PI * slice) / 2) + 2 * Math.sin((Math.PI * slice) / 2),
          0
        ]).flat()
      ),
      'seasonal components'
    );
    if (extended)
      expectValues(
        result['diagnostics'],
        [0, 1, 2].flatMap(cell => {
          const energy = Array.from(
            {length: 12},
            (_, slice) => Math.fround(sample(cell, slice)) ** 2
          ).reduce((sum, value) => sum + value, 0);
          return [1, 1, 12, Math.sqrt((energy * 1e-12) / 8), -1, 0];
        }),
        'seasonal diagnostics'
      );
  });
}

for (const distribution of ['gamma', 'fisk'] as const) {
  it(`${distribution}: preserves standardized quantiles with a near-maximum float32 scale in three chunks`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const scales = [Math.fround(3.402823466e38), 2 ** -120, 1];
    const values = scales.map(scale => harness.floats([scale / 4, scale / 2, scale]));
    const fits = scales.map(scale => harness.floats([distribution === 'gamma' ? 1 : 2, scale, 0]));
    const indices = scales.map((_, cell) =>
      initializeOutput(harness.output(`index-${cell}`, 'float32', 3))
    );
    const statuses = scales.map((_, cell) =>
      initializeOutput(harness.output(`status-${cell}`, 'uint32', 3))
    );
    const result = await harness.run(
      new GPURasterStandardizedIndex({
        cellCount: 3,
        chunkSliceCount: 3,
        distribution,
        values: makeVector('scaled-observations', values),
        fit: makeVector('scaled-fits', fits),
        parameters: harness.floats([3, 0]),
        output: {
          index: makeVector('scaled-quantiles', indices),
          status: makeVector('scaled-statuses', statuses)
        }
      })
    );
    // SciPy ndtri of the closed forms: gamma(shape=1) CDF = -expm1(-r),
    // Fisk(shape=2) CDF = r²/(1+r²), at r = 1/4, 1/2, 1. Every uploaded input is exact here.
    const expected =
      distribution === 'gamma'
        ? [-0.768149395303851, -0.27028802073873587, 0.33747496376420244]
        : [-1.5647264713617988, -0.8416212335729142, 0];
    for (let cell = 0; cell < scales.length; cell++) {
      expect(result[`status-${cell}`]).toEqual([0, 0, 0]);
      result[`index-${cell}`].forEach((value, sampleIndex) =>
        expect(value).toBeCloseTo(expected[sampleIndex], 4)
      );
    }
  });
}

for (const distribution of ['gamma', 'fisk'] as const) {
  it(`${distribution}: marks nonpositive shape or scale as unusable calibration`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const fits = [
      [0, 1, 0],
      [1, 0, 0],
      [-1, 1, 0],
      [1, -1, 0],
      [NaN, 1, 0],
      [1, NaN, 0]
    ];
    const result = await harness.run(
      new GPURasterStandardizedIndex({
        cellCount: fits.length,
        chunkSliceCount: 1,
        distribution,
        values: harness.floats(new Array(fits.length).fill(1)),
        fit: harness.floats(fits.flat()),
        parameters: harness.floats([1, 0]),
        output: {
          index: initializeOutput(harness.output('index', 'float32', fits.length)),
          status: initializeOutput(harness.output('status', 'uint32', fits.length))
        }
      })
    );
    expect(result['status']).toEqual(new Array(fits.length).fill(2));
    expect(result['index'].every(Number.isNaN)).toBe(true);
  });
}
