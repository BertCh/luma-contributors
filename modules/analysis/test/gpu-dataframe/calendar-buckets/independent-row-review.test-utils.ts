// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCalendarBuckets} from '../../../src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets';
import {GPUTimeWordsRelative} from '../../../src/dataframe-analysis/time-words-relative/gpu-time-words-relative';
import {GPUPeriodIndex} from '../../../src/dataframe-analysis/period-index/gpu-period-index';
import {GPUCellPeriodKeys} from '../../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {GPUClassAssignment} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {GPUFittingCoordinateEncoder} from '../../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {GPUForestInference} from '../../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/time-words';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';

export const REVIEW_CASES = [
  'calendar',
  'relative',
  'period',
  'keys',
  'classes',
  'fourier',
  'harmonics',
  'forest-row-major',
  'forest-feature-major',
  'forest-raster'
] as const;
export type ReviewCase = (typeof REVIEW_CASES)[number];
type Format = 'float32' | 'uint32' | 'uint32x2' | 'sint32';
type Values = Float32Array | Uint32Array | Int32Array;
type Field = {buffer: Buffer; prefix: number; count: number; storage: Values; format: Format};
const WORD_CONTEXT = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
} as const;
const NO_VALUE = 0xffffffff;
const CALENDAR_ORIGIN = Date.UTC(2024, 0, 1);

/** Guarded host-style buffers; logical vector chunks may share an alignment window. */
export function createReviewScenario(
  device: Device,
  kind: ReviewCase,
  lengths: readonly number[],
  separateOutputBuffers = false
) {
  const graph = new GPUCommandGraph(device);
  const rows = lengths.reduce((sum, length) => sum + length, 0);
  const buffers: Buffer[] = [];
  const outputs = new Map<string, Field[]>();
  const updates: ((encoding: number) => void)[] = [];
  let serial = 0;
  const keep = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const maskValues = (encoding: number) =>
    Array.from({length: rows}, (_, row) => Number((row + encoding * 2) % 7 !== 0));
  const isKept = (row: number, encoding: number) => (row + encoding * 2) % 7 !== 0;
  function allocate<SelectedFormat extends Format>(
    name: string,
    format: SelectedFormat,
    values: number[],
    prefix: number,
    count: number
  ) {
    const sentinel = format === 'float32' || format === 'sint32' ? -12345 : 0xdeadbeef;
    const storage =
      format === 'float32'
        ? new Float32Array(values.length + prefix + 4).fill(sentinel)
        : format === 'sint32'
          ? new Int32Array(values.length + prefix + 4).fill(sentinel)
          : new Uint32Array(values.length + prefix + 4).fill(sentinel);
    storage.set(values, prefix);
    const buffer = keep(
      device.createBuffer({
        data: storage,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      })
    );
    const imported = importGraphBuffer(graph, `${name}-${serial++}`, buffer, format);
    const view = graph.createDataView(imported.buffer, {
      format,
      length: count,
      byteOffset: prefix * 4
    });
    return {view, field: {buffer, prefix, count: values.length, storage, format}};
  }
  function packed<SelectedFormat extends Format>(
    name: string,
    format: SelectedFormat,
    values: number[]
  ) {
    return allocate(name, format, values, 3, values.length / (format === 'uint32x2' ? 2 : 1)).view;
  }
  function input<SelectedFormat extends Format>(
    name: string,
    format: SelectedFormat,
    width: number,
    getValues: (encoding: number) => number[]
  ) {
    const {view, field} = allocate(
      name,
      format,
      getValues(0),
      1,
      (rows * width) / (format === 'uint32x2' ? 2 : 1)
    );
    updates.push(encoding => {
      field.storage.set(getValues(encoding), field.prefix);
      field.buffer.write(field.storage);
    });
    return split(view, width);
  }
  function split<SelectedFormat extends Format>(
    view: GraphDataView<SelectedFormat>,
    width: number
  ) {
    let rowBase = 0;
    const components = view.format === 'uint32x2' ? 2 : 1;
    return createVectorView(
      view.buffer.id,
      view.format,
      lengths.map(length => {
        const chunk = graph.createDataView(view.buffer, {
          format: view.format,
          byteOffset: view.byteOffset + rowBase * width * 4,
          length: (length * width) / components
        });
        rowBase += length;
        return chunk;
      })
    );
  }
  function output<SelectedFormat extends Format>(name: string, format: SelectedFormat, width = 1) {
    const components = format === 'uint32x2' ? 2 : 1;
    if (!separateOutputBuffers) {
      const {view, field} = allocate(
        name,
        format,
        Array<number>(rows * width).fill(format === 'float32' ? -999 : 999),
        3,
        (rows * width) / components
      );
      outputs.set(name, [field]);
      return split(view, width);
    }
    const parts = lengths.map((length, index) =>
      allocate(
        name,
        format,
        Array<number>(length * width).fill(format === 'float32' ? -999 : 999),
        1 + (index % 4),
        (length * width) / components
      )
    );
    outputs.set(
      name,
      parts.map(part => part.field)
    );
    return createVectorView(
      name,
      format,
      parts.map(part => part.view)
    );
  }
  function scalarOutput(name: string, length: number) {
    const {view, field} = allocate(name, 'uint32', Array<number>(length).fill(999), 2, length);
    outputs.set(name, [field]);
    return view;
  }
  const words = (ticks: bigint[]) => Array.from(getInt64TimeWords(BigInt64Array.from(ticks)));
  const mask = kind === 'relative' ? undefined : input('mask', 'uint32', 1, maskValues);
  let contributor: GPUCommandNodeProducer & {readonly props: {id?: string}};
  let expected: (encoding: number) => Record<string, number[]>;
  if (kind === 'calendar') {
    contributor = new GPUCalendarBuckets({
      timestamps: input('timestamps', 'uint32x2', 2, () =>
        words(Array.from({length: rows}, (_, row) => BigInt(CALENDAR_ORIGIN + row * 3_600_000)))
      ),
      mask,
      parameters: packed('parameters', 'sint32', [0, 0]),
      output: {
        year: output('year', 'sint32'),
        hour: output('hour', 'uint32'),
        hourWeekdayCounts: scalarOutput('counts', 168)
      }
    });
    expected = encoding => {
      const counts = Array<number>(168).fill(0);
      const year: number[] = [],
        hour: number[] = [];
      for (let row = 0; row < rows; row++) {
        const date = new Date(CALENDAR_ORIGIN + row * 3_600_000);
        const kept = isKept(row, encoding);
        year.push(kept ? date.getUTCFullYear() : -2147483648);
        hour.push(kept ? date.getUTCHours() : NO_VALUE);
        if (kept) counts[((date.getUTCDay() + 6) % 7) * 24 + date.getUTCHours()]++;
      }
      return {year, hour, counts};
    };
  } else if (kind === 'relative') {
    contributor = new GPUTimeWordsRelative({
      timestamps: input('timestamps', 'uint32x2', 2, encoding =>
        words(Array.from({length: rows}, (_, row) => BigInt(1000 + row * 3 - 500 + encoding * 17)))
      ),
      output: output('relative', 'float32'),
      temporalContext: WORD_CONTEXT,
      origin: 1000n
    });
    expected = encoding => ({
      relative: Array.from({length: rows}, (_, row) => row * 3 - 500 + encoding * 17)
    });
  } else if (kind === 'period') {
    contributor = new GPUPeriodIndex({
      timestamps: input('timestamps', 'uint32x2', 2, () =>
        words(Array.from({length: rows}, (_, row) => BigInt(row - 20)))
      ),
      mask,
      temporalContext: WORD_CONTEXT,
      periods: {kind: 'fixed', origin: 0n, width: 3n, count: 100},
      output: {
        periods: output('periods', 'uint32'),
        status: {invalidCount: scalarOutput('invalid', 1)}
      }
    });
    expected = encoding => {
      const periods = Array.from({length: rows}, (_, row) =>
        !isKept(row, encoding)
          ? NO_VALUE
          : row < 20 || row >= 320
            ? 0xfffffffe
            : Math.floor((row - 20) / 3)
      );
      return {periods, invalid: [periods.filter(period => period === 0xfffffffe).length]};
    };
  } else if (kind === 'keys') {
    const cells = Array.from({length: rows}, (_, row) =>
      row % 19 === 0 ? [NO_VALUE, NO_VALUE] : [row * 4 + (row % 11 === 0 ? 1 : 3), 7 + (row % 5)]
    ).flat();
    const periods = Array.from({length: rows}, (_, row) => (row % 17 === 0 ? NO_VALUE : row % 6));
    contributor = new GPUCellPeriodKeys({
      cells: input('cells', 'uint32x2', 2, () => cells),
      periods: input('periods', 'uint32', 1, () => periods),
      mask,
      periodCount: 4,
      fill: 'ones',
      output: {
        keys: output('keys', 'uint32x2', 2),
        status: {invalidCount: scalarOutput('invalid', 1)}
      }
    });
    expected = encoding => {
      let invalid = 0;
      const keys = Array.from({length: rows}, (_, row) => {
        if (!isKept(row, encoding) || row % 19 === 0 || periods[row] === NO_VALUE)
          return [NO_VALUE, NO_VALUE];
        if (periods[row] >= 4 || cells[row * 2] % 4 !== 3) {
          invalid++;
          return [NO_VALUE, NO_VALUE];
        }
        return [row * 4 + periods[row], cells[row * 2 + 1]];
      }).flat();
      return {keys, invalid: [invalid]};
    };
  } else if (kind === 'classes') {
    const values = Array.from({length: rows}, (_, row) =>
      row % 13 === 0 ? NaN : row % 31 === 0 ? Infinity : (row % 9) - 3
    );
    contributor = new GPUClassAssignment({
      values: input('values', 'float32', 1, () => values),
      mask,
      breaks: packed('breaks', 'float32', [0, 1, 2, 3]),
      classCount: packed('class-count', 'uint32', [3]),
      output: output('classes', 'uint32')
    });
    expected = encoding => ({
      classes: values.map((value, row) =>
        !isKept(row, encoding) || Number.isNaN(value)
          ? NO_VALUE
          : Number(value >= 1) + Number(value >= 2)
      )
    });
  } else if (kind === 'fourier' || kind === 'harmonics') {
    const coordinates = Array.from({length: rows}, (_, row) =>
      row % 17 === 0
        ? [NaN, 0]
        : kind === 'fourier'
          ? [(row % 5) / 4, (Math.floor(row / 5) % 5) / 4]
          : [(row % 360) - 180, (row % 179) - 89]
    );
    const common = {
      coordinates: input('coordinates', 'float32', 2, () => coordinates.flat()),
      rowCount: rows,
      mask,
      output: {features: output('features', 'float32', 4), validity: output('validity', 'uint32')}
    };
    contributor = new GPUFittingCoordinateEncoder(
      kind === 'fourier'
        ? {
            ...common,
            encoding: 'fourier',
            dimensions: 2,
            frequencyCount: 2,
            frequencies: packed('frequencies', 'float32', [1, 0, 0, 1])
          }
        : {
            ...common,
            encoding: 'spherical-harmonics',
            maximumDegree: 1,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
          }
    );
    expected = encoding => ({
      validity: coordinates.map((point, row) =>
        Number(isKept(row, encoding) && Number.isFinite(point[0]))
      ),
      features: coordinates.flatMap(([first, second], row) => {
        if (!isKept(row, encoding) || !Number.isFinite(first)) return [0, 0, 0, 0];
        if (kind === 'fourier')
          return [
            Math.cos(2 * Math.PI * first),
            Math.sin(2 * Math.PI * first),
            Math.cos(2 * Math.PI * second),
            Math.sin(2 * Math.PI * second)
          ];
        const longitude = (first * Math.PI) / 180,
          latitude = (second * Math.PI) / 180,
          scale = Math.sqrt(3 / (4 * Math.PI));
        return [
          1 / Math.sqrt(4 * Math.PI),
          -scale * Math.cos(latitude) * Math.sin(longitude),
          scale * Math.sin(latitude),
          -scale * Math.cos(latitude) * Math.cos(longitude)
        ];
      })
    });
  } else {
    const points = Array.from({length: rows}, (_, row) => [row % 17 === 0 ? NaN : row, row % 5]);
    const floatBits = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0];
    const nodes = [
      0,
      1,
      2,
      floatBits(256),
      3,
      0,
      NO_VALUE,
      0,
      0,
      0,
      0,
      11,
      NO_VALUE,
      0,
      0,
      0,
      0,
      12,
      1,
      4,
      5,
      floatBits(2),
      3,
      0,
      NO_VALUE,
      0,
      0,
      0,
      0,
      21,
      NO_VALUE,
      0,
      0,
      0,
      0,
      22
    ];
    const flat = () => {
      if (kind !== 'forest-feature-major') return points.flat();
      let rowBase = 0;
      return lengths.flatMap(length => {
        const batch = points.slice(rowBase, rowBase + length);
        rowBase += length;
        return [batch.map(point => point[0]), batch.map(point => point[1])].flat();
      });
    };
    contributor = new GPUForestInference({
      model: {
        featureCount: 2,
        treeCount: 2,
        nodeCount: 6,
        outputCount: 3,
        maximumDepth: 1,
        task: 'classification',
        transform: 'identity'
      },
      nodes: packed('nodes', 'uint32', nodes),
      trees: packed('trees', 'uint32', [0, 3, floatBits(0.5), 0, 3, 6, floatBits(0.5), 0]),
      leafValues: packed(
        'leaf-values',
        'float32',
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
      ),
      parameters: packed('parameters', 'float32', [1, 0, 0, 0]),
      rowCount: rows,
      ...(kind === 'forest-raster'
        ? {
            rasterStack: {
              bands: [0, 1].map(channel => ({
                id: `band-${channel}`,
                format: 'float32' as const,
                storage: {
                  kind: 'buffer' as const,
                  values: packed(
                    `band-${channel}`,
                    'float32',
                    points.map(point => point[channel])
                  )
                }
              })),
              grid: {
                width: rows,
                height: 1,
                affine: [1, 0, 0, 0, -1, 0],
                pixelInterpretation: 'area'
              }
            }
          }
        : {
            features: input('features', 'float32', 2, flat),
            featureLayout: kind === 'forest-feature-major' ? 'feature-major' : 'row-major'
          }),
      mask,
      output: {
        predictions: output('predictions', 'float32'),
        validity: output('validity', 'uint32'),
        scores: output('scores', 'float32', 3),
        leafIndices: output('leafIndices', 'uint32', 2)
      }
    });
    expected = encoding => {
      const predictions: number[] = [],
        validity: number[] = [],
        scores: number[] = [],
        leafIndices: number[] = [];
      points.forEach(([first, second], row) => {
        const kept = isKept(row, encoding);
        validity.push(Number(kept));
        if (!kept) {
          predictions.push(NaN);
          scores.push(NaN, NaN, NaN);
          leafIndices.push(NO_VALUE, NO_VALUE);
          return;
        }
        const firstLeft = Number.isNaN(first) || first <= 256,
          secondLeft = second <= 2;
        const rowScores = [
          Number(firstLeft) * 0.5 + Number(!secondLeft) * 0.5,
          Number(!firstLeft) * 0.5,
          Number(secondLeft) * 0.5
        ];
        predictions.push(rowScores.indexOf(Math.max(...rowScores)));
        scores.push(...rowScores);
        leafIndices.push(firstLeft ? 11 : 12, secondLeft ? 21 : 22);
      });
      return {predictions, validity, scores, leafIndices};
    };
  }
  return {
    graph,
    contributor,
    createContributor: (id: string): GPUCommandNodeProducer => {
      const Contributor = contributor.constructor as new (props: {
        id?: string;
      }) => GPUCommandNodeProducer;
      return new Contributor({...contributor.props, id});
    },
    outputs,
    updateInputs: (encoding: number) => updates.forEach(update => update(encoding)),
    expected,
    async check(encoding: number) {
      const expectedValues = expected(encoding);
      const actual: Record<string, number[]> = {};
      for (const [name, fields] of outputs) {
        const values: number[] = [];
        for (const field of fields) {
          const bytes = await field.buffer.readAsync();
          const data =
            field.format === 'float32'
              ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
              : field.format === 'sint32'
                ? new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
                : new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
          expect(Array.from(data.slice(0, field.prefix))).toEqual(
            Array.from(field.storage.slice(0, field.prefix))
          );
          expect(Array.from(data.slice(field.prefix + field.count))).toEqual(
            Array.from(field.storage.slice(field.prefix + field.count))
          );
          values.push(...data.slice(field.prefix, field.prefix + field.count));
        }
        actual[name] = values;
        expect(values).toHaveLength(expectedValues[name].length);
        values.forEach((value, index) => {
          const wanted = expectedValues[name][index];
          if (Number.isNaN(wanted))
            expect(Number.isNaN(value), `${kind} ${name} ${index}`).toBe(true);
          else if (kind === 'fourier' || kind === 'harmonics')
            expect(value, `${kind} ${name} ${index}`).toBeCloseTo(wanted, 5);
          else expect(value, `${kind} ${name} ${index}`).toBe(wanted);
        });
      }
      return actual;
    },
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}
