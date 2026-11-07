// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ResidentNumericHistogram} from '../../../examples/deck/kepler-compute/src/resident-numeric-histogram';
import {
  getResidentNumericPlotBin,
  makeResidentNumericPlotEdges
} from '../../../examples/deck/kepler-compute/src/resident-numeric-plot-edges';
import type {ResidentPlotMeasureOperation} from '../../../examples/deck/kepler-compute/src/resident-plot-measure';

type Stored = number | null;

/** d3 sum/mean/min/max/median/deviation over the non-null numbers of a bin. */
function reduce(values: number[], operation: ResidentPlotMeasureOperation): number | null {
  const numbers = values.filter(value => Number.isFinite(value));
  if (operation === 'sum') return numbers.reduce((total, value) => total + value, 0);
  if (!numbers.length) return null;
  const sorted = [...numbers].sort((left, right) => left - right);
  if (operation === 'mean')
    return numbers.reduce((total, value) => total + value, 0) / numbers.length;
  if (operation === 'min') return sorted[0];
  if (operation === 'max') return sorted[sorted.length - 1];
  if (operation === 'median') {
    const position = (sorted.length - 1) / 2;
    const low = Math.floor(position);
    return sorted[low] + (sorted[Math.ceil(position)] - sorted[low]) * (position - low);
  }
  if (numbers.length < 2) return null;
  const mean = numbers.reduce((total, value) => total + value, 0) / numbers.length;
  return Math.sqrt(
    numbers.reduce((total, value) => total + (value - mean) ** 2, 0) / (numbers.length - 1)
  );
}

it('native numeric histogram classifies native edges for every format, null as zero, chunks, selection and y measures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const xParts: Record<string, Stored[][]> = {
    float: [
      [-10, -9.99, 0, 2.5, null, 4.9999999, 5, 99.99999, 100, 100.5],
      [],
      [-10.5, 7, 7, 12.5, 55, null, 100, 3, 33.3, 21]
    ],
    uint: [[0, 1, 2, 5, 10, 99, 100, 101], [], [4294967295, 7, 255, 12, null, 50, 100, 25]],
    sint: [[-2147483648, -10, -9, 0, 1, 99, 100, 2147483647], [], [-5, 7, 50, null, 100, 12, 33, 5]]
  };
  const yParts: Stored[][] = [
    [1, 2, 3, 4, 5, null, 7, 8, 9, 10],
    [],
    [11, 12, 13, 14, 15, 16, 17, 18, null, 20]
  ];
  const types = {float: new arrow.Float32(), uint: new arrow.Uint32(), sint: new arrow.Int32()};
  const results: string[] = [];
  for (const [kind, parts] of Object.entries(xParts)) {
    const lengths = parts.map(part => part.length);
    // The y column follows the x chunk lengths; shorter sources reuse the first values.
    const y = lengths.map((length, index) =>
      Array.from({length}, (_, row) => (yParts[index]?.length ? yParts[index][row] : row + 100))
    );
    const table = new arrow.Table(
      lengths.flatMap(
        (_, index) =>
          new arrow.Table({
            x: arrow.vectorFromArray(parts[index], types[kind as keyof typeof types]),
            y: arrow.vectorFromArray(y[index], new arrow.Float32())
          }).batches
      )
    );
    const dataset = session.importArrowDataset({id: `numeric-${kind}`, table});
    const column = dataset.value.columns.x;
    expect(column.data.map(chunk => chunk.length)).toEqual(lengths);
    const selectedFlags = parts.map((part, index) =>
      part.map((_, row) => (index === 0 && row === 3 ? 0 : (row + index) % 5 === 4 ? 0 : 1))
    );
    const selected = new GPUVector({
      name: 'selected',
      type: 'data',
      format: 'uint32',
      ownsData: true,
      data: column.data.map(
        (chunk, index) =>
          new GPUData({
            format: 'uint32',
            length: chunk.length,
            ownsBuffer: true,
            buffer: device.createBuffer({
              data: Uint32Array.from(selectedFlags[index].length ? selectedFlags[index] : [0]),
              usage: Buffer.STORAGE | Buffer.COPY_DST
            })
          })
      )
    });
    const spies = [...column.data, ...selected.data].map(chunk =>
      vi.spyOn(chunk.buffer, 'readAsync')
    );
    for (const [domain, interval] of [
      [[-10, 100], undefined],
      [[0, 100], 25],
      [[-3, 130], undefined]
    ] as [[number, number], number | undefined][]) {
      const plan = makeResidentNumericPlotEdges(domain, interval);
      const binCount = plan.thresholds.length + 1;
      const flat = parts.flat();
      const flatY = y.flat();
      const flatSelected = selectedFlags.flat();
      const expectedCounts = new Array(binCount).fill(0);
      const rowsByBin: number[][] = Array.from({length: binCount}, () => []);
      flat.forEach((value, row) => {
        const stored = value === null ? null : kind === 'float' ? Math.fround(value) : value;
        const bin = getResidentNumericPlotBin(stored, plan);
        if (bin >= 0 && flatSelected[row]) {
          expectedCounts[bin]++;
          rowsByBin[bin].push(row);
        }
      });
      for (const operation of [
        undefined,
        'mean',
        'sum',
        'min',
        'max',
        'median',
        'standardDeviation'
      ] as const) {
        const histogram = new ResidentNumericHistogram({
          session,
          datasetId: dataset.id,
          field: 'x',
          plan,
          mask: selected,
          workingMemoryBudgetBytes: 8 * 1024 * 1024,
          measure: operation ? {column: 'y', operation} : undefined
        });
        try {
          const encoder = device.createCommandEncoder();
          histogram.encode(encoder);
          device.submit(encoder.finish());
          const summary = (await histogram.readSummary())!;
          const label = `${kind} ${domain} ${interval} ${operation}`;
          expect(Array.from(summary.histograms['native-plot']), label).toEqual(expectedCounts);
          if (operation) {
            const measure = summary.measure!;
            for (let bin = 0; bin < binCount; bin++) {
              // Null y values are skipped like d3 skips null.
              const reduced = reduce(
                rowsByBin[bin]
                  .map(row => flatY[row])
                  .filter((value): value is number => value !== null),
                operation
              );
              if (operation === 'sum') {
                // An empty reduction is invalid and the bridge plots it as zero, as d3.sum does.
                expect(
                  measure.valid[bin] ? measure.values[bin] : 0,
                  `${label} bin ${bin}`
                ).toBeCloseTo(reduced ?? 0, 3);
              } else if (reduced === null) {
                expect(measure.valid[bin], `${label} bin ${bin} empty`).toBe(0);
              } else {
                expect(measure.valid[bin], `${label} bin ${bin} valid`).toBe(1);
                expect(measure.values[bin], `${label} bin ${bin}`).toBeCloseTo(reduced, 3);
              }
            }
          }
        } finally {
          const fence = device.createFence();
          await histogram.close(fence.signaled);
          fence.destroy();
        }
        results.push(`${kind}:${operation ?? 'count'}`);
      }
    }
    expect(spies.every(spy => spy.mock.calls.length === 0)).toBe(true);
    spies.forEach(spy => spy.mockRestore());
    selected.destroy();
  }
  expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
  await session.destroy();
  expect(results.length).toBe(3 * 3 * 7);
});

it('native numeric histogram median and deviation stay exact and within the declared memory at a million rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 512 * 1024 * 1024});
  const rowCount = 1_000_000;
  const x = new Float32Array(rowCount);
  const y = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    x[row] = ((row * 2654435761) % 1000003) / 10000;
    y[row] = ((row * 40503) % 9973) / 7;
  }
  const dataset = session.importArrowDataset({
    id: 'numeric-large',
    table: new arrow.Table({x: arrow.makeVector(x), y: arrow.makeVector(y)})
  });
  const selected = new GPUVector({
    name: 'selected',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: dataset.value.columns.x.data.map(
      chunk =>
        new GPUData({
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: new Uint32Array(chunk.length).fill(1),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
  const plan = makeResidentNumericPlotEdges([0, 100]);
  const binCount = plan.thresholds.length + 1;
  const rowsByBin: number[][] = Array.from({length: binCount}, () => []);
  for (let row = 0; row < rowCount; row++) {
    const bin = getResidentNumericPlotBin(x[row], plan);
    if (bin >= 0) rowsByBin[bin].push(row);
  }
  try {
    for (const operation of ['median', 'standardDeviation'] as const) {
      const histogram = new ResidentNumericHistogram({
        session,
        datasetId: dataset.id,
        field: 'x',
        plan,
        mask: selected,
        workingMemoryBudgetBytes: 256 * 1024 * 1024,
        measure: {column: 'y', operation}
      });
      try {
        const encoder = device.createCommandEncoder();
        histogram.encode(encoder);
        device.submit(encoder.finish());
        const summary = (await histogram.readSummary())!;
        expect(Array.from(summary.histograms['native-plot'])).toEqual(
          rowsByBin.map(rows => rows.length)
        );
        for (const bin of [0, 1, binCount >> 1, binCount - 2]) {
          const expected = reduce(
            rowsByBin[bin].map(row => y[row]),
            operation
          );
          expect(summary.measure!.valid[bin]).toBe(1);
          expect(summary.measure!.values[bin], `${operation} bin ${bin}`).toBeCloseTo(expected!, 2);
        }
      } finally {
        const fence = device.createFence();
        await histogram.close(fence.signaled);
        fence.destroy();
      }
    }
  } finally {
    selected.destroy();
    await session.destroy();
  }
}, 120000);

it('native grouped line series rank labels by first (bin, row), merge Others, and reduce counts and y per series', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const labels = ['delta', 'alpha', '', null, 'gamma', 'beta', 'alpha', 'epsilon', 'delta', 'zeta'];
  const rowCount = 90;
  const x = Array.from({length: rowCount}, (_, row) => ((row * 37) % 101) - 0.5);
  const group = Array.from(
    {length: rowCount},
    (_, row) => labels[(row * 7 + (row % 5)) % labels.length]
  );
  const y = Array.from({length: rowCount}, (_, row) =>
    row % 11 === 3 ? null : ((row * 13) % 17) + row / 10
  );
  const table = new arrow.Table({
    x: arrow.vectorFromArray(x, new arrow.Float32()),
    group: arrow.vectorFromArray(group),
    y: arrow.vectorFromArray(y, new arrow.Float32())
  });
  const dataset = session.importArrowDataset({
    id: 'numeric-groups',
    table: table.slice(0, 20).concat(table.slice(20, 20), table.slice(20))
  });
  const selectedFlags = Array.from({length: rowCount}, (_, row) => (row % 9 === 4 ? 0 : 1));
  const selected = new GPUVector({
    name: 'selected',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: dataset.value.columns.x.data.map((chunk, index) => {
      const start = dataset.value.columns.x.data
        .slice(0, index)
        .reduce((total, other) => total + other.length, 0);
      return new GPUData({
        format: 'uint32',
        length: chunk.length,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: Uint32Array.from(
            selectedFlags.slice(start, start + chunk.length).concat(chunk.length ? [] : [0])
          ),
          usage: Buffer.STORAGE | Buffer.COPY_DST
        })
      });
    })
  });
  const plan = makeResidentNumericPlotEdges([-0.5, 100]);
  const binCount = plan.thresholds.length + 1;
  try {
    for (const [numGroups, groupOthers, operation] of [
      [3, true, 'mean'],
      [3, false, 'median'],
      [10, true, 'standardDeviation'],
      ['ALL', false, undefined],
      [2, true, undefined]
    ] as const) {
      const histogram = new ResidentNumericHistogram({
        session,
        datasetId: dataset.id,
        field: 'x',
        plan,
        mask: selected,
        workingMemoryBudgetBytes: 8 * 1024 * 1024,
        grouping: {column: 'group', numGroups, groupOthers},
        measure: operation ? {column: 'y', operation} : undefined
      });
      try {
        const encoder = device.createCommandEncoder();
        histogram.encode(encoder);
        device.submit(encoder.finish());
        const summary = (await histogram.readSummary())!;
        const groups = summary.groups!;
        // CPU twin of native: bins ascending, rows ascending, labels null or empty skipped.
        const rows = Array.from({length: rowCount}, (_, row) => row)
          .filter(row => selectedFlags[row])
          .map(row => ({row, bin: getResidentNumericPlotBin(Math.fround(x[row]), plan)}))
          .filter(entry => entry.bin >= 0)
          .sort((left, right) => left.bin - right.bin || left.row - right.row);
        const order: string[] = [];
        for (const {row} of rows) {
          const label = group[row];
          if (label !== null && label !== '' && !order.includes(label)) order.push(label);
        }
        const limit = numGroups === 'ALL' ? 20 : numGroups;
        const kept = order.slice(0, limit);
        const others = groupOthers && order.length > kept.length;
        const label = `${numGroups} ${groupOthers} ${operation}`;
        expect(groups.labels, label).toEqual(kept);
        expect(groups.others, label).toBe(others);
        const slotOf = (name: string | null) =>
          name === null || name === ''
            ? -1
            : kept.includes(name)
              ? kept.indexOf(name)
              : others
                ? kept.length
                : -1;
        const cells = new Map<number, number[]>();
        for (const {row, bin} of rows) {
          const slot = slotOf(group[row]);
          if (slot < 0) continue;
          const key = slot * binCount + bin;
          cells.set(key, [...(cells.get(key) ?? []), row]);
        }
        const slotCount = kept.length + (others ? 1 : 0);
        for (let key = 0; key < groups.counts.length; key++) {
          const cell = key < slotCount * binCount ? (cells.get(key) ?? []) : [];
          expect(groups.counts[key], `${label} count ${key}`).toBe(cell.length);
          if (operation && cell.length) {
            const expected = reduce(
              cell.map(row => y[row]).filter((value): value is number => value !== null),
              operation
            );
            if (expected === null) expect(groups.measure!.valid[key], `${label} ${key}`).toBe(0);
            else {
              expect(groups.measure!.valid[key], `${label} ${key} valid`).toBe(1);
              expect(groups.measure!.values[key], `${label} ${key}`).toBeCloseTo(expected, 3);
            }
          }
        }
      } finally {
        const fence = device.createFence();
        await histogram.close(fence.signaled);
        fence.destroy();
      }
    }
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
  } finally {
    selected.destroy();
    await session.destroy();
  }
});
