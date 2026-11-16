// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {cellToChildren, getPentagons, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUCellAggregation,
  GPUCellLevelSelection,
  GPUCellPyramid,
  type GPUCellAggregationProps,
  type GPUCellFamily,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {
  createVectorView,
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  aggregateCellsOnCPU,
  getCellResolution,
  getCellParent,
  getQuadbinPointKeys,
  isValidCellKey,
  h3ToBigInt,
  joinCellKey,
  quadbinPointToCell,
  quadbinTileToCell,
  splitCellKey,
  type OracleCell
} from './cell-aggregation-oracle';
import {createPointPositions, createRandom} from './cell-aggregation-points';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch
} from '../../../src/analysis-core/kernels/graph-chunk-dispatch';
import {createCellRowKernelNode} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-row-kernel';

import {withGroupStatisticsTestBindingLimit} from '../../gpu-dataframe/group-statistics/group-statistics-test-device';

const SUM_SCALE = 65536;

type TableBuffers = Record<
  | 'cells'
  | 'counts'
  | 'sums'
  | 'sumValues'
  | 'minimums'
  | 'maximums'
  | 'count'
  | 'overflow'
  | 'total',
  Buffer
> & {capacity: number};

type ActualTable = {
  count: number;
  overflow: number;
  total: number;
  rows: OracleCell[];
  sumValues: number[];
  /** True when every row past `count` holds the empty key and a zero count. */
  hasEmptyTail: boolean;
};

class Fixture {
  readonly buffers: Buffer[] = [];
  constructor(readonly device: Device) {}

  input(values: Float32Array | Uint32Array): Buffer {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return buffer;
  }

  output(length: number): Buffer {
    const buffer = createOutputBuffer(this.device, length);
    this.buffers.push(buffer);
    return buffer;
  }

  createTable(capacity: number): TableBuffers {
    return {
      capacity,
      cells: this.output(2 * capacity),
      counts: this.output(capacity),
      sums: this.output(2 * capacity),
      sumValues: this.output(capacity),
      minimums: this.output(capacity),
      maximums: this.output(capacity),
      count: this.output(1),
      overflow: this.output(1),
      total: this.output(1)
    };
  }

  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

function importTable(graph: GPUCommandGraph, prefix: string, table: TableBuffers): GPUCellTable {
  const {capacity} = table;
  return {
    cells: importGraphBuffer(graph, `${prefix}-cells`, table.cells, 'uint32x2', capacity),
    counts: importGraphBuffer(graph, `${prefix}-counts`, table.counts, 'uint32', capacity),
    sums: importGraphBuffer(graph, `${prefix}-sums`, table.sums, 'uint32x2', capacity),
    sumValues: importGraphBuffer(
      graph,
      `${prefix}-sum-values`,
      table.sumValues,
      'float32',
      capacity
    ),
    minimums: importGraphBuffer(graph, `${prefix}-minimums`, table.minimums, 'float32', capacity),
    maximums: importGraphBuffer(graph, `${prefix}-maximums`, table.maximums, 'float32', capacity),
    status: {
      count: importGraphBuffer(graph, `${prefix}-count`, table.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, `${prefix}-overflow`, table.overflow, 'uint32', 1),
      requiredCount: importGraphBuffer(graph, `${prefix}-total`, table.total, 'uint32', 1)
    }
  };
}

async function readTable(table: TableBuffers): Promise<ActualTable> {
  const {capacity} = table;
  const [count] = await readUint32(table.count, 1);
  const [overflow] = await readUint32(table.overflow, 1);
  const [total] = await readUint32(table.total, 1);
  const cells = await readUint32(table.cells, 2 * capacity);
  const counts = await readUint32(table.counts, capacity);
  const sums = await readUint32(table.sums, 2 * capacity);
  const sumValues = await readFloat32(table.sumValues, capacity);
  const minimums = await readFloat32(table.minimums, capacity);
  const maximums = await readFloat32(table.maximums, capacity);
  const rows: OracleCell[] = [];
  for (let row = 0; row < count; row++) {
    rows.push({
      key: joinCellKey(cells[2 * row], cells[2 * row + 1]),
      count: counts[row],
      sum: BigInt.asIntN(64, joinCellKey(sums[2 * row], sums[2 * row + 1])),
      minimum: minimums[row],
      maximum: maximums[row]
    });
  }
  let hasEmptyTail = true;
  for (let row = count; row < capacity; row++) {
    hasEmptyTail &&=
      cells[2 * row] === 0xffffffff && cells[2 * row + 1] === 0xffffffff && counts[row] === 0;
  }
  return {
    count,
    overflow,
    total,
    rows,
    sumValues: sumValues.slice(0, count),
    hasEmptyTail
  };
}

function expectTable(actual: ActualTable, expected: OracleCell[], capacity: number, label: string) {
  expect(actual.total, `${label} total`).toBe(expected.length);
  expect(actual.count, `${label} count`).toBe(Math.min(expected.length, capacity));
  expect(actual.overflow, `${label} overflow`).toBe(expected.length > capacity ? 1 : 0);
  expect(actual.hasEmptyTail, `${label} empty tail`).toBe(true);
  const bounded = expected.slice(0, capacity);
  const format = (cells: OracleCell[]) =>
    cells.map(cell => `${cell.key.toString(16)} ${cell.count} ${cell.sum}`);
  expect(format(actual.rows), `${label} keys, counts, sums`).toEqual(format(bounded));
  // Extremes are bit exact (Object.is distinguishes -0).
  for (const [row, cell] of bounded.entries()) {
    expect(
      Object.is(actual.rows[row].minimum, Math.fround(cell.minimum)),
      `${label} min ${row}`
    ).toBe(true);
    expect(
      Object.is(actual.rows[row].maximum, Math.fround(cell.maximum)),
      `${label} max ${row}`
    ).toBe(true);
    const sumValue = Number(cell.sum) / SUM_SCALE;
    expect(Math.abs(actual.sumValues[row] - sumValue)).toBeLessThanOrEqual(
      Math.abs(sumValue) * 1e-6 + 1e-6
    );
  }
}

function createValues(seed: number, rows: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: rows}, () => {
    const roll = random();
    return roll < 0.01 ? NaN : roll < 0.02 ? -0 : Math.fround((random() - 0.4) * 1000);
  });
}

type AggregationScene = {
  chunkLength?: number;
  emptyInputChunks?: boolean;
  scratchTileSize?: number;
  family: GPUCellFamily;
  resolution: number;
  positions?: Float32Array;
  cells?: Uint32Array;
  wordOrder?: GPUCellAggregationProps['wordOrder'];
  values: Float32Array;
  capacity: number;
};

/** One compiled aggregation graph with writable values and mask. */
function createAggregation(fixture: Fixture, scene: AggregationScene) {
  const {device} = fixture;
  const rows = scene.values.length;
  const graph = new GPUCommandGraph(device, {id: 'cell-aggregation-graph'});
  const chunkRows = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
    view: GraphDataView<Format>
  ) => {
    if (!scene.chunkLength) return view;
    const chunks: GraphDataView<Format>[] = [];
    const addEmptyChunk = (row: number) =>
      chunks.push(
        graph.createDataView(view.buffer, {
          format: view.format,
          length: 0,
          byteOffset: view.byteOffset + row * view.rowByteLength
        })
      );
    if (scene.emptyInputChunks) addEmptyChunk(0);
    for (let row = 0; row < view.length; row += scene.chunkLength)
      chunks.push(
        graph.createDataView(view.buffer, {
          format: view.format,
          length: Math.min(scene.chunkLength, view.length - row),
          byteOffset: view.byteOffset + row * view.rowByteLength
        })
      );
    if (scene.emptyInputChunks) addEmptyChunk(view.length);
    return createVectorView(`${view.buffer.id}-chunks`, view.format, chunks);
  };
  const valuesBuffer = fixture.input(scene.values);
  const maskBuffer = fixture.input(new Uint32Array(rows).fill(1));
  const table = fixture.createTable(scene.capacity);
  graph.add(
    new GPUCellAggregation({
      scratchTileSize: scene.scratchTileSize,
      family: scene.family,
      resolution: scene.resolution,
      positions: scene.positions
        ? chunkRows(
            importGraphBuffer(graph, 'positions', fixture.input(scene.positions), 'float32x2', rows)
          )
        : undefined,
      cells: scene.cells
        ? chunkRows(importGraphBuffer(graph, 'cells', fixture.input(scene.cells), 'uint32x2', rows))
        : undefined,
      wordOrder: scene.wordOrder,
      values: chunkRows(importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows)),
      mask: chunkRows(importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows)),
      sumScale: SUM_SCALE,
      output: importTable(graph, 'table', table)
    })
  );
  const compiled = graph.compile();
  return {
    compiled,
    table,
    async run(values: Float32Array, mask: Uint32Array): Promise<ActualTable> {
      valuesBuffer.write(values);
      maskBuffer.write(mask);
      submitGraph(device, compiled, undefined);
      return readTable(table);
    }
  };
}

it('GPUCellAggregation quadbin point keys and tables match the CPU reference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const resolution of [0, 1, 7, 15, 16, 21, 26]) {
    const fixture = new Fixture(device);
    const positions = createPointPositions(100 + resolution, 6000, resolution);
    // A few NaN coordinates are skipped.
    positions[10] = NaN;
    positions[33] = NaN;
    const rows = positions.length / 2;
    const values = createValues(resolution, rows);
    const keys = getQuadbinPointKeys(positions, resolution);
    const expected = aggregateCellsOnCPU({
      family: 'quadbin',
      resolution,
      keys,
      values,
      sumScale: SUM_SCALE
    });
    const aggregation = createAggregation(fixture, {
      family: 'quadbin',
      resolution,
      positions,
      values,
      capacity: rows
    });
    const actual = await aggregation.run(values, new Uint32Array(rows).fill(1));
    expectTable(actual, expected, rows, `resolution ${resolution}`);
    if (resolution >= 7) {
      expect(expected.length, 'fine levels hold many cells').toBeGreaterThan(rows / 4);
    }
    aggregation.compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellAggregation folds hot cells that span many accumulation tiles exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const random = createRandom(77);
  // 40000 rows: most share one coarse cell (runs of many 256-row tiles), the rest are spread so
  // runs also end mid-tile, and a handful of rows are NaN.
  const points: number[] = [];
  for (let index = 0; index < 40000; index++) {
    if (index % 10 < 7) {
      points.push(10 + random() * 0.001, 20 + random() * 0.001);
    } else if (index % 10 < 9) {
      points.push(10.5 + (index % 7) * 0.2, 20.2 + (index % 5) * 0.2);
    } else {
      points.push(random() * 360 - 180, random() * 170 - 85);
    }
  }
  const positions = Float32Array.from(points);
  positions[20] = NaN;
  positions[99] = NaN;
  const rows = positions.length / 2;
  const values = createValues(5, rows);
  const resolution = 8;
  const expected = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution,
    keys: getQuadbinPointKeys(positions, resolution),
    values,
    sumScale: SUM_SCALE
  });
  const aggregation = createAggregation(fixture, {
    family: 'quadbin',
    resolution,
    positions,
    values,
    capacity: rows
  });
  const actual = await aggregation.run(values, new Uint32Array(rows).fill(1));
  expectTable(actual, expected, rows, 'hot cells');
  expect(Math.max(...expected.map(cell => cell.count))).toBeGreaterThan(20000);
  aggregation.compiled.destroy();
  fixture.destroy();
});

it('GPUCellAggregation follows per-frame masks and values without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const resolution = 9;
  const positions = createPointPositions(7, 4000, resolution);
  const rows = positions.length / 2;
  const keys = getQuadbinPointKeys(positions, resolution);
  const aggregation = createAggregation(fixture, {
    family: 'quadbin',
    resolution,
    positions,
    values: createValues(1, rows),
    capacity: rows
  });
  const compiled = aggregation.compiled;
  const totals: number[] = [];
  for (const seed of [1, 2, 3, 1]) {
    const random = createRandom(seed * 17);
    const mask = Uint32Array.from({length: rows}, () => (random() < 0.2 * seed ? 0 : 1));
    const values = createValues(seed, rows);
    const actual = await aggregation.run(values, mask);
    const expected = aggregateCellsOnCPU({
      family: 'quadbin',
      resolution,
      keys,
      values,
      mask,
      sumScale: SUM_SCALE
    });
    expectTable(actual, expected, rows, `seed ${seed}`);
    totals.push(actual.total);
    expect(aggregation.compiled).toBe(compiled);
  }
  expect(new Set(totals).size).toBeGreaterThan(2);
  expect(totals[0]).toBe(totals[3]);
  // An all-zero mask leaves an empty table.
  const empty = await aggregation.run(createValues(1, rows), new Uint32Array(rows));
  expect([empty.count, empty.total, empty.overflow, empty.hasEmptyTail]).toEqual([0, 0, 0, true]);
  compiled.destroy();
  fixture.destroy();
});

/** Pre-keyed H3 rows: random cells, pentagons, poles and antimeridian, finer, coarser and invalid. */
function createH3Keys(seed: number, rows: number, resolution: number): (bigint | null)[] {
  const random = createRandom(seed);
  const keys: bigint[] = [];
  const special = [
    ...getPentagons(resolution + 1),
    ...cellToChildren(getPentagons(resolution)[3], resolution + 2),
    latLngToCell(89.9, 10, resolution + 1),
    latLngToCell(-89.9, -170, resolution + 3),
    latLngToCell(0, 180, resolution + 1),
    latLngToCell(0, -179.9999, resolution + 1),
    latLngToCell(65.5, 179.99, resolution + 2),
    latLngToCell(-16.2, -179.999, resolution + 2)
  ];
  keys.push(...special.map(h3ToBigInt));
  while (keys.length < rows) {
    const roll = random();
    const latitude = random() * 180 - 90;
    const longitude = random() * 360 - 180;
    if (roll < 0.05) {
      // Coarser than the table: skipped.
      keys.push(h3ToBigInt(latLngToCell(latitude, longitude, Math.max(resolution - 2, 0))));
    } else if (roll < 0.08) {
      // Invalid: garbage words, a used digit set to 7, and the wrong mode.
      const cell = h3ToBigInt(latLngToCell(latitude, longitude, resolution + 1));
      keys.push(
        [0x123456789abcdefn, cell | (7n << BigInt(3 * (15 - 1))), cell ^ (3n << 59n)][
          Math.floor(random() * 3)
        ]
      );
    } else if (roll < 0.4) {
      // Clustered fine cells.
      keys.push(
        h3ToBigInt(latLngToCell(40 + random() * 0.3, -74 + random() * 0.3, resolution + 3))
      );
    } else {
      keys.push(
        h3ToBigInt(
          latLngToCell(latitude, longitude, Math.min(resolution + Math.floor(random() * 4), 15))
        )
      );
    }
  }
  return keys.slice(0, rows);
}

function packKeys(keys: readonly (bigint | null)[], highLow = false): Uint32Array {
  const words = new Uint32Array(keys.length * 2);
  for (const [row, key] of keys.entries()) {
    const [low, high] = splitCellKey(key ?? 0n);
    words[2 * row] = highLow ? high : low;
    words[2 * row + 1] = highLow ? low : high;
  }
  return words;
}

it('GPUCellAggregation aggregates pre-keyed H3 and Quadbin rows (CARTO columns)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(5);
  const scenes: {
    family: GPUCellFamily;
    resolution: number;
    keys: (bigint | null)[];
    highLow?: boolean;
  }[] = [
    {family: 'h3', resolution: 5, keys: createH3Keys(1, 3000, 5)},
    // Width 7 + 30 bits: the two-word sort path.
    {family: 'h3', resolution: 10, keys: createH3Keys(2, 3000, 10)},
    {
      family: 'h3',
      resolution: 0,
      keys: createH3Keys(3, 1000, 0),
      highLow: true
    },
    {
      family: 'quadbin',
      resolution: 12,
      keys: Array.from({length: 3000}, (_, row) => {
        const longitude = random() * 360 - 180;
        const latitude = random() * 170 - 85;
        const level = row % 10 === 0 ? 8 : 12 + Math.floor(random() * 15);
        const cell = quadbinPointToCell(longitude, latitude, Math.min(level, 26));
        return row % 37 === 0 ? cell - 1n : cell;
      })
    }
  ];
  for (const scene of scenes) {
    const fixture = new Fixture(device);
    const rows = scene.keys.length;
    const values = createValues(scene.resolution + 40, rows);
    const expected = aggregateCellsOnCPU({
      family: scene.family,
      resolution: scene.resolution,
      keys: scene.keys,
      values,
      sumScale: SUM_SCALE
    });
    const aggregation = createAggregation(fixture, {
      family: scene.family,
      resolution: scene.resolution,
      cells: packKeys(scene.keys, scene.highLow),
      wordOrder: scene.highLow ? 'high-low' : 'little-endian',
      values,
      capacity: rows
    });
    const actual = await aggregation.run(values, new Uint32Array(rows).fill(1));
    expectTable(actual, expected, rows, `${scene.family} ${scene.resolution}`);
    expect(expected.length).toBeGreaterThan(scene.resolution === 0 ? 50 : 300);
    aggregation.compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellPyramid roll-ups equal direct aggregation and switch levels per frame', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {family, levels} of [
    {family: 'quadbin' as const, levels: [18, 14, 9, 3, 0]},
    {family: 'h3' as const, levels: [9, 6, 2]}
  ]) {
    const fixture = new Fixture(device);
    const finest = levels[0];
    let keys: (bigint | null)[];
    let positions: Float32Array | undefined;
    if (family === 'quadbin') {
      positions = createPointPositions(77, 8000, finest);
      keys = getQuadbinPointKeys(positions, finest);
    } else {
      keys = createH3Keys(9, 8000, finest);
    }
    const rows = keys.length;
    const values = createValues(8, rows);
    // Capacities hold every cell here; the overflow test covers truncated levels.
    const capacities = levels.map((_, levelIndex) => rows + 2 + levelIndex);
    const tables = capacities.map(capacity => fixture.createTable(capacity));
    const levelCountsBuffer = fixture.output(levels.length);
    const graph = new GPUCommandGraph(device, {id: 'cell-pyramid-graph'});
    const levelCounts = importGraphBuffer(
      graph,
      'level-counts',
      levelCountsBuffer,
      'uint32',
      levels.length
    );
    const pyramid = new GPUCellPyramid({
      family,
      positions: positions
        ? importGraphBuffer(graph, 'positions', fixture.input(positions), 'float32x2', rows)
        : undefined,
      cells: positions
        ? undefined
        : importGraphBuffer(graph, 'cells', fixture.input(packKeys(keys)), 'uint32x2', rows),
      values: importGraphBuffer(graph, 'values', fixture.input(values), 'float32', rows),
      sumScale: SUM_SCALE,
      resolutions: levels,
      output: {
        levels: levels.map((_, levelIndex) =>
          importTable(graph, `level-${levelIndex}`, tables[levelIndex])
        ),
        levelCounts
      }
    });
    graph.add(pyramid);
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const counts: number[] = [];
    // The pyramid only sees rows that reach the finest level; coarser pre-keyed rows are skipped.
    const finestKeys = keys.map(key =>
      key !== null && isValidCellKey(family, key) && getCellResolution(family, key) >= finest
        ? key
        : null
    );
    for (const [levelIndex, resolution] of levels.entries()) {
      const expected = aggregateCellsOnCPU({
        family,
        resolution,
        keys: finestKeys,
        values,
        sumScale: SUM_SCALE
      });
      const actual = await readTable(tables[levelIndex]);
      expectTable(actual, expected, capacities[levelIndex], `${family} level ${resolution}`);
      counts.push(actual.count);
    }
    expect(await readUint32(levelCountsBuffer, levels.length)).toEqual(counts);

    // Level selection runs in its own per-frame graph: rewrite one uint32, no rebuild.
    const selectionGraph = new GPUCommandGraph(device, {
      id: 'cell-level-selection-graph'
    });
    const activeLevel = new GPUParameterBuffer(device, {
      id: 'active-level',
      format: 'uint32',
      length: 1
    });
    const countBuffer = fixture.output(1);
    const drawBuffer = fixture.output(4);
    drawBuffer.write(Uint32Array.from([6, 0, 0, 0]));
    const levelFirstRows = levels.map((_, levelIndex) => levelIndex * 100);
    selectionGraph.add(
      new GPUCellLevelSelection({
        levelCounts: importGraphBuffer(
          selectionGraph,
          'level-counts',
          levelCountsBuffer,
          'uint32',
          levels.length
        ),
        activeLevel: activeLevel.importToGraph(selectionGraph),
        levelFirstRows,
        output: {
          status: {
            count: importGraphBuffer(selectionGraph, 'count', countBuffer, 'uint32', 1)
          },
          drawArguments: importGraphBuffer(selectionGraph, 'draw', drawBuffer, 'uint32', 4)
        }
      })
    );
    const selection = selectionGraph.compile();
    for (const level of [2, 0, levels.length - 1, 99]) {
      activeLevel.write(Uint32Array.from([level]));
      submitGraph(device, selection, undefined);
      const selected = Math.min(level, levels.length - 1);
      expect(await readUint32(countBuffer, 1)).toEqual([counts[selected]]);
      expect(await readUint32(drawBuffer, 4)).toEqual([
        6,
        counts[selected],
        0,
        levelFirstRows[selected]
      ]);
    }
    selection.destroy();
    activeLevel.destroy();
    compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellAggregation and roll-ups clamp to capacity and propagate overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const levels = [10, 6, 1];
  const positions = createPointPositions(31, 3000, levels[0]);
  const rows = positions.length / 2;
  const keys = getQuadbinPointKeys(positions, levels[0]);
  const values = createValues(31, rows);
  const capacities = [25, 1000, 8];
  const tables = capacities.map(capacity => fixture.createTable(capacity));
  const graph = new GPUCommandGraph(device, {id: 'cell-overflow-graph'});
  graph.add(
    new GPUCellPyramid({
      family: 'quadbin',
      positions: importGraphBuffer(graph, 'positions', fixture.input(positions), 'float32x2', rows),
      values: importGraphBuffer(graph, 'values', fixture.input(values), 'float32', rows),
      sumScale: SUM_SCALE,
      resolutions: levels,
      output: {
        levels: levels.map((_, levelIndex) =>
          importTable(graph, `level-${levelIndex}`, tables[levelIndex])
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const finest = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: levels[0],
    keys,
    values,
    sumScale: SUM_SCALE
  });
  expect(finest.length).toBeGreaterThan(capacities[0]);
  const level0 = await readTable(tables[0]);
  // The 25 smallest keys survive, exactly.
  expectTable(level0, finest, capacities[0], 'finest');
  // Level 1 rolls up only the surviving cells and inherits the overflow flag.
  const surviving = new Set(finest.slice(0, capacities[0]).map(cell => cell.key));
  const survivingKeys = keys.map(key => (key !== null && surviving.has(key) ? key : null));
  const level1Expected = aggregateCellsOnCPU({
    family: 'quadbin',
    resolution: levels[1],
    keys: survivingKeys,
    values,
    sumScale: SUM_SCALE
  });
  const level1 = await readTable(tables[1]);
  expect(level1.overflow).toBe(1);
  expect(level1.rows.map(cell => [cell.key, cell.count, cell.sum])).toEqual(
    level1Expected.map(cell => [cell.key, cell.count, cell.sum])
  );
  const level2 = await readTable(tables[2]);
  expect(level2.overflow).toBe(1);
  compiled.destroy();
  fixture.destroy();
});

it.each([
  ['quadbin', 8],
  ['quadbin', 26],
  ['h3', 8],
  ['h3', 12]
] as const)('GPUCellAggregation %s resolution %i is identical for packed, uneven and boundary batches', async (family, resolution) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 769;
  const positions = createPointPositions(317, rows, resolution).slice(0, rows * 2);
  positions[8] = NaN;
  const values = createValues(83, rows);
  values[2] = -0;
  values[258] = Infinity;
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 === 0 ? 0 : 1));
  const keys =
    family === 'quadbin'
      ? getQuadbinPointKeys(positions, resolution)
      : Array.from({length: rows}, (_, row) =>
          h3ToBigInt(latLngToCell((row % 150) - 75, (row % 350) - 175, 12))
        );
  keys[5] = 0n;
  for (const lengths of [
    [3, 257, rows - 260],
    [0, 1, 255, 256, 257, 0]
  ]) {
    for (const inputMode of family === 'quadbin'
      ? (['points', 'little-endian', 'high-low'] as const)
      : (['little-endian', 'high-low'] as const)) {
      for (const capacity of [16, rows]) {
        const fixture = new Fixture(device);
        const words = Uint32Array.from(
          keys.flatMap(key => {
            const [low, high] = splitCellKey(key ?? 0n);
            return inputMode === 'high-low' ? [high, low] : [low, high];
          })
        );
        const packed = createAggregation(fixture, {
          family,
          resolution,
          values,
          capacity,
          ...(inputMode === 'points' ? {positions} : {cells: words, wordOrder: inputMode})
        });
        const graph = new GPUCommandGraph(device);
        const input = <Format extends 'float32x2' | 'float32' | 'uint32' | 'uint32x2'>(
          name: string,
          format: Format,
          data: Float32Array | Uint32Array,
          components = 1
        ) => {
          let firstRow = 0;
          const buffers: Buffer[] = [];
          const chunks: GraphDataView<Format>[] = lengths.map((length, chunkIndex) => {
            // Three unused prefix rows exercise unaligned binding prefixes in every input column.
            const padded =
              data instanceof Float32Array
                ? new Float32Array((Math.max(length, 1) + 3) * components)
                : new Uint32Array((Math.max(length, 1) + 3) * components);
            padded.set(
              data.subarray(firstRow * components, (firstRow + length) * components),
              3 * components
            );
            const buffer = fixture.input(padded);
            buffers.push(buffer);
            firstRow += length;
            const backing = importGraphBuffer(graph, `${name}-${chunkIndex}`, buffer, format);
            return graph.createDataView(backing.buffer, {
              format,
              length,
              byteOffset: 3 * components * 4
            });
          });
          return {view: createVectorView(name, format, chunks), buffers};
        };
        const valueInput = input('values', 'float32', values);
        const maskInput = input('mask', 'uint32', mask);
        const table = fixture.createTable(capacity);
        graph.add(
          new GPUCellAggregation({
            family,
            resolution,
            sumScale: SUM_SCALE,
            ...(inputMode === 'points'
              ? {positions: input('positions', 'float32x2', positions, 2).view}
              : {cells: input('cells', 'uint32x2', words, 2).view, wordOrder: inputMode}),
            values: valueInput.view,
            mask: maskInput.view,
            output: importTable(graph, 'output', table)
          })
        );
        const compiled = graph.compile();
        try {
          for (const frame of [0, 1, 2]) {
            const currentMask = frame === 1 ? new Uint32Array(rows) : mask;
            const currentValues =
              frame === 2 ? Float32Array.from(values, value => -value * 0.5) : values;
            let firstRow = 0;
            for (const [chunkIndex, length] of lengths.entries()) {
              if (length > 0) {
                maskInput.buffers[chunkIndex].write(
                  currentMask.subarray(firstRow, firstRow + length),
                  12
                );
                valueInput.buffers[chunkIndex].write(
                  currentValues.subarray(firstRow, firstRow + length),
                  12
                );
              }
              firstRow += length;
            }
            submitGraph(device, compiled, undefined);
            const actual = await readTable(table);
            expect(actual).toEqual(await packed.run(currentValues, currentMask));
            // Compare all words, including unused rows, NaN payloads and float32 styling sums.
            for (const column of [
              'cells',
              'counts',
              'sums',
              'sumValues',
              'minimums',
              'maximums',
              'count',
              'overflow',
              'total'
            ] as const) {
              const length =
                column === 'count' || column === 'overflow' || column === 'total'
                  ? 1
                  : capacity * (column === 'cells' || column === 'sums' ? 2 : 1);
              expect(await readUint32(table[column], length)).toEqual(
                await readUint32(packed.table[column], length)
              );
            }
            expectTable(
              actual,
              aggregateCellsOnCPU({
                family,
                resolution,
                keys: inputMode === 'points' ? getQuadbinPointKeys(positions, resolution) : keys,
                values: currentValues,
                mask: currentMask,
                sumScale: SUM_SCALE
              }),
              capacity,
              `${family} resolution ${resolution} ${inputMode} batches`
            );
          }
        } finally {
          compiled.destroy();
          packed.compiled.destroy();
          fixture.destroy();
        }
      }
    }
  }
});

it('GPUCellAggregation counts more than 2^24 / 16 rows across three record batches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const lengths = [17, 262147, 786429];
  const graph = new GPUCommandGraph(device);
  const chunks = lengths.map((length, chunkIndex) =>
    importGraphBuffer(
      graph,
      `positions-${chunkIndex}`,
      fixture.input(new Float32Array(length * 2)),
      'float32x2',
      length
    )
  );
  const table = fixture.createTable(1);
  const imported = importTable(graph, 'output', table);
  graph.add(
    new GPUCellAggregation({
      family: 'quadbin',
      resolution: 0,
      positions: createVectorView('positions', 'float32x2', chunks),
      output: {cells: imported.cells, counts: imported.counts, status: imported.status}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const totalRows = lengths.reduce((total, length) => total + length, 0);
    expect(totalRows).toBeGreaterThan(2 ** 24 / 16);
    expect(await readUint32(table.count, 1)).toEqual([1]);
    expect(await readUint32(table.overflow, 1)).toEqual([0]);
    expect(await readUint32(table.counts, 1)).toEqual([totalRows]);
    expect(await readUint32(table.cells, 2)).toEqual(splitCellKey(quadbinPointToCell(0, 0, 0)));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('GPUCellAggregation has closed-form counts, sums and extremes over 30 million chunked rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const lengths = [7_000_001, 9_000_003, 13_999_996];
  const graph = new GPUCommandGraph(device);
  const keys = Array.from({length: 4}, (_, cell) =>
    quadbinTileToCell(cell % 2, Math.floor(cell / 2), 1)
  );
  const cellChunks: GraphDataView<'uint32x2'>[] = [];
  const valueChunks: GraphDataView<'float32'>[] = [];
  for (const [chunkIndex, length] of lengths.entries()) {
    cellChunks.push(createTransientView(graph, `input-cells-${chunkIndex}`, 'uint32x2', length));
    valueChunks.push(createTransientView(graph, `input-values-${chunkIndex}`, 'float32', length));
  }
  const cellRows = createVectorView('input-cells', 'uint32x2', cellChunks);
  const valueRows = createVectorView('input-values', 'float32', valueChunks);
  const parameters = createGraphChunkDispatchParameters(graph, {
    id: 'generated-chunk-parameters',
    plan: planGraphChunkDispatch('generated-input', {cells: cellRows, values: valueRows})
  });
  for (const upload of parameters.nodes) {
    if (upload.type === 'copy') graph.addCopyPass(upload);
  }
  for (const [chunkIndex, length] of lengths.entries()) {
    // Generate inputs on the GPU with runtime row bases. Uneven batches begin at different
    // phases of the pattern, so a missing global row offset changes the closed-form result.
    const generator = createCellRowKernelNode(graph, {
      id: `generate-batch-${chunkIndex}`,
      operation: 'CellChunkTestData',
      variant: 'generate',
      length,
      parameters: parameters.slots[chunkIndex],
      bindings: [
        {
          name: 'generatedCells',
          view: cellChunks[chunkIndex],
          components: 2,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'generatedValues',
          view: valueChunks[chunkIndex],
          components: 1,
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const TEST_KEYS = array<vec2u, 4>(${keys
        .map(
          key =>
            `vec2u(${splitCellKey(key)
              .map(word => `${word}u`)
              .join(', ')})`
        )
        .join(', ')});`,
      body: `let sourceRow = chunkParameters[0] + index;
  let key = TEST_KEYS[sourceRow % 4u];
  generatedCells[generatedCellsOffset + 2u * index] = key.x;
  generatedCells[generatedCellsOffset + 2u * index + 1u] = key.y;
  generatedValues[generatedValuesOffset + index] = f32(i32(sourceRow % 8u) - 3) * 0.5;`
    });
    if (generator.type === 'compute') graph.addComputePass(generator);
  }
  const table = fixture.createTable(4);
  graph.add(
    new GPUCellAggregation({
      family: 'quadbin',
      resolution: 1,
      sumScale: SUM_SCALE,
      cells: createVectorView('input-cells', 'uint32x2', cellChunks),
      values: createVectorView('input-values', 'float32', valueChunks),
      output: importTable(graph, 'output', table)
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const count = 7_500_000;
    const expected: OracleCell[] = keys
      .map((key, cell) => ({
        key,
        count,
        sum: BigInt(count) * BigInt(cell - 1) * BigInt(SUM_SCALE / 2),
        minimum: (cell - 3) / 2,
        maximum: (cell + 1) / 2
      }))
      .sort((first, second) => (first.key < second.key ? -1 : 1));
    expectTable(await readTable(table), expected, 4, '30 million rows');
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}, 120000);

it.each([
  'quadbin',
  'h3'
] as const)('GPUCellPyramid %s chunked levels match direct aggregation and propagate overflow', async family => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 769;
  const levels = family === 'quadbin' ? [26, 8, 2] : [12, 8, 2];
  const positions = createPointPositions(521, rows, levels[0]).slice(0, rows * 2);
  const keys =
    family === 'quadbin'
      ? getQuadbinPointKeys(positions, levels[0])
      : Array.from({length: rows}, (_, row) =>
          h3ToBigInt(latLngToCell((row % 150) - 75, (row % 350) - 175, levels[0]))
        );
  const words = Uint32Array.from(keys.flatMap(key => splitCellKey(key ?? 0n)));
  const values = createValues(933, rows);
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 9 ? 1 : 0));
  const lengths = [0, 1, 255, 256, 257, 0];
  for (const overflow of [false, true]) {
    const fixture = new Fixture(device);
    const capacities = overflow ? [16, 32, 8] : [rows, rows, rows];
    const tables = capacities.map(capacity => fixture.createTable(capacity));
    const graph = new GPUCommandGraph(device);
    const chunks = <Format extends 'uint32x2' | 'uint32' | 'float32'>(
      name: string,
      format: Format,
      data: Uint32Array | Float32Array,
      components = 1
    ) => {
      let firstRow = 0;
      const buffers: Buffer[] = [];
      const views = lengths.map((length, chunkIndex) => {
        const content = data.slice(firstRow * components, (firstRow + length) * components);
        const buffer = fixture.input(content.length ? content : new Uint32Array(components));
        buffers.push(buffer);
        firstRow += length;
        return importGraphBuffer(graph, `${name}-${chunkIndex}`, buffer, format, length);
      });
      return {view: createVectorView(name, format, views), buffers};
    };
    const cellInput = chunks('cells', 'uint32x2', words, 2);
    const valueInput = chunks('values', 'float32', values);
    const maskInput = chunks('mask', 'uint32', mask);
    const levelCounts = fixture.output(levels.length);
    graph.add(
      new GPUCellPyramid({
        family,
        resolutions: levels,
        sumScale: SUM_SCALE,
        cells: cellInput.view,
        values: valueInput.view,
        mask: maskInput.view,
        output: {
          levels: tables.map((table, level) => importTable(graph, `level-${level}`, table)),
          levelCounts: importGraphBuffer(
            graph,
            'level-counts',
            levelCounts,
            'uint32',
            levels.length
          )
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const frame of [0, 1]) {
        let currentMask = frame === 0 ? mask : new Uint32Array(rows);
        let firstRow = 0;
        for (const [chunkIndex, length] of lengths.entries()) {
          if (length)
            maskInput.buffers[chunkIndex].write(currentMask.subarray(firstRow, firstRow + length));
          firstRow += length;
        }
        submitGraph(device, compiled, undefined);
        let inheritedOverflow = 0;
        const counts: number[] = [];
        for (const [level, resolution] of levels.entries()) {
          const direct = createAggregation(fixture, {
            family,
            resolution,
            cells: words,
            values,
            capacity: capacities[level]
          });
          try {
            const expected = await direct.run(values, currentMask);
            const actual = await readTable(tables[level]);
            inheritedOverflow |= expected.overflow;
            expect(actual).toEqual({...expected, overflow: inheritedOverflow});
            expectTable(
              {...actual, overflow: expected.overflow},
              aggregateCellsOnCPU({
                family,
                resolution,
                keys,
                values,
                mask: currentMask,
                sumScale: SUM_SCALE
              }),
              capacities[level],
              `direct ${family} level ${resolution}`
            );
            counts.push(actual.count);
            // Only rows belonging to retained finer cells reach the next level. This is the
            // documented overflow contract; direct aggregation must use the same surviving rows.
            const retainedKeys = new Set(actual.rows.map(cell => cell.key));
            currentMask = Uint32Array.from(currentMask, (keep, row) => {
              const key = keys[row];
              return keep &&
                key !== null &&
                retainedKeys.has(getCellParent(family, key, resolution))
                ? 1
                : 0;
            });
          } finally {
            direct.compiled.destroy();
          }
        }
        expect(await readUint32(levelCounts, levels.length)).toEqual(counts);
        expect(inheritedOverflow).toBe(overflow && frame === 0 ? 1 : 0);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('GPUCellLevelSelection publishes independently specified counts and first rows per encoding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const graph = new GPUCommandGraph(device);
  const activeLevelBuffer = fixture.input(new Uint32Array(1));
  const count = fixture.output(1);
  const firstRow = fixture.output(1);
  const draw = fixture.output(4);
  draw.write(Uint32Array.from([6, 0, 0, 0]));
  graph.add(
    new GPUCellLevelSelection({
      levelCounts: importGraphBuffer(
        graph,
        'level-counts',
        fixture.input(Uint32Array.from([5, 17, 0])),
        'uint32'
      ),
      activeLevel: importGraphBuffer(graph, 'active-level', activeLevelBuffer, 'uint32'),
      levelFirstRows: [8, 48, 128],
      output: {
        status: {count: importGraphBuffer(graph, 'count', count, 'uint32')},
        firstRow: importGraphBuffer(graph, 'first-row', firstRow, 'uint32'),
        drawArguments: importGraphBuffer(graph, 'draw', draw, 'uint32')
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const [activeLevel, expectedCount, expectedFirstRow] of [
      [0, 5, 8],
      [1, 17, 48],
      [99, 0, 128],
      [0, 5, 8]
    ]) {
      activeLevelBuffer.write(Uint32Array.of(activeLevel));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count, 1)).toEqual([expectedCount]);
      expect(await readUint32(firstRow, 1)).toEqual([expectedFirstRow]);
      expect(await readUint32(draw, 4)).toEqual([6, expectedCount, 0, expectedFirstRow]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('fixed cell tiles preserve global counts, sums, extremes and exact overflow across repeated runs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const rows = 515;
  const positions = new Float32Array(rows * 2);
  const values = Float32Array.from({length: rows}, (_, row) => (row % 17) - 8);
  for (let row = 0; row < rows; row++) {
    positions[2 * row] = [-150, -90, -30, 30, 90, 150][row % 6];
    positions[2 * row + 1] = row % 2 ? 30 : -30;
  }
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 5 === 0 ? 0 : 1));
  values[31] = Number.NaN;
  const scene: AggregationScene = {
    family: 'quadbin',
    resolution: 8,
    positions,
    values,
    capacity: 3,
    scratchTileSize: 64
  };
  const run = createAggregation(fixture, scene);
  const keys = getQuadbinPointKeys(positions, scene.resolution);
  try {
    for (const currentMask of [mask, new Uint32Array(rows), new Uint32Array(rows).fill(1)]) {
      const expected = aggregateCellsOnCPU({
        keys,
        family: 'quadbin',
        resolution: scene.resolution,
        values,
        mask: currentMask,
        sumScale: SUM_SCALE
      });
      expectTable(await run.run(values, currentMask), expected, scene.capacity, 'fixed tiles');
    }
  } finally {
    run.compiled.destroy();
    fixture.destroy();
  }
});

it('tiled extrema compile with runtime NaNs and refresh empty rows across encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const positions = new Float32Array(10);
  const values = Float32Array.of(-4, -1, -7, 9, 3);
  const run = createAggregation(fixture, {
    family: 'quadbin',
    resolution: 8,
    positions,
    values,
    capacity: 3,
    scratchTileSize: 2,
    chunkLength: 3
  });
  try {
    for (const frame of [
      {values, mask: new Uint32Array(5).fill(1), minimum: -7, maximum: 9},
      {values, mask: new Uint32Array(5), minimum: Number.NaN, maximum: Number.NaN},
      {
        values: Float32Array.of(8, 2, 1, -3, 5),
        mask: new Uint32Array(5).fill(1),
        minimum: -3,
        maximum: 8
      }
    ]) {
      const actual = await run.run(frame.values, frame.mask);
      const expected = aggregateCellsOnCPU({
        family: 'quadbin',
        resolution: 8,
        keys: getQuadbinPointKeys(positions, 8),
        values: frame.values,
        mask: frame.mask,
        sumScale: SUM_SCALE
      });
      expectTable(actual, expected, 3, 'runtime extrema NaNs');
      expect(await readFloat32(run.table.minimums, 3)).toEqual([
        frame.minimum,
        Number.NaN,
        Number.NaN
      ]);
      expect(await readFloat32(run.table.maximums, 3)).toEqual([
        frame.maximum,
        Number.NaN,
        Number.NaN
      ]);
      expect(await readUint32(run.table.counts, 3)).toEqual(frame.mask[0] ? [5, 0, 0] : [0, 0, 0]);
    }
  } finally {
    run.compiled.destroy();
    fixture.destroy();
  }
});

it('cell aggregation accepts end-of-buffer empty chunks with exact independent cell totals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scratchTileSize of [undefined, 2]) {
    const fixture = new Fixture(device);
    const positions = Float32Array.of(-150, 0, 0, 0, 150, 0);
    const values = Float32Array.of(1, 2, 3);
    const mask = Uint32Array.of(1, 1, 1);
    const run = createAggregation(fixture, {
      family: 'quadbin',
      resolution: 8,
      positions,
      values,
      capacity: 2,
      chunkLength: 1,
      emptyInputChunks: true,
      scratchTileSize
    });
    try {
      const expected = aggregateCellsOnCPU({
        family: 'quadbin',
        resolution: 8,
        keys: getQuadbinPointKeys(positions, 8),
        values,
        mask,
        sumScale: SUM_SCALE
      });
      expectTable(await run.run(values, mask), expected, 2, 'empty trailing chunks');
    } finally {
      run.compiled.destroy();
      fixture.destroy();
    }
  }
});

it('cell aggregation automatically tiles scalar scratch beyond a 64 KiB binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new Fixture(device);
  const rows = 20003;
  const distinct = [0, 1, 2, 2 ** 24, 2 ** 25, 2 ** 26 - 1].map(column =>
    quadbinTileToCell(column, 0, 26)
  );
  const keys = Array.from({length: rows}, (_, row) => distinct[row % distinct.length]);
  const cells = new Uint32Array(rows * 2);
  for (const [row, key] of keys.entries()) cells.set(splitCellKey(key), row * 2);
  const values = Float32Array.from({length: rows}, (_, row) => (row % 11) - 5);
  const mask = new Uint32Array(rows).fill(1);
  const run = await withGroupStatisticsTestBindingLimit(device, () =>
    createAggregation(fixture, {
      family: 'quadbin',
      resolution: 26,
      cells,
      values,
      capacity: 3,
      chunkLength: 4096
    })
  );
  try {
    expectTable(
      await run.run(values, mask),
      aggregateCellsOnCPU({
        family: 'quadbin',
        resolution: 26,
        keys,
        values,
        mask,
        sumScale: SUM_SCALE
      }),
      3,
      'beyond one binding'
    );
  } finally {
    run.compiled.destroy();
    fixture.destroy();
  }
});

for (const wordOrder of ['little-endian', 'high-low'] as const) {
  it(`tiled H3 ${wordOrder} merges duplicates across tiles with masks, invalid rows, tails and overflow`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new Fixture(device);
    const distinct = [
      latLngToCell(37.78, -122.4, 7),
      latLngToCell(40.71, -74, 7),
      latLngToCell(51.51, -0.13, 7),
      latLngToCell(-33.87, 151.21, 7),
      getPentagons(7)[0]
    ].map(h3ToBigInt);
    const rows = 515;
    const keys = Array.from({length: rows}, (_, row) =>
      row % 53 === 0 ? 0xffffffffffffffffn : distinct[row % distinct.length]
    );
    const cells = new Uint32Array(rows * 2);
    for (const [row, key] of keys.entries()) {
      const words = splitCellKey(key);
      cells.set(wordOrder === 'high-low' ? [words[1], words[0]] : words, row * 2);
    }
    const values = Float32Array.from({length: rows}, (_, row) => (row % 17) - 8);
    values[31] = Number.NaN;
    values[134] = Number.POSITIVE_INFINITY;
    values[257] = Number.NEGATIVE_INFINITY;
    const mask = Uint32Array.from({length: rows}, (_, row) => (row % 5 === 0 ? 0 : 1));
    const run = createAggregation(fixture, {
      family: 'h3',
      resolution: 6,
      cells,
      wordOrder,
      values,
      capacity: 3,
      scratchTileSize: 64,
      chunkLength: 47
    });
    try {
      for (const currentMask of [mask, new Uint32Array(rows), new Uint32Array(rows).fill(1)]) {
        const expected = aggregateCellsOnCPU({
          family: 'h3',
          resolution: 6,
          keys,
          values,
          mask: currentMask,
          sumScale: SUM_SCALE
        });
        expectTable(await run.run(values, currentMask), expected, 3, `${wordOrder} H3 tiles`);
      }
    } finally {
      run.compiled.destroy();
      fixture.destroy();
    }
  });
}
