// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import fixture from '../oracles/fixtures/rs02-fractional-categories.json';

async function createFixture(polygonCoverage = false, areaScale = 1) {
  const testDevice = await getWebGPUTestDevice('core');
  requireTestDevice(testDevice);
  const device = testDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  function allocate<Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    values: number[]
  ) {
    const typedValues = format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values);
    const buffer = createOutputBuffer(device, values.length + 8);
    const padded =
      format === 'uint32'
        ? new Uint32Array(values.length + 8).fill(123)
        : new Float32Array(values.length + 8).fill(123);
    padded.set(typedValues, 4);
    buffer.write(padded);
    const base = importGraphBuffer(graph, `buffer-${buffers.length}`, buffer, format);
    const view = graph.createDataView(base.buffer, {
      format,
      byteOffset: 16,
      length: values.length / (format === 'float32x2' ? 2 : 1)
    });
    buffers.push(buffer);
    return {
      buffer,
      view,
      write(next: number[]) {
        buffer.write(format === 'uint32' ? Uint32Array.from(next) : Float32Array.from(next), 16);
      },
      async read() {
        const data =
          format === 'uint32'
            ? await readUint32(buffer, values.length + 8)
            : await readFloat32(buffer, values.length + 8);
        expect(Array.from(data.slice(0, 4))).toEqual([123, 123, 123, 123]);
        expect(Array.from(data.slice(-4))).toEqual([123, 123, 123, 123]);
        return Array.from(data.slice(4, -4));
      }
    };
  }
  const {zoneCount, categoryCapacity, width, height} = fixture;
  const grid = {
    width,
    height,
    affine: [2 * areaScale, 0, 0, 0, -3, 6] as const,
    pixelInterpretation: 'area' as const
  };
  const values = allocate('uint32', fixture.frames[0].values);
  const mask = allocate('uint32', fixture.frames[0].validity);
  // Spare slots intentionally contain invalid garbage, beyond the active tuple count.
  const tupleCapacity = polygonCoverage
    ? width * height * zoneCount
    : fixture.coverage.zoneIds.length + 3;
  const initialCount = fixture.coverage.zoneIds.length;
  const pad = (data: number[], fill: number) => [
    ...data,
    ...Array.from({length: tupleCapacity - data.length}, () => fill)
  ];
  const zones = allocate('uint32', pad(fixture.coverage.zoneIds, 999));
  const cells = allocate('uint32', pad(fixture.coverage.cellIds, 999));
  const fractions = allocate('float32', pad(fixture.coverage.fractions, NaN));
  const count = allocate('uint32', [initialCount]);
  const requiredCount = allocate('uint32', [initialCount]);
  const overflow = allocate('uint32', [0]);
  const zoneValidity = allocate(
    'uint32',
    Array.from({length: zoneCount}, () => 1)
  );
  const coverage = {
    zoneIds: zones.view,
    cellIds: cells.view,
    fractions: fractions.view,
    zoneValidity: zoneValidity.view,
    status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view}
  };
  if (polygonCoverage) {
    graph.add(
      new GPURasterFractionalCoverage({
        grid,
        zones: {
          positions: allocate('float32x2', fixture.geometry.positions).view,
          ringOffsets: allocate('uint32', fixture.geometry.ringOffsets).view,
          polygonOffsets: allocate('uint32', fixture.geometry.polygonOffsets).view,
          featureOffsets: allocate('uint32', fixture.geometry.featureOffsets).view
        },
        parameters: allocate(
          'float32',
          Array.from(getGPURasterFractionalCoverageParameterValues(grid))
        ).view,
        output: coverage
      })
    );
  }
  const categoryZeros = Array.from({length: zoneCount * categoryCapacity}, () => 0);
  const zoneZeros = Array.from({length: zoneCount}, () => 0);
  const output = {
    coverageSums: allocate('float32', categoryZeros),
    classAreas: allocate('float32', categoryZeros),
    proportions: allocate('float32', categoryZeros),
    coveredArea: allocate('float32', zoneZeros),
    validArea: allocate('float32', zoneZeros),
    missingArea: allocate('float32', zoneZeros),
    majority: allocate('uint32', zoneZeros),
    minority: allocate('uint32', zoneZeros),
    validity: allocate('uint32', zoneZeros)
  };
  graph.add(
    new GPURasterFractionalCategoricalStatistics({
      grid,
      zoneCount,
      categoryCapacity,
      input: {
        id: 'categories',
        format: 'uint32',
        noDataValue: fixture.noDataValue,
        storage: {kind: 'buffer', values: values.view},
        validity: mask.view
      },
      coverage,
      output: {
        coverageSums: output.coverageSums.view,
        classAreas: output.classAreas.view,
        proportions: output.proportions.view,
        coveredArea: output.coveredArea.view,
        validArea: output.validArea.view,
        missingArea: output.missingArea.view,
        majority: output.majority.view,
        minority: output.minority.view,
        validity: output.validity.view
      }
    })
  );
  const compiled = graph.compile();
  return {
    values,
    mask,
    zones,
    cells,
    fractions,
    count,
    requiredCount,
    overflow,
    zoneValidity,
    tupleCapacity,
    output,
    async run() {
      submitGraph(device, compiled, undefined);
      const result: Record<keyof typeof output, number[]> = {
        coverageSums: [],
        classAreas: [],
        proportions: [],
        coveredArea: [],
        validArea: [],
        missingArea: [],
        majority: [],
        minority: [],
        validity: []
      };
      for (const name of Object.keys(output) as (keyof typeof output)[])
        result[name] = await output[name].read();
      return result;
    },
    destroy() {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

// Float32 triangle intersections and shoelace areas add rounding before categorical reduction.
// Measured hole-polygon class area: 12 -> 11.999998092651367 (2 ULP, relative 1.59e-7).
// Budget four float32 epsilons for absolute/relative error only on composed geometry results.
// Direct exactextract tuples retain their stricter absolute bound; discrete outputs stay exact.
function expectFloat(actual: number, expected: number, polygonCoverage: boolean, label: string) {
  const absolute = polygonCoverage ? 4 * 2 ** -23 : 5e-7;
  const relative = polygonCoverage ? 4 * 2 ** -23 : 0;
  expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(
    absolute + relative * Math.abs(expected)
  );
}

function expectReference(
  actual: Awaited<ReturnType<Awaited<ReturnType<typeof createFixture>>['run']>>,
  expected: (typeof fixture.frames)[number]['expected'],
  polygonCoverage = false
) {
  for (const name of Object.keys(expected) as (keyof typeof expected)[]) {
    expect(actual[name].length, name).toBe(expected[name].length);
    for (const [index, value] of expected[name].entries()) {
      if (value === null) expect(actual[name][index], `${name}/${index}`).toBeNaN();
      else if (['majority', 'minority', 'validity'].includes(name))
        expect(actual[name][index], `${name}/${index}`).toBe(value);
      else expectFloat(actual[name][index], value, polygonCoverage, `${name}/${index}`);
    }
  }
  for (let zone = 0; zone < fixture.zoneCount; zone++) {
    const classes = actual.classAreas.slice(
      zone * fixture.categoryCapacity,
      (zone + 1) * fixture.categoryCapacity
    );
    expectFloat(
      classes.reduce((sum, area) => sum + area, 0),
      actual.validArea[zone],
      polygonCoverage,
      `class-area conservation/${zone}`
    );
    expectFloat(
      actual.validArea[zone] + actual.missingArea[zone],
      actual.coveredArea[zone],
      polygonCoverage,
      `missing-area conservation/${zone}`
    );
    if (actual.validity[zone])
      expect(
        actual.proportions
          .slice(zone * fixture.categoryCapacity, (zone + 1) * fixture.categoryCapacity)
          .reduce((sum, proportion) => sum + proportion, 0)
      ).toBeCloseTo(1, 6);
  }
}

async function getComposedExpected(
  context: Awaited<ReturnType<typeof createFixture>>,
  frame: (typeof fixture.frames)[number]
) {
  // Validate the geometry producer independently before interpreting the reducer's outputs.
  const [count] = await context.count.read();
  const zones = (await context.zones.read()).slice(0, count);
  const cells = (await context.cells.read()).slice(0, count);
  const fractions = (await context.fractions.read()).slice(0, count);
  const cellCount = fixture.width * fixture.height;
  const pairCount = fixture.zoneCount * cellCount;
  expect(count).toBeLessThanOrEqual(pairCount);
  expect(await context.requiredCount.read()).toEqual([count]);
  expect(await context.overflow.read()).toEqual([0]);

  // Float32 clipping can emit positive boundary slivers where exactextract has zero support.
  // Compare every zone/cell, including expected zeros, at the unchanged geometry bound; do not
  // drop these represented input weights when independently classifying categorical winners.
  const actualDense = Array<number>(pairCount).fill(0);
  const expectedDense = Array<number>(pairCount).fill(0);
  let previousPair = -1;
  for (let tuple = 0; tuple < count; tuple++) {
    expect(zones[tuple]).toBeLessThan(fixture.zoneCount);
    expect(cells[tuple]).toBeLessThan(cellCount);
    const pair = zones[tuple] * cellCount + cells[tuple];
    expect(pair, `unique ordered zone/cell tuple ${tuple}`).toBeGreaterThan(previousPair);
    expect(fractions[tuple], `positive compact fraction ${tuple}`).toBeGreaterThan(0);
    expect(fractions[tuple], `bounded compact fraction ${tuple}`).toBeLessThanOrEqual(1);
    actualDense[pair] = fractions[tuple];
    previousPair = pair;
  }
  for (let tuple = 0; tuple < fixture.coverage.fractions.length; tuple++) {
    expectedDense[fixture.coverage.zoneIds[tuple] * cellCount + fixture.coverage.cellIds[tuple]] =
      fixture.coverage.fractions[tuple];
  }
  for (let pair = 0; pair < pairCount; pair++) {
    expectFloat(actualDense[pair], expectedDense[pair], true, `polygon coverage pair/${pair}`);
  }

  // This is an independent classification of captured INPUT weights, never GPU result winners.
  // Every zone/class in this fixture has at most two cells: summing in double then rounding once
  // is exactly the float32 class sum, independent of the parallel reduction tree's ordering.
  const supports = Array.from({length: fixture.zoneCount}, () =>
    Array<number>(fixture.categoryCapacity).fill(0)
  );
  const contributions = supports.map(row => row.map(() => 0));
  for (let tuple = 0; tuple < count; tuple++) {
    const cell = cells[tuple];
    const category = frame.values[cell];
    if (!frame.validity[cell] || category === fixture.noDataValue) continue;
    supports[zones[tuple]][category] += fractions[tuple];
    contributions[zones[tuple]][category]++;
  }
  for (const row of contributions) expect(row.every(value => value <= 2)).toBe(true);
  const majority: number[] = [];
  const minority: number[] = [];
  for (const [zone, row] of supports.entries()) {
    const represented = row
      .map((support, category) => ({category, support: Math.fround(support)}))
      .filter(entry => entry.support > 0)
      .sort((left, right) => left.support - right.support || left.category - right.category);
    majority.push(represented.at(-1)?.category ?? 0xffffffff);
    minority.push(represented[0]?.category ?? 0xffffffff);
    if (
      majority[zone] !== frame.expected.majority[zone] ||
      minority[zone] !== frame.expected.minority[zone]
    ) {
      console.log(
        'RS02 represented coverage diagnostic',
        JSON.stringify({
          frame: frame.name,
          zone,
          tuples: fractions.flatMap((fraction, index) =>
            zones[index] === zone ? [{cell: cells[index], fraction}] : []
          ),
          classSupport: represented,
          exactextractMajority: frame.expected.majority[zone],
          representedMajority: majority[zone],
          exactextractMinority: frame.expected.minority[zone],
          representedMinority: minority[zone]
        })
      );
    }
  }
  // All float outputs still compare to the external exactextract geometry reference.
  // Only discrete winners adapt to the represented input weights; validity remains external.
  return {...frame.expected, majority, minority};
}

for (const polygonCoverage of [false, true]) {
  it(
    polygonCoverage
      ? 'RS02 polygon composition retains exactextract geometry bounds and exact represented-weight winners'
      : 'RS02 matches exactextract partial coverage, ties, absent classes and missing area from reference tuples',
    async () => {
      const context = await createFixture(polygonCoverage);
      try {
        for (const frame of [...fixture.frames, fixture.frames[0]]) {
          // Restoring the first frame also verifies that missing/support state is cleared.
          context.values.write(frame.values);
          context.mask.write(frame.validity);
          const actual = await context.run();
          const expected = polygonCoverage
            ? await getComposedExpected(context, frame)
            : frame.expected;
          expectReference(actual, expected, polygonCoverage);
        }
      } finally {
        context.destroy();
      }
    }
  );
}

it('RS02 rejects malformed active inputs and incomplete coverage, then recovers without rebuilding', async () => {
  const context = await createFixture();
  const initialCount = fixture.coverage.zoneIds.length;
  const cases = [
    () => context.values.write([fixture.categoryCapacity, ...fixture.frames[0].values.slice(1)]),
    () => context.fractions.write([NaN, ...fixture.coverage.fractions.slice(1)]),
    () => context.fractions.write([-0.25, ...fixture.coverage.fractions.slice(1)]),
    () => context.fractions.write([1.25, ...fixture.coverage.fractions.slice(1)]),
    () => context.zones.write([fixture.zoneCount, ...fixture.coverage.zoneIds.slice(1)]),
    () =>
      context.cells.write([fixture.width * fixture.height, ...fixture.coverage.cellIds.slice(1)]),
    () => context.overflow.write([1]),
    () => context.count.write([context.tupleCapacity + 1]),
    () => context.requiredCount.write([initialCount + 1])
  ];
  try {
    for (const mutate of cases) {
      mutate();
      const actual = await context.run();
      for (const name of [
        'coverageSums',
        'classAreas',
        'proportions',
        'coveredArea',
        'validArea',
        'missingArea'
      ] as const)
        expect(actual[name].every(Number.isNaN), name).toBe(true);
      expect(actual.validity).toEqual(Array(fixture.zoneCount).fill(0));
      expect(actual.majority).toEqual(Array(fixture.zoneCount).fill(0xffffffff));
      expect(actual.minority).toEqual(Array(fixture.zoneCount).fill(0xffffffff));
      context.values.write(fixture.frames[0].values);
      context.fractions.write(fixture.coverage.fractions);
      context.zones.write(fixture.coverage.zoneIds);
      context.cells.write(fixture.coverage.cellIds);
      context.overflow.write([0]);
      context.count.write([initialCount]);
      context.requiredCount.write([initialCount]);
      expectReference(await context.run(), fixture.frames[0].expected);
    }
    context.zoneValidity.write([0, 1, 1, 1, 1]);
    const invalidZone = await context.run();
    expect(invalidZone.coveredArea[0]).toBeNaN();
    expect(invalidZone.validity[0]).toBe(0);
    expect(invalidZone.coveredArea.slice(1)).toEqual(
      fixture.frames[0].expected.coveredArea.slice(1)
    );
  } finally {
    context.destroy();
  }
});

it('RS02 ignores inactive capacity and zero-coverage classes and preserves support across tuple partitions', async () => {
  const context = await createFixture();
  try {
    // Reordering disjoint tuple partitions preserves every source contribution.
    for (const [allocation, data] of [
      [context.zones, fixture.coverage.zoneIds],
      [context.cells, fixture.coverage.cellIds],
      [context.fractions, fixture.coverage.fractions]
    ] as const)
      allocation.write([...data].reverse());
    expectReference(await context.run(), fixture.frames[0].expected);
    context.count.write([0]);
    context.requiredCount.write([0]);
    const empty = await context.run();
    expect(empty.classAreas).toEqual(Array(fixture.zoneCount * fixture.categoryCapacity).fill(0));
    expect(empty.coveredArea).toEqual(Array(fixture.zoneCount).fill(0));
    expect(empty.validity).toEqual(Array(fixture.zoneCount).fill(0));
    expect(empty.proportions.every(Number.isNaN)).toBe(true);
    context.count.write([fixture.coverage.zoneIds.length]);
    context.requiredCount.write([fixture.coverage.zoneIds.length]);
    context.fractions.write(Array(fixture.coverage.zoneIds.length).fill(0));
    expect((await context.run()).validity).toEqual(Array(fixture.zoneCount).fill(0));
  } finally {
    context.destroy();
  }
});

it('RS02 rejects float32 area overflow per zone without misclassifying it as missing data', async () => {
  const context = await createFixture(false, 1e37);
  try {
    const actual = await context.run();
    // Every input coefficient and cell area is finite; six covered cells overflow total area.
    expect(actual.coveredArea[4]).toBeNaN();
    expect(actual.validArea[4]).toBeNaN();
    expect(actual.missingArea[4]).toBeNaN();
    expect(actual.validity[4]).toBe(0);
    expect(actual.majority[4]).toBe(0xffffffff);
    expect(actual.proportions.slice(4 * fixture.categoryCapacity).every(Number.isNaN)).toBe(true);
    // A smaller zone is still representable, and its missing support is explicit.
    expect(Number.isFinite(actual.coveredArea[0])).toBe(true);
    expect(actual.validity[0]).toBe(1);
    expect(actual.missingArea[0]).toBeGreaterThan(0);
  } finally {
    context.destroy();
  }
});

it('RS02 a zone edge on a cell boundary leaves no sliver when the cell size does not divide exactly (fr-fields-07)', async () => {
  const testDevice = await getWebGPUTestDevice('core');
  requireTestDevice(testDevice);
  const device = testDevice;
  // A 4 x 3 grid of 0.3 x 0.7 cells, rows advancing toward larger y, origin (0, 0). Neither cell
  // size is a float32 number, and the zone corners are the float32 values nearest k * 0.3 and
  // k * 0.7. Their quotients are not integers even in exact arithmetic: float32(0.9) /
  // float32(0.3) is 2.9999998, one float32 step below 3, so a kernel that trusts the quotient
  // puts a sliver of the zone into the next cell whatever the quality of the GPU's division.
  //   zone 0: x in [0.3, 0.9], y in [0.7, 1.4], exactly cells 5 and 6 (row 1, columns 1 and 2)
  //   zone 1: x in [0.9, 1.2], y in [1.4, 2.1], exactly cell 11 (row 2, column 3), which is masked
  // Zone 1 therefore has no valid support. A sliver in a neighbouring valid cell would give it
  // one: the failure this spec had before the coverage kernel snapped pixel coordinates.
  const cellWidth = 0.3;
  const cellHeight = 0.7;
  const grid = {
    width: 4,
    height: 3,
    affine: [cellWidth, 0, 0, 0, cellHeight, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const getRectangle = (left: number, bottom: number, right: number, top: number) => [
    left * cellWidth,
    bottom * cellHeight,
    right * cellWidth,
    bottom * cellHeight,
    right * cellWidth,
    top * cellHeight,
    left * cellWidth,
    top * cellHeight
  ];
  const positions = Float32Array.from([...getRectangle(1, 1, 3, 2), ...getRectangle(3, 2, 4, 3)]);
  // The premise of the test: the float32 quotient is not the integer.
  expect(Math.fround(Math.fround(3 * cellWidth) / Math.fround(cellWidth))).not.toBe(3);
  const zoneCount = 2;
  const categoryCapacity = 2;
  const cellCount = grid.width * grid.height;
  const tupleCapacity = zoneCount * cellCount;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `sliver-input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `sliver-output-${identifier++}`, buffer, format, length)
    };
  };
  const zoneIds = output(tupleCapacity, 'uint32');
  const cellIds = output(tupleCapacity, 'uint32');
  const fractions = output(tupleCapacity, 'float32');
  const count = output(1, 'uint32');
  const requiredCount = output(1, 'uint32');
  const overflow = output(1, 'uint32');
  const zoneValidity = output(zoneCount, 'uint32');
  const coverage = {
    zoneIds: zoneIds.view,
    cellIds: cellIds.view,
    fractions: fractions.view,
    zoneValidity: zoneValidity.view,
    status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view}
  };
  graph.add(
    new GPURasterFractionalCoverage({
      id: 'sliver-coverage',
      grid,
      zones: {
        positions: input(positions, 'float32x2'),
        ringOffsets: input(Uint32Array.of(0, 4, 8), 'uint32'),
        polygonOffsets: input(Uint32Array.of(0, 1, 2), 'uint32'),
        featureOffsets: input(Uint32Array.of(0, 1, 2), 'uint32')
      },
      parameters: input(getGPURasterFractionalCoverageParameterValues(grid), 'float32'),
      output: coverage
    })
  );
  const categories = Uint32Array.of(0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0);
  const mask = new Uint32Array(cellCount).fill(1);
  mask[11] = 0;
  const statistics = {
    coverageSums: output(zoneCount * categoryCapacity, 'float32'),
    classAreas: output(zoneCount * categoryCapacity, 'float32'),
    proportions: output(zoneCount * categoryCapacity, 'float32'),
    coveredArea: output(zoneCount, 'float32'),
    validArea: output(zoneCount, 'float32'),
    missingArea: output(zoneCount, 'float32'),
    majority: output(zoneCount, 'uint32'),
    minority: output(zoneCount, 'uint32'),
    validity: output(zoneCount, 'uint32')
  };
  graph.add(
    new GPURasterFractionalCategoricalStatistics({
      id: 'sliver-categories',
      grid,
      zoneCount,
      categoryCapacity,
      input: {
        id: 'sliver-category-band',
        format: 'uint32',
        storage: {kind: 'buffer', values: input(categories, 'uint32')},
        validity: input(mask, 'uint32')
      },
      coverage,
      output: {
        coverageSums: statistics.coverageSums.view,
        classAreas: statistics.classAreas.view,
        proportions: statistics.proportions.view,
        coveredArea: statistics.coveredArea.view,
        validArea: statistics.validArea.view,
        missingArea: statistics.missingArea.view,
        majority: statistics.majority.view,
        minority: statistics.minority.view,
        validity: statistics.validity.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Exactly the three covered cells, each whole: no tuple for any neighbour.
    expect(await readUint32(requiredCount.buffer, 1)).toEqual([3]);
    expect(await readUint32(count.buffer, 1)).toEqual([3]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(zoneIds.buffer, 3)).toEqual([0, 0, 1]);
    expect(await readUint32(cellIds.buffer, 3)).toEqual([5, 6, 11]);
    expect(await readFloat32(fractions.buffer, 3)).toEqual([1, 1, 1]);
    expect(await readUint32(zoneValidity.buffer, zoneCount)).toEqual([1, 1]);
    // Zone 0: one cell of each category. Zone 1: only the masked cell, so no valid support.
    const cellArea = Math.fround(cellWidth) * Math.fround(cellHeight);
    expect(await readUint32(statistics.validity.buffer, zoneCount)).toEqual([1, 0]);
    expect(await readUint32(statistics.majority.buffer, zoneCount)).toEqual([1, 0xffffffff]);
    expect(await readUint32(statistics.minority.buffer, zoneCount)).toEqual([0, 0xffffffff]);
    expect(await readFloat32(statistics.coverageSums.buffer, 4)).toEqual([1, 1, 0, 0]);
    const proportions = await readFloat32(statistics.proportions.buffer, 4);
    expect(proportions.slice(0, 2)).toEqual([0.5, 0.5]);
    expect(proportions[2]).toBeNaN();
    expect(proportions[3]).toBeNaN();
    const classAreas = await readFloat32(statistics.classAreas.buffer, 4);
    const missingArea = await readFloat32(statistics.missingArea.buffer, zoneCount);
    const validArea = await readFloat32(statistics.validArea.buffer, zoneCount);
    for (const [actual, expected] of [
      [classAreas[0], cellArea],
      [classAreas[1], cellArea],
      [classAreas[2], 0],
      [classAreas[3], 0],
      [missingArea[0], 0],
      [missingArea[1], cellArea],
      [validArea[0], 2 * cellArea],
      [validArea[1], 0]
    ]) {
      expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-6);
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
