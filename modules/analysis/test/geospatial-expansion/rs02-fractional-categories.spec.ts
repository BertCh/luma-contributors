// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterFractionalCoverage} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
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
