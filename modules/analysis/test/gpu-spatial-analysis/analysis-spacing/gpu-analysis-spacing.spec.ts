// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUAnalysisSpacing,
  GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
  decodeGPUAnalysisSpacingStatistics
} from '../../../src/gpu-spatial-analysis/analysis-spacing/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/index';
import {
  computeGPUAnalysisSpacingStatistics,
  type GPUAnalysisSpacingStatistics
} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/index';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Fixture = {
  name: string;
  positions: Float32Array;
  geographic?: boolean;
  mask?: Uint32Array;
  rank?: number;
};

function compareStatistics(
  actual: GPUAnalysisSpacingStatistics | undefined,
  fixture: Fixture
): void {
  const {positions, mask, geographic} = fixture;
  const selected: number[] = [];
  for (let row = 0; row < positions.length / 2; row++) {
    if (!mask || mask[row] !== 0) selected.push(positions[row * 2], positions[row * 2 + 1]);
  }
  const expected = computeGPUAnalysisSpacingStatistics(selected, {
    coordinateModel: geographic ? 'geographic' : 'planar',
    neighborRank: fixture.rank ?? 3,
    // The oracle must search the same full population, including the 200k fixture.
    maximumSampleSize: Math.max(2, positions.length / 2)
  });
  if (!expected || !actual) {
    if (expected !== actual)
      throw new Error(
        `${fixture.name}: empty summary mismatch ${JSON.stringify({actual, expected})}`
      );
    return;
  }
  const failures: string[] = [];
  const same = (label: string, result: unknown, reference: unknown) => {
    if (result !== reference) failures.push(`${label}: ${result} != ${reference}`);
  };
  const close = (
    label: string,
    result: number,
    reference: number,
    absolute = geographic ? 3 : 0.00005
  ) => {
    if (
      !Number.isFinite(result) ||
      Math.abs(result - reference) > absolute + Math.abs(reference) * 0.00002
    ) {
      failures.push(`${label}: ${result} != ${reference}`);
    }
  };
  same('pointCount', actual.pointCount, expected.pointCount);
  same('sampleSize', actual.sampleSize, expected.pointCount);
  same('sampled', actual.sampled, false);
  same('duplicates', actual.duplicateLocationCount, expected.duplicateLocationCount);
  same('coordinateModel', actual.coordinateModel, expected.coordinateModel);
  same('unit', actual.unit, expected.unit);
  for (let component = 0; component < 4; component++)
    close(`extent[${component}]`, actual.extent[component], expected.extent[component], 0.00005);
  // Longitudes ±180 describe the same centre.
  let longitudeDifference = actual.center[0] - expected.center[0];
  if (geographic) longitudeDifference -= 360 * Math.round(longitudeDifference / 360);
  const centerTolerance = geographic
    ? 0.00005
    : 0.00005 + Math.max(expected.extentWidth, expected.extentHeight) * 0.000001;
  close('center longitude difference', longitudeDifference, 0, centerTolerance);
  close('center latitude difference', actual.center[1] - expected.center[1], 0, centerTolerance);
  for (const name of [
    'extentWidth',
    'extentHeight',
    'extentDiagonal',
    'standardDistance',
    'medianDistanceFromCenter'
  ] as const) {
    close(name, actual[name], expected[name]);
  }
  same('nearest presence', Boolean(actual.nearestNeighbor), Boolean(expected.nearestNeighbor));
  if (actual.nearestNeighbor && expected.nearestNeighbor) {
    for (const name of [
      'minimum',
      'lowerQuartile',
      'median',
      'upperQuartile',
      'mean',
      'maximum',
      'upperFence'
    ] as const) {
      close(
        `nearest.${name}`,
        actual.nearestNeighbor[name],
        expected.nearestNeighbor[name],
        geographic ? 0.02 : 0.00005
      );
    }
    same(
      'aboveUpperFenceCount',
      actual.nearestNeighbor.aboveUpperFenceCount,
      expected.nearestNeighbor.aboveUpperFenceCount
    );
    same(
      'distinct median presence',
      actual.nearestNeighbor.nonZeroMedian !== undefined,
      expected.nearestNeighbor.nonZeroMedian !== undefined
    );
    if (expected.nearestNeighbor.nonZeroMedian !== undefined)
      close(
        'distinct median',
        actual.nearestNeighbor.nonZeroMedian!,
        expected.nearestNeighbor.nonZeroMedian,
        geographic ? 0.02 : 0.00005
      );
  }
  same('curve presence', Boolean(actual.kDistance), Boolean(expected.kDistance));
  if (actual.kDistance && expected.kDistance) {
    same('curve rank', actual.kDistance.k, expected.kDistance.k);
    same('curve length', actual.kDistance.distances.length, expected.kDistance.distances.length);
    for (let index = 0; index < expected.kDistance.distances.length; index++)
      close(
        `curve[${index}]`,
        actual.kDistance.distances[index],
        expected.kDistance.distances[index],
        geographic ? 0.02 : 0.00005
      );
  }
  if (failures.length)
    throw new Error(
      `${fixture.name}: GPU spacing differs from full CPU reference\n${failures.slice(0, 30).join('\n')}\nGPU ${JSON.stringify(actual)}\nCPU ${JSON.stringify(expected)}`
    );
}

async function createHarness(fixture: Fixture, neighborCapacity?: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const activeBuffers = device.statsManager.getStats('GPU Resource Counts').get('Buffers Active');
  const initialBufferCount = activeBuffers.count;
  const rowCount = fixture.positions.length / 2;
  const positions = createInputBuffer(device, fixture.positions);
  const mask = createInputBuffer(device, fixture.mask ?? new Uint32Array(rowCount).fill(1));
  const identities = createInputBuffer(
    device,
    Uint32Array.from({length: rowCount}, (_, index) => rowCount - index)
  );
  const summary = createOutputBuffer(device, GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
  const graph = new GPUCommandGraph(device, {id: `spacing-test-${fixture.name}`});
  const spatialContext: GPUSpatialContext = fixture.geographic
    ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
    : {coordinateSpace: 'planar', metric: 'native'};
  graph.add(
    new GPUAnalysisSpacing({
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
      mask: importGraphBuffer(graph, 'mask', mask, 'uint32', rowCount),
      identities: importGraphBuffer(graph, 'identities', identities, 'uint32', rowCount),
      spatialContext,
      neighborCapacity,
      neighborRank: fixture.rank,
      gridSize: rowCount > 10000 ? [256, 256] : [16, 16],
      output: {
        summary: importGraphBuffer(
          graph,
          'summary',
          summary,
          'uint32',
          GPU_ANALYSIS_SPACING_SUMMARY_LENGTH
        )
      }
    })
  );
  const compiled = graph.compile();
  let readbackCount = 0;
  return {
    async run(nextPositions?: Float32Array, nextMask?: Uint32Array) {
      if (nextPositions) positions.write(nextPositions);
      if (nextMask) mask.write(nextMask);
      summary.write(new Uint32Array(GPU_ANALYSIS_SPACING_SUMMARY_LENGTH).fill(0x7f7f7f7f));
      const started = performance.now();
      submitGraph(device, compiled);
      // The entire result and every status flag travel through this one bounded readback.
      const bytes = await summary.readAsync();
      readbackCount++;
      const words = new Uint32Array(
        bytes.buffer,
        bytes.byteOffset,
        GPU_ANALYSIS_SPACING_SUMMARY_LENGTH
      ).slice();
      return {words, elapsedMilliseconds: performance.now() - started};
    },
    get readbackCount() {
      return readbackCount;
    },
    destroy() {
      compiled.destroy();
      // Imported buffers remain caller-owned after graph teardown.
      expect(positions.destroyed).toBe(false);
      expect(summary.destroyed).toBe(false);
      positions.destroy();
      mask.destroy();
      identities.destroy();
      summary.destroy();
      expect(activeBuffers.count, `${fixture.name}: leaked graph or caller buffers`).toBe(
        initialBufferCount
      );
    }
  };
}

it('matches the CPU reference on duplicates, ID-ordered kNN rows, degeneracy, masks and large translated f32 positions', async () => {
  const fixtures: Fixture[] = [
    {name: 'rectangle', positions: new Float32Array([0, 0, 3, 0, 0, 4, 3, 4])},
    // Nearest and third-nearest appear in different ID orders across rows.
    {
      name: 'unsorted-distances',
      positions: new Float32Array([0, 0, 9, 0, 1, 0, 4, 0, 2, 0, 100, 0])
    },
    {
      name: 'duplicate-locations',
      positions: new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 10, 0])
    },
    {name: 'all-identical', positions: new Float32Array(40).fill(7)},
    {name: 'singleton', positions: new Float32Array([123, 456])},
    {name: 'fewer-than-k', positions: new Float32Array([0, 0, 2, 0, 10, 0]), rank: 8},
    {
      name: 'invalid-rows',
      positions: new Float32Array([0, 0, NaN, 1, 3, 4, Infinity, 0, 200, 200]),
      mask: new Uint32Array([1, 1, 1, 1, 0])
    },
    {name: 'all-masked', positions: new Float32Array([1, 2, 3, 4]), mask: new Uint32Array(2)},
    {name: 'all-invalid', positions: new Float32Array([NaN, 0, 0, Infinity])},
    {
      name: 'translated-f32',
      positions: new Float32Array([
        10000000, 10000000, 10000003, 10000000, 10000000, 10000004, 10000003, 10000004
      ])
    }
  ];
  for (const fixture of fixtures) {
    const harness = await createHarness(fixture);
    try {
      const result = await harness.run();
      expect(result.words[2], fixture.name).toBe(0);
      if (fixture.name === 'invalid-rows') expect(result.words[1]).toBe(2);
      compareStatistics(decodeGPUAnalysisSpacingStatistics(result.words), fixture);
      expect(harness.readbackCount).toBe(1);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('matches geographic dateline, poles, antipodes, 0–360 longitude and invalid-latitude fixtures', async () => {
  const fixtures: Fixture[] = [
    {
      name: 'dateline',
      positions: new Float32Array([179.5, 1, -179.5, 1, 179, 2, -179, 2, 180, 1, -180, 1])
    },
    {name: 'poles', positions: new Float32Array([0, 90, 130, 90, -70, 90, 0, -90, 0, 85, 30, 85])},
    {name: 'antipodal', positions: new Float32Array([0, 0, 180, 0])},
    {name: 'source-longitudes', positions: new Float32Array([350, 2, 10, 1, 355, 0, 5, -1])},
    {name: 'broad-longitudes', positions: new Float32Array([150, 1, 250, 2, 200, 0, 220, -1])},
    {name: 'geographic-duplicates', positions: new Float32Array([12, 34, 12, 34, 12, 34])},
    {name: 'invalid-latitudes', positions: new Float32Array([0, 90.1, 179, 1, -179, 2, 0, -91])}
  ];
  for (const source of fixtures) {
    const fixture = {...source, geographic: true};
    const harness = await createHarness(fixture);
    try {
      compareStatistics(decodeGPUAnalysisSpacingStatistics((await harness.run()).words), fixture);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it.each([
  {name: 'tiny-negative-longitude', positions: new Float32Array([-0.00001, 0, 0, 0])},
  {
    name: 'close-negative-longitudes',
    positions: new Float32Array([170, 0, -100, 0, -10, 0, -10.00001, 0])
  }
])('preserves geographic precision for $name', async source => {
  const fixture = {...source, geographic: true};
  const harness = await createHarness(fixture);
  try {
    const {words} = await harness.run();
    const statistics = decodeGPUAnalysisSpacingStatistics(words);
    compareStatistics(statistics, fixture);
    expect(statistics?.nearestNeighbor?.minimum).toBeGreaterThan(1);
    expect(words[2]).toBe(0);
    expect(words[27]).toBe(0);
  } finally {
    harness.destroy();
  }
});

it.each([
  1e20, -1e20, 3.4028232635611926e38, -3.4028232635611926e38
])('recognizes stored-f32 longitude %s and its exact remainder as duplicate locations', async longitude => {
  const storedLongitude = Math.fround(longitude);
  const remainder = Number(BigInt(storedLongitude) % 360n);
  const canonical =
    remainder > 180 ? remainder - 360 : remainder <= -180 ? remainder + 360 : remainder;
  const fixture: Fixture = {
    name: `large-longitude-${longitude}`,
    geographic: true,
    positions: new Float32Array([storedLongitude, 0, canonical, 0])
  };
  const harness = await createHarness(fixture);
  try {
    const {words} = await harness.run();
    const statistics = decodeGPUAnalysisSpacingStatistics(words);
    const evidence = JSON.stringify({
      storedLongitude,
      canonical,
      statistics,
      words: Array.from(words.slice(0, 32))
    });
    compareStatistics(statistics, fixture);
    expect(statistics?.duplicateLocationCount, evidence).toBe(1);
    expect(statistics?.nearestNeighbor?.minimum, evidence).toBe(0);
    expect(statistics?.nearestNeighbor?.maximum, evidence).toBe(0);
    expect(words[25], evidence).toBe(1);
    expect(words[2], evidence).toBe(0);
    expect(words[27], evidence).toBe(0);
  } finally {
    harness.destroy();
  }
});

it('reports exact candidate scratch overflow independently of the fixed summary capacity', async () => {
  const fixture: Fixture = {
    name: 'overflow',
    positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0])
  };
  const harness = await createHarness(fixture, 2);
  try {
    const {words} = await harness.run();
    expect(words[2]).toBe(0);
    expect(words[27]).toBe(1);
    expect(words[3]).toBe(12);
    expect(() => decodeGPUAnalysisSpacingStatistics(words)).toThrow(
      /candidateOverflow.*requiredCount 12/
    );
  } finally {
    harness.destroy();
  }
});

it('resubmits one compiled graph with changed coordinates and masks without stale summary words', async () => {
  const fixture: Fixture = {
    name: 'resubmit',
    positions: new Float32Array([0, 0, 3, 0, 0, 4, 3, 4])
  };
  const harness = await createHarness(fixture);
  try {
    const first = (await harness.run()).words;
    compareStatistics(decodeGPUAnalysisSpacingStatistics(first), fixture);
    const next = {
      name: 'resubmit-duplicates',
      positions: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1])
    };
    compareStatistics(
      decodeGPUAnalysisSpacingStatistics((await harness.run(next.positions)).words),
      next
    );
    expect(
      decodeGPUAnalysisSpacingStatistics((await harness.run(undefined, new Uint32Array(4))).words)
    ).toBeUndefined();
    const repeated = (await harness.run(fixture.positions, new Uint32Array(4).fill(1))).words;
    expect(repeated).toEqual(first);
    expect(harness.readbackCount).toBe(4);
  } finally {
    harness.destroy();
  }
});

it('computes full-population spacing on 200,000 uniform, dateline-city and repeated points', async () => {
  const positions = new Float32Array(400000);
  let randomState = 1701;
  for (let index = 0; index < positions.length; index++) {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    positions[index] = (randomState / 4294967296) * 10000;
  }
  const geographicPositions = Float32Array.from(positions, (value, index) => {
    if (index % 2 === 1) return 40 + value / 100000;
    const longitude = 179.95 + value / 100000;
    return longitude > 180 ? longitude - 360 : longitude;
  });
  const fixtures: Fixture[] = [
    {name: '200k-full-population', positions},
    {name: '200k-dateline-city', positions: geographicPositions, geographic: true},
    {name: '200k-identical', positions: new Float32Array(400000).fill(7)}
  ];
  for (const fixture of fixtures) {
    const harness = await createHarness(fixture);
    try {
      const result = await harness.run();
      try {
        compareStatistics(decodeGPUAnalysisSpacingStatistics(result.words), fixture);
      } catch (error) {
        throw new Error(
          `${fixture.name}: ${result.elapsedMilliseconds.toFixed(1)} ms including one 2176-byte readback; ${String(error)}`
        );
      }
      expect(harness.readbackCount).toBe(1);
      expect(result.words[0]).toBe(200000);
      expect(result.words[6]).toBe(512);
    } finally {
      harness.destroy();
    }
  }
}, 180000);
