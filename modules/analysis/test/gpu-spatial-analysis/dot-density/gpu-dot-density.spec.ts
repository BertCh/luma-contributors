// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUDotDensityParameterValues,
  GPUDotDensity,
  GPURandomPointsInPolygon,
  type GPUDotDensitySettings
} from '../../../src/gpu-spatial-analysis/dot-density/index';
import {generateDotCountsOnCPU, type DotDensityCPUResult} from './dot-density-cpu';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createPolygonColumns,
  isInSceneHole,
  SCENE_FEATURES,
  type SceneFeature
} from './dot-density-scene';
import shapelyFixtureJson from '../../oracles/fixtures/geometry-dot-density.json';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * Dots are sampled in f32 inside GPU-triangulated polygons, so a dot on a polygon edge can be off
 * by a few ulp of the coordinate (below 100, so about 1e-5). The grid lookup of the shapely
 * fixture accepts a dot within this distance of an inside cell.
 */
const EDGE_TOLERANCE = 1e-4;

it('B1 negative mask steps preserve the normalized sampling distribution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features: SceneFeature[] = [
    [
      [
        [
          [0.25, 0.25],
          [0.75, 0.25],
          [0.75, 7.75],
          [0.25, 7.75]
        ]
      ]
    ]
  ];
  const run = async (northUp: boolean) => {
    const fixture = createFixture(device, {
      features,
      values: Float32Array.of(1024),
      capacity: 1024,
      mask: {
        width: 1,
        height: 8,
        normalizeFeatures: true,
        weights: Float32Array.from(
          northUp
            ? [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
            : [0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.25]
        )
      }
    });
    try {
      return await fixture.run({
        seed: 9,
        maskPlacement: {origin: [0, northUp ? 8 : 0], cellSize: [1, northUp ? -1 : 1]}
      });
    } finally {
      fixture.destroy();
    }
  };
  const positive = await run(false);
  const negative = await run(true);
  expect(negative.failedCount).toBe(0);
  expect(Array.from(negative.positions).every(Number.isFinite)).toBe(true);
  expect(Array.from(negative.positions)).toEqual(Array.from(positive.positions));
  // Equal-area halves, normalized acceptance weights 1 and 1/3 => lower half probability 3/4.
  const lowerCount = Array.from(
    {length: negative.count},
    (_, row) => negative.positions[2 * row + 1]
  ).filter(y => y < 4).length;
  expect(lowerCount).toBeGreaterThan(0.7 * 1024);
  expect(lowerCount).toBeLessThan(0.8 * 1024);
});

it('OUT-13 slotStarts delimit both polygon samplers after final capacity clamping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two positive-area right triangles, each requesting exactly two dots.
  const features: SceneFeature[] = [
    [
      [
        [
          [0, 0],
          [2, 0],
          [0, 2]
        ]
      ]
    ],
    [
      [
        [
          [10, 0],
          [12, 0],
          [10, 2]
        ]
      ]
    ]
  ];
  for (const form of ['values', 'counts'] as const) {
    const fixture = createFixture(device, {
      features,
      capacity: 3,
      ...(form === 'values' ? {values: Float32Array.of(2, 2)} : {counts: Uint32Array.of(2, 2)})
    });
    try {
      const result = await fixture.run({seed: 3});
      expect(Array.from(result.slotStarts)).toEqual([0, 2]);
      expect(Array.from(result.slotCounts)).toEqual([2, 2]);
      expect(result.count).toBe(3);
      expect(result.requiredCount).toBe(4);
      expect(result.overflow).toBe(1);
      expect(Array.from(result.featureIds)).toEqual([0, 0, 1]);
      const intervals = Array.from(result.slotStarts, (start, slot) => [
        Math.min(start, result.count),
        Math.min(start + result.slotCounts[slot], result.count)
      ]);
      expect(intervals).toEqual([
        [0, 2],
        [2, 3]
      ]);
      for (const [start, end] of intervals) {
        const coordinates = result.positions.subarray(2 * start, 2 * end);
        expect(coordinates.length).toBe(2 * (end - start));
        expect(Array.from(coordinates).every(Number.isFinite)).toBe(true);
      }
    } finally {
      fixture.destroy();
    }
  }
});

type ShapelyFeature = {
  xs: number[];
  ys: number[];
  inside: boolean[][];
  area: number;
  boundingBoxFill: number;
  rings: number[][][][];
};

/** Shapely `contains` lookup written by test/oracles/geometry-dot-density.py. */
const SHAPELY_FIXTURE = shapelyFixtureJson as unknown as {
  shapely: string;
  features: ShapelyFeature[];
};

function findCell(edges: number[], value: number): number {
  for (let cell = 0; cell < edges.length - 1; cell++) {
    if (value >= edges[cell] && value < edges[cell + 1]) {
      return cell;
    }
  }
  return -1;
}

/** Whether a point is inside the shapely-contained cells of a feature, within EDGE_TOLERANCE. */
function isInsideShapelyFeature(feature: ShapelyFeature, x: number, y: number): boolean {
  for (const deltaX of [-EDGE_TOLERANCE, 0, EDGE_TOLERANCE]) {
    for (const deltaY of [-EDGE_TOLERANCE, 0, EDGE_TOLERANCE]) {
      const column = findCell(feature.xs, x + deltaX);
      const row = findCell(feature.ys, y + deltaY);
      if (column >= 0 && row >= 0 && feature.inside[row][column]) {
        return true;
      }
    }
  }
  return false;
}

type Mask = {
  weights: Float32Array;
  width: number;
  height: number;
  normalizeFeatures?: boolean;
};

type GPUResult = DotDensityCPUResult & {failedCount: number; positions: Float32Array};

type Fixture = {
  run(settings: GPUDotDensitySettings): Promise<GPUResult>;
  destroy(): void;
};

type FixtureProps = {
  features?: SceneFeature[];
  values?: Float32Array;
  counts?: Uint32Array;
  categoryCount?: number;
  capacity: number;
  maximumAttempts?: number;
  mask?: Mask;
};

function createFixture(device: Device, props: FixtureProps): Fixture {
  const polygons = createPolygonColumns(props.features ?? SCENE_FEATURES);
  const featureCount = polygons.featureOffsets.length - 1;
  const categoryCount = props.categoryCount ?? 1;
  const slotCount = featureCount * categoryCount;
  const {capacity} = props;
  const graph = new GPUCommandGraph(device, {id: 'dot-density-graph'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = values.length
      ? createInputBuffer(device, values)
      : createOutputBuffer(device, 2);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'dot-parameters',
    format: 'uint32',
    length: 8
  });
  const outputs = {
    positions: output(capacity * 2),
    ids: output(capacity),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1),
    total: output(1),
    categories: output(capacity),
    failed: output(1),
    slotCounts: output(slotCount),
    slotStarts: output(slotCount)
  };
  const polygonViews = {
    positions: importGraphBuffer(
      graph,
      'polygon-positions',
      input(polygons.positions),
      'float32x2',
      polygons.positions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      input(polygons.featureOffsets),
      'uint32',
      polygons.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      input(polygons.polygonOffsets),
      'uint32',
      polygons.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      input(polygons.ringOffsets),
      'uint32',
      polygons.ringOffsets.length
    )
  };
  const mask = props.mask
    ? {
        weights: importGraphBuffer(
          graph,
          'mask',
          input(props.mask.weights),
          'float32',
          props.mask.weights.length
        ),
        width: props.mask.width,
        height: props.mask.height,
        normalizeFeatures: props.mask.normalizeFeatures
      }
    : undefined;
  const sharedOutput = {
    positions: importGraphBuffer(graph, 'o-positions', outputs.positions, 'float32x2', capacity),
    dots: {
      ids: importGraphBuffer(graph, 'o-ids', outputs.ids, 'uint32', capacity),
      status: {
        count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
        candidateOverflow: importGraphBuffer(
          graph,
          'o-candidate-overflow',
          outputs.candidateOverflow,
          'uint32',
          1
        ),
        requiredCount: importGraphBuffer(graph, 'o-total', outputs.total, 'uint32', 1)
      }
    },
    status: {failedCount: importGraphBuffer(graph, 'o-failed', outputs.failed, 'uint32', 1)},
    slotCounts: importGraphBuffer(graph, 'o-slot-counts', outputs.slotCounts, 'uint32', slotCount),
    slotStarts: importGraphBuffer(graph, 'o-slot-offsets', outputs.slotStarts, 'uint32', slotCount)
  };
  const parameters = parameterBuffer.importToGraph(graph);
  if (props.values) {
    graph.add(
      new GPUDotDensity({
        id: 'dots',
        ...polygonViews,
        values: importGraphBuffer(graph, 'values', input(props.values), 'float32', slotCount),
        categoryCount,
        parameters,
        maximumAttempts: props.maximumAttempts,
        mask,
        output: {
          ...sharedOutput,
          categories: importGraphBuffer(
            graph,
            'o-categories',
            outputs.categories,
            'uint32',
            capacity
          )
        }
      })
    );
  } else {
    graph.add(
      new GPURandomPointsInPolygon({
        id: 'points',
        ...polygonViews,
        counts: importGraphBuffer(graph, 'counts', input(props.counts!), 'uint32', featureCount),
        parameters,
        maximumAttempts: props.maximumAttempts,
        mask,
        output: sharedOutput
      })
    );
  }
  const compiled = graph.compile();
  return {
    async run(settings) {
      parameterBuffer.write(getGPUDotDensityParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      const slotStarts = Uint32Array.from(await readUint32(outputs.slotStarts, slotCount));
      return {
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        candidateOverflow: (await readUint32(outputs.candidateOverflow, 1))[0],
        requiredCount: (await readUint32(outputs.total, 1))[0],
        failedCount: (await readUint32(outputs.failed, 1))[0],
        slotCounts: Uint32Array.from(await readUint32(outputs.slotCounts, slotCount)),
        slotStarts,
        positions: Float32Array.from(await readFloat32(outputs.positions, capacity * 2)),
        featureIds: Uint32Array.from(await readUint32(outputs.ids, capacity)),
        categories: Uint32Array.from(await readUint32(outputs.categories, capacity))
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectMatchesOracle(
  actual: GPUResult,
  expected: DotDensityCPUResult,
  checkCategories: boolean
): void {
  expect(Array.from(actual.slotCounts)).toEqual(Array.from(expected.slotCounts));
  expect(Array.from(actual.slotStarts)).toEqual(Array.from(expected.slotStarts));
  expect(actual.requiredCount).toBe(expected.requiredCount);
  expect(actual.count).toBe(expected.count);
  expect(actual.overflow).toBe(expected.overflow);
  expect(actual.candidateOverflow).toBe(expected.candidateOverflow);
  expect(Array.from(actual.featureIds.slice(0, actual.count))).toEqual(
    Array.from(expected.featureIds)
  );
  if (checkCategories) {
    expect(Array.from(actual.categories.slice(0, actual.count))).toEqual(
      Array.from(expected.categories)
    );
  }
  // Rows past the count are cleared.
  for (let dot = actual.count; dot < actual.featureIds.length; dot++) {
    expect(actual.featureIds[dot]).toBe(0xffffffff);
    expect(Number.isNaN(actual.positions[2 * dot])).toBe(true);
  }
}

/** Counts of dots with a NaN position among the first `count` dots. */
function countMissingDots(actual: GPUResult): number {
  let missing = 0;
  for (let dot = 0; dot < actual.count; dot++) {
    if (Number.isNaN(actual.positions[2 * dot])) {
      missing++;
    }
  }
  return missing;
}

/** Whether a point is inside or on the triangle, within a small tolerance. */
function isInTriangle(
  point: [number, number],
  triangle: [number, number][],
  tolerance: number
): boolean {
  const sides = [0, 1, 2].map(index => {
    const [ax, ay] = triangle[index];
    const [bx, by] = triangle[(index + 1) % 3];
    const length = Math.hypot(bx - ax, by - ay);
    return ((bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)) / length;
  });
  return sides.every(side => side >= -tolerance) || sides.every(side => side <= tolerance);
}

const VALUES = new Float32Array([370.3, 120.6, 80.2, 45.5, 0, Number.NaN]);

/** The shapely fixture features as scene features (rings with `[x, y]` vertices). */
const FIXTURE_SCENE: SceneFeature[] = SHAPELY_FIXTURE.features.map(
  feature => feature.rings as SceneFeature
);

it('GPUDotDensity matches the CPU count oracle, places every dot and avoids holes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const fixture = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 2048
  });
  try {
    for (const settings of [
      {seed: 5, dotsPerUnit: 1},
      {seed: 6, dotsPerUnit: 2.5}
    ]) {
      const actual = await fixture.run(settings);
      const expected = generateDotCountsOnCPU({
        ...polygons,
        values: VALUES,
        categoryCount: 2,
        parameters: getGPUDotDensityParameterValues(settings),
        capacity: 2048
      });
      expectMatchesOracle(actual, expected, true);
      expect(actual.count).toBeGreaterThan(500);
      // Exact area-weighted sampling places every dot of a valid polygon.
      expect(actual.failedCount).toBe(0);
      expect(countMissingDots(actual)).toBe(0);
      for (let dot = 0; dot < actual.count; dot++) {
        expect(isInSceneHole(actual.positions[2 * dot], actual.positions[2 * dot + 1])).toBe(false);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('D-16: dots of a 5% fill C shape, a holed multipolygon and a same-winding hole are all placed inside (shapely)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(SHAPELY_FIXTURE.shapely).toMatch(/^2\./);
  const [cShape, holed, sameWinding] = SHAPELY_FIXTURE.features;
  // The C shape is the 5% bounding-box fill case: 32-attempt rejection lost about 19% of its dots.
  expect(cShape.boundingBoxFill).toBeLessThan(0.06);
  const counts = new Uint32Array([4000, 3000, 2000]);
  const fixture = createFixture(device, {
    features: FIXTURE_SCENE,
    counts,
    capacity: 16384
  });
  try {
    const actual = await fixture.run({seed: 21});
    expect(actual.count).toBe(9000);
    expect(Array.from(actual.slotCounts)).toEqual([4000, 3000, 2000]);
    // Rule 9: counts per feature are exact because no dot is dropped.
    expect(actual.failedCount).toBe(0);
    expect(countMissingDots(actual)).toBe(0);
    const features = [cShape, holed, sameWinding];
    // Per grid cell: dots must be in shapely-contained cells and spread in proportion to area.
    const cellCounts = features.map(feature => feature.inside.map(row => row.map(() => 0)));
    for (let dot = 0; dot < actual.count; dot++) {
      const feature = features[actual.featureIds[dot]];
      const x = actual.positions[2 * dot];
      const y = actual.positions[2 * dot + 1];
      expect(isInsideShapelyFeature(feature, x, y), `dot ${dot} (${x}, ${y})`).toBe(true);
      const column = findCell(feature.xs, x);
      const row = findCell(feature.ys, y);
      if (column >= 0 && row >= 0) {
        cellCounts[actual.featureIds[dot]][row][column]++;
      }
    }
    for (let featureIndex = 0; featureIndex < features.length; featureIndex++) {
      const feature = features[featureIndex];
      const total = counts[featureIndex];
      for (let row = 0; row < feature.ys.length - 1; row++) {
        for (let column = 0; column < feature.xs.length - 1; column++) {
          if (!feature.inside[row][column]) {
            continue;
          }
          const cellArea =
            (feature.xs[column + 1] - feature.xs[column]) * (feature.ys[row + 1] - feature.ys[row]);
          const expectedDots = (total * cellArea) / feature.area;
          // Binomial: 6 standard deviations plus a few edge dots.
          const limit = 6 * Math.sqrt(expectedDots) + 3;
          expect(
            Math.abs(cellCounts[featureIndex][row][column] - expectedDots),
            `feature ${featureIndex} cell ${row},${column}`
          ).toBeLessThan(limit);
        }
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('GPURandomPointsInPolygon keeps points stable when a count grows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const small = createFixture(device, {
    features: FIXTURE_SCENE,
    counts: new Uint32Array([100, 50, 20]),
    capacity: 512
  });
  const large = createFixture(device, {
    features: FIXTURE_SCENE,
    counts: new Uint32Array([300, 50, 20]),
    capacity: 512
  });
  try {
    const before = await small.run({seed: 8});
    const after = await large.run({seed: 8});
    // Feature 0 is the first slot, so its first 100 points are a stable prefix.
    for (let index = 0; index < 200; index++) {
      expect(after.positions[index]).toBe(before.positions[index]);
    }
  } finally {
    small.destroy();
    large.destroy();
  }
});

it('GPUDotDensity keeps dot positions stable when the dot value changes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 4096
  });
  try {
    const coarse = await fixture.run({seed: 11, dotsPerUnit: 0.5});
    const fine = await fixture.run({seed: 11, dotsPerUnit: 3});
    let comparedDots = 0;
    for (let slot = 0; slot < 6; slot++) {
      expect(fine.slotCounts[slot]).toBeGreaterThanOrEqual(coarse.slotCounts[slot]);
      for (let rank = 0; rank < coarse.slotCounts[slot]; rank++) {
        const coarseDot = coarse.slotStarts[slot] + rank;
        const fineDot = fine.slotStarts[slot] + rank;
        // Bitwise equal: same Philox draws on the same triangulation.
        expect(fine.positions[2 * fineDot]).toBe(coarse.positions[2 * coarseDot]);
        expect(fine.positions[2 * fineDot + 1]).toBe(coarse.positions[2 * coarseDot + 1]);
        comparedDots++;
      }
    }
    expect(comparedDots).toBeGreaterThan(200);
    // Replay of the first frame on the same compiled graph is bitwise identical.
    const replay = await fixture.run({seed: 11, dotsPerUnit: 0.5});
    expect(new Uint32Array(replay.positions.buffer)).toEqual(
      new Uint32Array(coarse.positions.buffer)
    );
  } finally {
    fixture.destroy();
  }
});

it('GPUDotDensity clamps to capacity and applies a mask, counting masked-out dots as failed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const small = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 100
  });
  try {
    const settings = {seed: 2, dotsPerUnit: 1};
    const actual = await small.run(settings);
    const expected = generateDotCountsOnCPU({
      ...polygons,
      values: VALUES,
      categoryCount: 2,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 100
    });
    expect(actual.count).toBe(100);
    expect(actual.overflow).toBe(1);
    expect(actual.requiredCount).toBeGreaterThan(100);
    expectMatchesOracle(actual, expected, true);
    expect(actual.failedCount).toBe(0);
  } finally {
    small.destroy();
  }

  const weights = new Float32Array(100);
  for (let row = 0; row < 10; row++) {
    weights.fill(0.75, row * 10 + 5, row * 10 + 10);
  }
  const mask = {weights, width: 10, height: 10};
  const maskedValues = new Float32Array([300, 50, 40]);
  const masked = createFixture(device, {
    values: maskedValues,
    capacity: 1024,
    maximumAttempts: 8,
    mask
  });
  try {
    const settings = {
      seed: 4,
      dotsPerUnit: 1,
      maskPlacement: {origin: [0, 0], cellSize: [1, 1]}
    } satisfies GPUDotDensitySettings;
    const actual = await masked.run(settings);
    const expected = generateDotCountsOnCPU({
      ...polygons,
      values: maskedValues,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 1024
    });
    expectMatchesOracle(actual, expected, false);
    // Features 1 and 2 lie outside the mask: every one of their dots fails. Feature 0 loses a dot
    // with probability (1 - 0.75 * 0.48)^8 (half of the square, 96% outside the hole) = 3%, so 300
    // dots lose 9 +- 3 of them; 40 extra failures is a 10 sigma bound.
    const outsideMask = expected.slotCounts[1] + expected.slotCounts[2];
    expect(actual.failedCount).toBeGreaterThanOrEqual(outsideMask);
    expect(actual.failedCount).toBeLessThan(outsideMask + 40);
    expect(countMissingDots(actual)).toBe(actual.failedCount);
    for (let dot = 0; dot < actual.count; dot++) {
      const x = actual.positions[2 * dot];
      if (!Number.isNaN(x)) {
        expect(actual.featureIds[dot]).toBe(0);
        expect(x).toBeGreaterThanOrEqual(5);
      }
    }
  } finally {
    masked.destroy();
  }
});

it('GPUDotDensity mask.normalizeFeatures keeps counts of low-weight features', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Raster of 60 x 12 unit cells: the square feature (x 0..10) has weight 0.05 on its right half,
  // the triangles (x 20..30) a uniform 0.02, and the sliver feature (x 40..50) has weight 0.
  const width = 60;
  const height = 12;
  const weights = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    weights.fill(0.05, row * width + 5, row * width + 10);
    weights.fill(0.02, row * width + 20, row * width + 30);
  }
  const values = new Float32Array([200, 100, 50]);
  const settings = {
    seed: 9,
    dotsPerUnit: 1,
    maskPlacement: {origin: [0, 0], cellSize: [1, 1]}
  } as const;
  const run = async (normalizeFeatures: boolean) => {
    const fixture = createFixture(device, {
      values,
      capacity: 1024,
      mask: {weights, width, height, normalizeFeatures}
    });
    try {
      return await fixture.run(settings);
    } finally {
      fixture.destroy();
    }
  };
  const absolute = await run(false);
  const normalized = await run(true);
  const requested = Array.from(normalized.slotCounts);
  // Without normalization the small weights lose most dots of features 0 and 1.
  expect(absolute.failedCount).toBeGreaterThan(normalized.failedCount + 100);
  // With it only feature 2 (no positive weight under its box) fails. Features 0 and 1 accept at
  // least half the candidates, so 32 attempts fail with probability below 1e-9.
  expect(normalized.failedCount).toBe(requested[2]);
  expect(normalized.count).toBe(absolute.count);
  for (let dot = 0; dot < normalized.count; dot++) {
    const x = normalized.positions[2 * dot];
    if (normalized.featureIds[dot] === 0) {
      expect(x).toBeGreaterThanOrEqual(5);
      expect(x).toBeLessThanOrEqual(10);
    }
    if (normalized.featureIds[dot] === 1) {
      expect(x).toBeGreaterThanOrEqual(20);
    }
  }
});

it('GPURandomPointsInPolygon matches the count oracle and places every point of a thin sliver', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const counts = new Uint32Array([500, 300, 64]);
  const fixture = createFixture(device, {
    counts,
    capacity: 1024
  });
  try {
    const settings = {seed: 77};
    const actual = await fixture.run(settings);
    const expected = generateDotCountsOnCPU({
      ...polygons,
      counts,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 1024
    });
    expectMatchesOracle(actual, expected, false);
    expect(actual.count).toBe(864);
    // The sliver covers 0.1% of its bounding box, where 3 rejection attempts almost always failed.
    expect(actual.failedCount).toBe(0);
    for (let dot = 0; dot < 500; dot++) {
      const x = actual.positions[2 * dot];
      const y = actual.positions[2 * dot + 1];
      expect(isInSceneHole(x, y)).toBe(false);
      expect(x >= 0 && x <= 10 && y >= 0 && y <= 10).toBe(true);
    }
    const sliver: [number, number][] = [
      [40, 0],
      [50, 10],
      [49.98, 10]
    ];
    for (let dot = 800; dot < 864; dot++) {
      const point: [number, number] = [actual.positions[2 * dot], actual.positions[2 * dot + 1]];
      expect(isInTriangle(point, sliver, 1e-4), `sliver dot ${dot}`).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('D-16: a zero-area feature gets no slots and reports its requested dots as failed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Feature 1 is a collinear shell (zero area); feature 0 is a normal square.
  const features: SceneFeature[] = [
    [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ]
      ]
    ],
    [
      [
        [
          [10, 0],
          [15, 0],
          [20, 0]
        ]
      ]
    ]
  ];
  const fixture = createFixture(device, {
    features,
    counts: new Uint32Array([30, 20]),
    capacity: 64
  });
  try {
    const actual = await fixture.run({seed: 1});
    expect(actual.count).toBe(30);
    expect(actual.requiredCount).toBe(30);
    expect(Array.from(actual.slotCounts)).toEqual([30, 0]);
    expect(actual.failedCount).toBe(20);
    expect(countMissingDots(actual)).toBe(0);
    for (let dot = 0; dot < 30; dot++) {
      expect(Number.isNaN(actual.positions[2 * dot])).toBe(false);
    }
    for (let dot = 30; dot < 50; dot++) {
      expect(Number.isNaN(actual.positions[2 * dot])).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('clip-shaped empty and degenerate rings preserve bitwise polygon samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const empty: number[][] = [];
  const point = [[90, 90]];
  const segment = [
    [90, 90],
    [91, 91]
  ];
  const collinear = [
    [90, 90],
    [91, 91],
    [92, 92],
    [90, 90]
  ];
  const repeated = [
    [90, 90],
    [90, 90],
    [90, 90]
  ];
  // Preserve feature rows (Philox keys), removing only empty polygons and degenerate rings.
  const padded = SCENE_FEATURES.map(feature => [
    [],
    [empty],
    [point],
    [segment],
    [collinear, feature[0][0]],
    ...feature.flatMap(polygon => [
      [polygon[0], empty, point, segment, collinear, repeated, ...polygon.slice(1), empty],
      [],
      [empty]
    ])
  ]);
  for (const source of [
    {counts: new Uint32Array([257, 193, 79])},
    {values: new Float32Array([257.3, 193.7, 79.1])}
  ]) {
    const clean = createFixture(device, {...source, capacity: 600});
    const clipped = createFixture(device, {...source, features: padded, capacity: 600});
    try {
      for (const seed of [1, 43]) {
        const expected = await clean.run({seed});
        const actual = await clipped.run({seed});
        expect(actual.count).toBe(expected.count);
        expect(actual.failedCount).toBe(0);
        expect(actual.slotCounts).toEqual(expected.slotCounts);
        expect(actual.featureIds).toEqual(expected.featureIds);
        expect(new Uint32Array(actual.positions.buffer)).toEqual(
          new Uint32Array(expected.positions.buffer)
        );
      }
    } finally {
      clean.destroy();
      clipped.destroy();
    }
  }
});

it('empty features between live features consume no capacity or Philox ranks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features: SceneFeature[] = [SCENE_FEATURES[0], [], [[], [[]]], SCENE_FEATURES[1]];
  for (const source of [
    {counts: new Uint32Array([17, 23, 29, 31])},
    {values: new Float32Array([17, 23, 29, 31])}
  ]) {
    const actualFixture = createFixture(device, {...source, features, capacity: 48});
    const expectedFixture = createFixture(device, {
      features,
      counts: new Uint32Array([17, 0, 0, 31]),
      capacity: 48
    });
    try {
      const actual = await actualFixture.run({seed: 9});
      const expected = await expectedFixture.run({seed: 9});
      expect(actual.count).toBe(48);
      expect(actual.requiredCount).toBe(48);
      expect(actual.overflow).toBe(0);
      expect(actual.failedCount).toBe(52);
      expect(Array.from(actual.slotCounts)).toEqual([17, 0, 0, 31]);
      expect(new Uint32Array(actual.positions.buffer)).toEqual(
        new Uint32Array(expected.positions.buffer)
      );
    } finally {
      actualFixture.destroy();
      expectedFixture.destroy();
    }
  }
});

it('entirely empty polygon input produces zero points and reports all requests', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const features of [
    [[], []],
    [[[], [[]]], [[]]]
  ] satisfies SceneFeature[][]) {
    for (const source of [{counts: new Uint32Array([3, 5])}, {values: new Float32Array([3, 5])}]) {
      const fixture = createFixture(device, {...source, features, capacity: 8});
      try {
        const actual = await fixture.run({seed: 11});
        expect(actual.count).toBe(0);
        expect(actual.requiredCount).toBe(0);
        expect(actual.overflow).toBe(0);
        expect(actual.failedCount).toBe(8);
        expect(Array.from(actual.slotCounts)).toEqual([0, 0]);
        expect(Array.from(actual.positions).every(Number.isNaN)).toBe(true);
      } finally {
        fixture.destroy();
      }
    }
  }
});
