// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/index';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUInverseDistanceWeightingParameterValues,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  type GPUInverseDistanceWeightingSettings
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {GPUInverseDistanceWeighting} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {GPUTerrainContours} from '../../../src/gpu-terrain/terrain-analysis/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {countContourSegments} from '../../gpu-terrain/terrain-analysis/terrain-analysis-oracle';
import {makeExtentGrid} from './extent-grid';
import {GREAT_CIRCLE_REFERENCE} from './great-circle-reference';
import {interpolateInverseDistanceWeightingOnCPU} from './spatial-interpolation-oracle';
import {requireTestDevice} from '../../../src/testing/require-test-device';

/**
 * GPU versus float64 oracle tolerance: `|gpu - cpu| <= 2e-5 * valueScale`, where `valueScale` is
 * the largest absolute sample value. Covers f32 log2/exp2 weights and f32 accumulation over up to
 * a few hundred contributors.
 */
const RELATIVE_TOLERANCE = 2e-5;

type Scene = {
  positions: Float32Array;
  values: Float32Array;
  mask?: Uint32Array;
  width: number;
  height: number;
  indexGridSize: [number, number];
  indexBounds: [number, number, number, number];
  maximumNeighborCount: number;
  invocationTileSize?: number;
  spatialContext?: GPUSpatialContext;
  longitudeSearch?: 'clipped' | 'wrapped';
};

type Result = {values: Float32Array; counts: Uint32Array};

/**
 * Per-frame settings plus the frame's raster rectangle. The contributor reads the placement from
 * its grid, so `run` builds a grid of the scene size over `extent` (rows advancing north, which
 * packs `extent` unchanged) and `expectParity` passes `extent` to the oracle.
 */
type FrameSettings = GPUInverseDistanceWeightingSettings & {
  extent: readonly [number, number, number, number];
  /** Row order of the frame's grid. Defaults to rows advancing north. */
  rowDirection?: 'north' | 'south';
};

type Fixture = {
  run(settings: FrameSettings): Promise<Result>;
  /** Graph compilations after the first; per-frame changes must keep it at zero. */
  rebuildCount: number;
  destroy(): void;
};

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

function createFixture(device: Device, scene: Scene): Fixture {
  const sampleCount = scene.values.length;
  const cellCount = scene.width * scene.height;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'idw-parameters',
    format: 'float32',
    length: GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH
  });
  const valuesOut = track(createOutputBuffer(device, cellCount));
  const countsOut = track(createOutputBuffer(device, cellCount));
  const graph = new GPUCommandGraph(device, {id: 'idw-graph'});
  graph.add(
    new GPUInverseDistanceWeighting({
      id: 'idw',
      spatialContext: scene.spatialContext,
      longitudeSearch: scene.longitudeSearch,
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, scene.positions)),
        'float32x2',
        sampleCount
      ),
      values: importGraphBuffer(
        graph,
        'values',
        track(createInputBuffer(device, scene.values)),
        'float32',
        sampleCount
      ),
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            sampleCount
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      grid: makeExtentGrid(scene.width, scene.height, [0, 0, 1, 1]),
      indexGridSize: scene.indexGridSize,
      indexBounds: scene.indexBounds,
      maximumNeighborCount: scene.maximumNeighborCount,
      invocationTileSize: scene.invocationTileSize,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        counts: importGraphBuffer(graph, 'counts-out', countsOut, 'uint32', cellCount)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      const {extent, rowDirection, ...packed} = settings;
      parameterBuffer.write(
        getGPUInverseDistanceWeightingParameterValues(
          packed,
          makeExtentGrid(scene.width, scene.height, extent, rowDirection)
        )
      );
      submitGraph(device, compiled, undefined);
      return {
        values: Float32Array.from(await readFloat32(valuesOut, cellCount)),
        counts: Uint32Array.from(await readUint32(countsOut, cellCount))
      };
    },
    get rebuildCount() {
      return compileCount - 1;
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

function expectParity(actual: Result, scene: Scene, settings: FrameSettings): Result {
  const expected = interpolateInverseDistanceWeightingOnCPU({
    ...scene,
    extent: settings.extent,
    searchRadius: settings.searchRadius,
    power: settings.power ?? 2,
    neighborCount: Math.min(settings.neighborCount ?? 0, scene.maximumNeighborCount),
    minimumNeighborCount: settings.minimumNeighborCount ?? 1,
    taper: settings.taper
  });
  expect(Array.from(actual.counts)).toEqual(Array.from(expected.counts));
  let valueScale = 1;
  for (const value of scene.values) {
    if (Number.isFinite(value)) {
      valueScale = Math.max(valueScale, Math.abs(value));
    }
  }
  const tolerance = RELATIVE_TOLERANCE * valueScale;
  let worst = 0;
  for (let cell = 0; cell < expected.values.length; cell++) {
    const gpu = actual.values[cell];
    const cpu = expected.values[cell];
    if (Number.isNaN(cpu) || Number.isNaN(gpu)) {
      expect(Number.isNaN(gpu), `cell ${cell} nodata`).toBe(Number.isNaN(cpu));
      continue;
    }
    worst = Math.max(worst, Math.abs(gpu - cpu));
  }
  expect(worst).toBeLessThanOrEqual(tolerance);
  return expected;
}

function createRandomScene(seed: number, sampleCount: number): Scene {
  const random = createRandom(seed);
  const positions = new Float32Array(sampleCount * 2);
  const values = new Float32Array(sampleCount);
  const mask = new Uint32Array(sampleCount);
  for (let row = 0; row < sampleCount; row++) {
    positions[2 * row] = random() * 100;
    positions[2 * row + 1] = random() * 60;
    // A smooth field plus noise so contours exist.
    values[row] =
      Math.sin(positions[2 * row] / 15) * 40 + positions[2 * row + 1] + (random() - 0.5) * 5;
    mask[row] = random() < 0.1 ? 0 : 1;
  }
  // A few NaN values, non-finite positions, and samples outside the index domain are skipped.
  values[3] = NaN;
  positions[2 * 7] = NaN;
  positions[2 * 11] = 150;
  positions[2 * 13 + 1] = -5;
  return {
    positions,
    values,
    mask,
    width: 40,
    height: 24,
    indexGridSize: [16, 10],
    indexBounds: [0, 0, 100, 60],
    maximumNeighborCount: 12
  };
}

it('GPUInverseDistanceWeighting matches the CPU oracle and changes parameters without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createRandomScene(1, 900);
  const fixture = createFixture(device, scene);
  const extent = [0, 0, 100, 60] as const;
  const frames: FrameSettings[] = [
    {extent, searchRadius: 10, power: 2},
    {extent, searchRadius: 10, power: 0},
    {extent, searchRadius: 10, power: 1},
    {extent, searchRadius: 10, power: 3.5},
    {extent, searchRadius: 4, power: 2},
    {extent, searchRadius: 25, power: 2, minimumNeighborCount: 3},
    {extent, searchRadius: 20, power: 2, neighborCount: 1},
    {extent, searchRadius: 20, power: 2, neighborCount: 5},
    // k above the compile-time capacity is clamped to 12.
    {extent, searchRadius: 20, power: 2, neighborCount: 40},
    {extent, searchRadius: 3, power: 2, minimumNeighborCount: 4},
    // Panned and zoomed extents, partly outside the index domain.
    {extent: [-20, 10, 60, 50], searchRadius: 8, power: 2},
    {
      extent: [30.5, 20.25, 40.5, 26.25],
      searchRadius: 6,
      power: 2.5,
      neighborCount: 8
    },
    {extent, searchRadius: Infinity, power: 2, neighborCount: 6},
    // Franke-Nielson taper, with and without the nearest-k limit.
    {extent, searchRadius: 10, power: 2, taper: 'modified-shepard'},
    {extent, searchRadius: 15, power: 3, neighborCount: 6, taper: 'modified-shepard'},
    {extent, searchRadius: 10, power: 2}
  ];
  let first: Result | undefined;
  for (const settings of frames) {
    const actual = await fixture.run(settings);
    const expected = expectParity(actual, scene, settings);
    first ??= actual;
    // Every frame exercises interpolated (non-NaN) cells.
    expect(expected.values.some(value => !Number.isNaN(value))).toBe(true);
  }
  // Same parameters give a bitwise-identical raster (in-cell IDs are re-sorted every encoding).
  const repeat = await fixture.run(frames[0]);
  expect(Array.from(new Uint32Array(repeat.values.buffer))).toEqual(
    Array.from(new Uint32Array(first!.values.buffer))
  );
  // Radius 3 with minimumNeighborCount 4 leaves nodata cells.
  const sparse = await fixture.run(frames[9]);
  expect(Array.from(sparse.values).some(Number.isNaN)).toBe(true);
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting follows the grid row order: a north-up grid mirrors the rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createRandomScene(7, 300);
  const fixture = createFixture(device, scene);
  const extent = [0, 0, 100, 60] as const;
  const settings = {extent, searchRadius: 12, power: 2, neighborCount: 6};
  const north = await fixture.run(settings);
  const northUp = await fixture.run({...settings, rowDirection: 'south'});
  expectParity(north, scene, settings);
  // The north-up grid's row y is the north-advancing grid's row height - 1 - y. The cell centers
  // are the same real numbers but come from different f32 arithmetic (start + (row + 0.5) * step
  // with a negative step), so compare to the oracle's mirrored raster within the usual tolerance
  // instead of bitwise.
  const {width, height} = scene;
  const valueScale = Math.max(...Array.from(scene.values).filter(Number.isFinite));
  let interpolatedCount = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const mirrored = north.values[(height - 1 - y) * width + x];
      const actual = northUp.values[y * width + x];
      if (Number.isNaN(mirrored)) {
        expect(Number.isNaN(actual), `cell ${x},${y} nodata`).toBe(true);
      } else {
        interpolatedCount++;
        expect(Math.abs(actual - mirrored), `cell ${x},${y}`).toBeLessThanOrEqual(
          RELATIVE_TOLERANCE * valueScale
        );
      }
    }
  }
  expect(interpolatedCount).toBeGreaterThan(0);
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting resolves exact hits and nearest-k ties by the smallest row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Extent [0, 8] x [0, 8] at 8 x 8 puts cell centers on half-integers, exactly representable.
  const positions = Float32Array.from([
    // Rows 0-1: duplicates on center (2.5, 3.5) with different values; row 0 must win.
    2.5, 3.5, 2.5, 3.5,
    // Rows 2-5: four samples at distance 1 around center (5.5, 5.5); row 2 is the tie winner.
    6.5, 5.5, 4.5, 5.5, 5.5, 6.5, 5.5, 4.5,
    // Row 6: masked exact hit on center (0.5, 0.5); row 7: NaN-valued exact hit on (7.5, 7.5).
    0.5, 0.5, 7.5, 7.5
  ]);
  const scene: Scene = {
    positions,
    values: Float32Array.from([11, 99, 1, 2, 3, 4, 50, NaN]),
    mask: Uint32Array.from([1, 1, 1, 1, 1, 1, 0, 1]),
    width: 8,
    height: 8,
    indexGridSize: [4, 4],
    indexBounds: [0, 0, 8, 8],
    maximumNeighborCount: 4
  };
  const fixture = createFixture(device, scene);
  const extent = [0, 0, 8, 8] as const;
  const at = (x: number, y: number) => y * 8 + x;

  let actual = await fixture.run({extent, searchRadius: 1.5, power: 2});
  expectParity(actual, scene, {extent, searchRadius: 1.5, power: 2});
  expect(actual.values[at(2, 3)]).toBe(11);
  expect(actual.counts[at(2, 3)]).toBe(2);
  // Equidistant samples give their mean.
  expect(actual.values[at(5, 5)]).toBeCloseTo(2.5, 5);
  expect(actual.counts[at(5, 5)]).toBe(4);
  // The masked and NaN exact hits are ignored: no other sample is in range.
  expect(Number.isNaN(actual.values[at(0, 0)])).toBe(true);
  expect(Number.isNaN(actual.values[at(7, 7)])).toBe(true);
  expect(actual.counts[at(0, 0)]).toBe(0);

  // k = 1 among four equal distances keeps row 2 (value 1).
  const nearestOne = {extent, searchRadius: 1.5, power: 2, neighborCount: 1};
  actual = await fixture.run(nearestOne);
  expectParity(actual, scene, nearestOne);
  expect(actual.values[at(5, 5)]).toBe(1);
  expect(actual.counts[at(5, 5)]).toBe(1);
  // k = 2 keeps rows 2 and 3.
  const nearestTwo = {extent, searchRadius: 1.5, power: 2, neighborCount: 2};
  actual = await fixture.run(nearestTwo);
  expect(actual.values[at(5, 5)]).toBeCloseTo(1.5, 5);
  // An exact hit wins regardless of minimumNeighborCount.
  actual = await fixture.run({
    extent,
    searchRadius: 0.5,
    power: 2,
    minimumNeighborCount: 5
  });
  expect(actual.values[at(2, 3)]).toBe(11);
  expect(Number.isNaN(actual.values[at(5, 5)])).toBe(true);
  // Invalid radius or power gives an all-nodata raster with zero counts.
  for (const invalid of [
    {extent, searchRadius: -1, power: 2},
    {extent, searchRadius: NaN, power: 2},
    {extent, searchRadius: 2, power: -1}
  ]) {
    actual = await fixture.run(invalid);
    expect(Array.from(actual.values).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual.counts).every(count => count === 0)).toBe(true);
  }
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting stays finite for tiny distances and large powers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Samples about 1.2e-7 and 1e-3 from the single cell center; d^-12 overflows f32 without the
  // log-space accumulator.
  const scene: Scene = {
    positions: Float32Array.from([0.5 + 1e-7, 0.5, 0.501, 0.5]),
    values: Float32Array.from([10, 20]),
    width: 1,
    height: 1,
    indexGridSize: [1, 1],
    indexBounds: [0, 0, 1, 1],
    maximumNeighborCount: 0
  };
  const fixture = createFixture(device, scene);
  const settings = {
    extent: [0, 0, 1, 1] as const,
    searchRadius: 1,
    power: 12
  };
  const actual = await fixture.run(settings);
  expectParity(actual, scene, settings);
  expect(actual.values[0]).toBeCloseTo(10, 4);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting output feeds GPUTerrainContours in the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createRandomScene(4, 600);
  const {width, height} = scene;
  const cellCount = width * height;
  const sampleCount = scene.values.length;
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'idw-contour-parameters',
    format: 'float32',
    length: GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
    values: getGPUInverseDistanceWeightingParameterValues(
      {searchRadius: 4, power: 2, minimumNeighborCount: 2},
      makeExtentGrid(width, height, [0, 0, 100, 60])
    )
  });
  const buffers = {
    positions: createInputBuffer(device, scene.positions),
    values: createInputBuffer(device, scene.values),
    surface: createOutputBuffer(device, cellCount),
    vertices: createOutputBuffer(device, 2 * 2 * 4096),
    segmentCount: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'idw-contours'});
  const surface = importGraphBuffer(graph, 'surface', buffers.surface, 'float32', cellCount);
  graph.add(
    new GPUInverseDistanceWeighting({
      id: 'idw',
      positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', sampleCount),
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32', sampleCount),
      parameters: parameterBuffer.importToGraph(graph),
      grid: makeExtentGrid(width, height, [0, 0, 100, 60]),
      indexGridSize: scene.indexGridSize,
      indexBounds: scene.indexBounds,
      output: {values: surface}
    })
  );
  const level = 30;
  graph.add(
    new GPUTerrainContours({
      grid: {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: {
        id: 'surface',
        format: 'float32',
        storage: {kind: 'buffer', values: surface}
      },
      levels: [
        {
          level,
          vertices: importGraphBuffer(graph, 'vertices', buffers.vertices, 'float32x2', 2 * 4096),
          status: {
            count: importGraphBuffer(graph, 'segment-count', buffers.segmentCount, 'uint32', 1)
          }
        }
      ],
      output: {
        status: {overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const raster = await readFloat32(buffers.surface, cellCount);
  const [segmentCount] = await readUint32(buffers.segmentCount, 1);
  const [overflow] = await readUint32(buffers.overflow, 1);
  const valid = raster.map(value => (Number.isNaN(value) ? 0 : 1));
  expect(valid.some(flag => flag === 0)).toBe(true);
  expect(segmentCount).toBeGreaterThan(10);
  expect(segmentCount).toBe(countContourSegments(raster, width, height, level, valid));
  expect(overflow).toBe(0);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
});

it('GPUInverseDistanceWeighting ring walk matches the oracle on clustered samples and an unbounded radius', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Most samples sit in one corner, a few are far away, and the raster extends past the index
  // domain, so rings stop early near the cluster and run to the domain edge in the empty area.
  const random = createRandom(9);
  const sampleCount = 500;
  const positions = new Float32Array(sampleCount * 2);
  const values = new Float32Array(sampleCount);
  for (let row = 0; row < sampleCount; row++) {
    const isCluster = row < 450;
    positions[2 * row] = isCluster ? random() * 8 : 20 + random() * 80;
    positions[2 * row + 1] = isCluster ? random() * 6 : 10 + random() * 50;
    values[row] = positions[2 * row] + 2 * positions[2 * row + 1] + random();
  }
  // Duplicated positions give equal distances that the row tie-break must order.
  positions.copyWithin(2 * 21, 2 * 20, 2 * 20 + 2);
  positions.copyWithin(2 * 31, 2 * 30, 2 * 30 + 2);
  const scene: Scene = {
    positions,
    values,
    width: 33,
    height: 21,
    indexGridSize: [48, 30],
    indexBounds: [0, 0, 100, 60],
    maximumNeighborCount: 10
  };
  const fixture = createFixture(device, scene);
  for (const settings of [
    {extent: [0, 0, 100, 60] as const, searchRadius: Infinity, power: 2, neighborCount: 7},
    {extent: [-30, -20, 130, 80] as const, searchRadius: Infinity, power: 1.5, neighborCount: 10},
    {extent: [0, 0, 100, 60] as const, searchRadius: 15, power: 2, neighborCount: 3},
    {extent: [0, 0, 100, 60] as const, searchRadius: Infinity, power: 2, neighborCount: 1}
  ]) {
    expectParity(await fixture.run(settings), scene, settings);
  }
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting great-circle mode matches haversine weights at 47 N', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: geo_oracle.py (haversine, R = 6371008.8 m). The radius is in meters; positions are
  // absolute f32 lon/lat (about 0.4 m ulp), so the bound is 2e-3 of the data scale.
  const reference = GREAT_CIRCLE_REFERENCE;
  for (const maximumNeighborCount of [0, 8]) {
    const fixture = createFixture(device, {
      positions: Float32Array.from(reference.positions),
      values: Float32Array.from(reference.values),
      width: reference.width,
      height: reference.height,
      indexGridSize: [4, 4],
      indexBounds: [7.7, 46.8, 8.3, 47.2],
      maximumNeighborCount,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
    });
    const scale = Math.max(...reference.values);
    for (const testCase of reference.idw) {
      const actual = await fixture.run({
        extent: reference.extent as [number, number, number, number],
        searchRadius: testCase.radius,
        power: testCase.power,
        taper: testCase.taper as 'none' | 'modified-shepard'
      });
      for (let cell = 0; cell < actual.values.length; cell++) {
        const expected = testCase.expected[cell];
        if (Number.isNaN(expected)) {
          expect(Number.isNaN(actual.values[cell]), `${testCase.taper} cell ${cell} nodata`).toBe(
            true
          );
        } else {
          expect(
            Math.abs(actual.values[cell] - expected),
            `${testCase.taper} cell ${cell}`
          ).toBeLessThanOrEqual(2e-3 * scale);
        }
      }
    }
    fixture.destroy();
  }
});

it('IDW storage top-k matches the CPU oracle at 65, 256 and 1024, including equal-distance ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cardinal = [
    [1.5, 0.5],
    [-0.5, 0.5],
    [0.5, 1.5],
    [0.5, -0.5]
  ];
  const sampleCount = 1100;
  const scene: Scene = {
    positions: Float32Array.from(
      Array.from({length: sampleCount}, (_, row) => cardinal[row % 4]).flat()
    ),
    values: Float32Array.from({length: sampleCount}, (_, row) => row),
    width: 1,
    height: 1,
    indexGridSize: [4, 4],
    indexBounds: [-1, -1, 2, 2],
    maximumNeighborCount: 1024
  };
  const storage = createFixture(device, scene);
  const privateList = createFixture(device, {...scene, maximumNeighborCount: 64});
  try {
    for (const neighborCount of [1, 16, 64, 65, 256, 1024, 0]) {
      const settings = {extent: [0, 0, 1, 1] as const, searchRadius: 1, power: 2, neighborCount};
      const actual = await storage.run(settings);
      expectParity(actual, scene, settings);
      const kept = neighborCount || sampleCount;
      // Every distance is exactly one; (distance, row) keeps the first k values.
      expect(actual.counts[0]).toBe(kept);
      expect(actual.values[0]).toBeCloseTo((kept - 1) / 2, 3);
      if (neighborCount > 0 && neighborCount <= 64) {
        const previous = await privateList.run(settings);
        expect(new Uint32Array(actual.values.buffer)).toEqual(
          new Uint32Array(previous.values.buffer)
        );
        expect(actual.counts).toEqual(previous.counts);
      }
    }
  } finally {
    storage.destroy();
    privateList.destroy();
  }
});

it('IDW storage top-k reuses scratch across nondivisible gather tiles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = {
    ...createRandomScene(18, 1100),
    width: 5,
    height: 1,
    maximumNeighborCount: 1024,
    invocationTileSize: 2
  };
  const fixture = createFixture(device, scene);
  try {
    for (const neighborCount of [1, 64, 65, 256, 1024, 0]) {
      const settings = {extent: [0, 0, 100, 60] as const, searchRadius: Infinity, neighborCount};
      expectParity(await fixture.run(settings), scene, settings);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  false,
  true
])('IDW storage batches preserve masked, missing and exact-hit semantics (greatCircle=%s)', async greatCircle => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const extent = [7.75, 46.875, 7.90625, 46.90625] as const;
  const positions = Float32Array.from(
    Array.from({length: 130}, (_, row) => [
      7.75 + (row % 13) / 128,
      46.875 + Math.floor(row / 13) / 1024
    ]).flat()
  );
  positions.set([7.765625, 46.890625, 7.765625, 46.890625]);
  positions[10] = NaN;
  const values = Float32Array.from({length: 130}, (_, row) => 7 + (row % 7));
  values[0] = 11;
  values[1] = 99;
  values[3] = NaN;
  const mask = new Uint32Array(130).fill(1);
  mask[4] = 0;
  const scene: Scene = {
    positions,
    values,
    mask,
    width: 5,
    height: 1,
    indexGridSize: [8, 8],
    indexBounds: [7.5, 46.5, 8, 47],
    maximumNeighborCount: 65,
    spatialContext: greatCircle
      ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      : undefined
  };
  const tiled = createFixture(device, {...scene, invocationTileSize: 2});
  const untiled = createFixture(device, scene);
  try {
    for (const searchRadius of [greatCircle ? 1500 : 0.02, Infinity]) {
      const settings = {extent, searchRadius, neighborCount: 65, power: 2};
      const actual = await tiled.run(settings);
      const other = await untiled.run(settings);
      const expectedValues: number[] = [];
      const expectedCounts: number[] = [];
      for (let column = 0; column < 5; column++) {
        const center = [
          extent[0] + ((column + 0.5) * (extent[2] - extent[0])) / 5,
          (extent[1] + extent[3]) / 2
        ];
        const candidates = Array.from({length: values.length}, (_, row) => {
          const deltaLongitude = ((positions[2 * row] - center[0]) * Math.PI) / 180;
          const deltaLatitude = ((positions[2 * row + 1] - center[1]) * Math.PI) / 180;
          const haversine =
            Math.sin(deltaLatitude / 2) ** 2 +
            Math.cos((center[1] * Math.PI) / 180) *
              Math.cos((positions[2 * row + 1] * Math.PI) / 180) *
              Math.sin(deltaLongitude / 2) ** 2;
          const distance = greatCircle
            ? 2 * 6371008.8 * Math.asin(Math.sqrt(Math.min(haversine, 1)))
            : Math.hypot(positions[2 * row] - center[0], positions[2 * row + 1] - center[1]);
          return {row, distance};
        })
          .filter(
            entry =>
              mask[entry.row] !== 0 &&
              Number.isFinite(values[entry.row]) &&
              Number.isFinite(entry.distance) &&
              entry.distance <= searchRadius
          )
          .sort((first, second) => first.distance - second.distance || first.row - second.row)
          .slice(0, 65);
        expectedCounts.push(candidates.length);
        if (!candidates.length) {
          expectedValues.push(NaN);
          continue;
        }
        if (candidates[0].distance === 0) {
          expectedValues.push(values[candidates[0].row]);
          continue;
        }
        const weightSum = candidates.reduce((sum, entry) => sum + 1 / entry.distance ** 2, 0);
        expectedValues.push(
          candidates.reduce((sum, entry) => sum + values[entry.row] / entry.distance ** 2, 0) /
            weightSum
        );
      }
      expect(Array.from(actual.counts)).toEqual(expectedCounts);
      expect(Array.from(other.counts)).toEqual(expectedCounts);
      for (let cell = 0; cell < 5; cell++) {
        for (const result of [actual, other]) {
          if (Number.isNaN(expectedValues[cell]))
            expect(Number.isNaN(result.values[cell])).toBe(true);
          else
            expect(Math.abs(result.values[cell] - expectedValues[cell])).toBeLessThanOrEqual(
              greatCircle ? 0.2 : 0.002
            );
        }
      }
      expect(actual.values[0]).toBe(11);
      expect(new Uint32Array(actual.values.buffer)).toEqual(new Uint32Array(other.values.buffer));
    }
  } finally {
    tiled.destroy();
    untiled.destroy();
  }
});

it('PTS-117 characterizes both clipped seam sides separately from spherical IDW distances', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Equator distances independently R*pi/180 * [.0625,.1875,.1875,.3125]:
  // [6949.6925146,20849.0775438,20849.0775438,34748.4625730] metres.
  // Full inverse-square average = (1+2/9+3/9+4/25)/(1+1/9+1/9+1/25)
  // = 1.3591549295774648. Clipped first two candidates give 1.1.
  for (const sign of [-1, 1]) {
    for (const seam of [false, true]) {
      const shift = seam ? 180 : 0;
      const longitude = (x: number) => sign * (x + shift > 180 ? x + shift - 360 : x + shift);
      const query = longitude(-0.0625);
      const fixture = createFixture(device, {
        positions: Float32Array.from([-0.125, -0.25, 0.125, 0.25].flatMap(x => [longitude(x), 0])),
        values: Float32Array.of(1, 2, 3, 4),
        width: 1,
        height: 1,
        indexGridSize: [8, 2],
        indexBounds: seam ? [-180, -1, 180, 1] : [-1, -1, 1, 1],
        maximumNeighborCount: 4,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      });
      try {
        const result = await fixture.run({
          extent: [query - 0.015625, -0.015625, query + 0.015625, 0.015625],
          searchRadius: 50000,
          power: 2,
          neighborCount: 4
        });
        expect(Array.from(result.counts)).toEqual([seam ? 2 : 4]);
        expect(result.values).toHaveLength(1);
        expect(Number.isFinite(result.values[0])).toBe(true);
        expect(result.values[0]).toBeCloseTo(seam ? 1.1 : 1.3591549295774648, 3);
      } finally {
        fixture.destroy();
      }
    }
  }
});

it('PTS-117 wraps both IDW seam sides and translated controls', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Equator distances independently R*pi/180 * [.0625,.1875,.1875,.3125]:
  // [6949.6925146,20849.0775438,20849.0775438,34748.4625730] metres.
  // Full inverse-square average = (1+2/9+3/9+4/25)/(1+1/9+1/9+1/25)
  // = 1.3591549295774648. Clipped first two candidates give 1.1.
  for (const sign of [-1, 1]) {
    for (const seam of [false, true]) {
      const shift = seam ? 180 : 0;
      const longitude = (x: number) => sign * (x + shift > 180 ? x + shift - 360 : x + shift);
      const query = longitude(-0.0625);
      const fixture = createFixture(device, {
        positions: Float32Array.from([-0.125, -0.25, 0.125, 0.25].flatMap(x => [longitude(x), 0])),
        values: Float32Array.of(1, 2, 3, 4),
        width: 1,
        height: 1,
        indexGridSize: [8, 2],
        indexBounds: seam ? [-180, -1, 180, 1] : [-1, -1, 1, 1],
        maximumNeighborCount: 4,
        longitudeSearch: 'wrapped',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      });
      try {
        const result = await fixture.run({
          extent: [query - 0.015625, -0.015625, query + 0.015625, 0.015625],
          searchRadius: 50000,
          power: 2,
          neighborCount: 4
        });
        expect(Array.from(result.counts)).toEqual([4]);
        expect(result.values).toHaveLength(1);
        expect(Number.isFinite(result.values[0])).toBe(true);
        expect(result.values[0]).toBeCloseTo(1.3591549295774648, 3);
      } finally {
        fixture.destroy();
      }
    }
  }
});

it('PTS-117 wrapped polar caps and coarse overlapping seam intervals visit samples once', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const polar of [false, true]) {
    const query = polar ? [0, 89.9] : [179.9375, 0];
    const fixture = createFixture(device, {
      positions: Float32Array.from(
        polar ? [0, 89, 90, 89, 180, 89, -90, 89] : [179.875, 0, 179.75, 0, -179.875, 0, -179.75, 0]
      ),
      values: Float32Array.of(1, 2, 3, 4),
      width: 1,
      height: 1,
      indexGridSize: [1, 1],
      indexBounds: [-180, -90, 180, 90],
      maximumNeighborCount: 4,
      longitudeSearch: 'wrapped',
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
    });
    try {
      const result = await fixture.run({
        extent: [
          query[0] - 0.015625,
          query[1] - 0.015625,
          query[0] + 0.015625,
          query[1] + 0.015625
        ],
        searchRadius: polar ? 200000 : 50000,
        power: 0,
        neighborCount: 0
      });
      expect(Array.from(result.counts)).toEqual([4]);
      expect(result.values).toHaveLength(1);
      expect(Number.isFinite(result.values[0])).toBe(true);
      expect(result.values[0]).toBeCloseTo(2.5, 5); // Four equal weights: (1+2+3+4)/4.
    } finally {
      fixture.destroy();
    }
  }
});

it('R1 B1 S1 wrapped IDW includes inclusive seam endpoints and multi-turn queries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [query, sample, radius] of [
    [179.99998474121094, -180, 2],
    [-179.99998474121094, 180, 2],
    [-720, 0, 2000],
    [720, 0, 2000],
    [1080, 0, 2000]
  ]) {
    const halfWidth = Math.abs(query) < 180 ? 2 ** -16 : 0.5;
    const fixture = createFixture(device, {
      positions: Float32Array.of(sample, 0),
      values: Float32Array.of(7),
      width: 1,
      height: 1,
      indexGridSize: [12, 1],
      indexBounds: [-180, -1, 180, 1],
      maximumNeighborCount: 4,
      longitudeSearch: 'wrapped',
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
    });
    try {
      const result = await fixture.run({
        extent: [query - halfWidth, -0.5, query + halfWidth, 0.5],
        searchRadius: radius,
        power: 2,
        neighborCount: 0
      });
      expect(result.counts).toHaveLength(1);
      expect(result.values).toHaveLength(1);
      expect(Array.from(result.counts)).toEqual([1]);
      expect(Number.isFinite(result.values[0])).toBe(true);
      // A single represented neighbor always interpolates its observed value.
      expect(result.values[0]).toBe(7);
    } finally {
      fixture.destroy();
    }
  }
});
