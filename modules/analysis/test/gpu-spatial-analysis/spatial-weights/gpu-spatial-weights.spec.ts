// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  GPUContiguityWeights,
  GPUChunkedSpatialWeightsTransform,
  GPULatticeWeights,
  GPUSpatialLag,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {
  assertValidCSR,
  computeContiguityOracle,
  computeLagOracle,
  computeLatticeOracle,
  computeTransformOracle,
  createSeededRandom,
  flattenFeatures,
  flattenPolygons,
  isSymmetricPattern,
  type OracleCSR,
  type OraclePolygons,
  type PolygonLayout
} from './spatial-weights-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {LIBPYSAL_CONSTRUCTION_REFERENCE, type LibpysalRows} from './spatial-weights-u4-oracle';
import {createPairStarvedContiguityFixture} from './contiguity-loss.test-utils';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';

it('final1 fuzzy contiguity clears a reused candidate word before a finite Moran statistic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const fixture = createPairStarvedContiguityFixture();
    const candidate = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const fuzzy = new GPUFuzzyContiguity({
      polygons: {
        ...rig.polygons(fixture.input),
        positions: rig.input(fixture.input.positions, 'float32x2', 8)
      },
      tolerance: 0,
      output: {
        ...rig.weightsOutput(2, 2).spatialWeights,
        status: {overflow: overflow.view, candidateOverflow: candidate.view},
        featureStatus: rig.output('uint32', 2).view,
        added: rig.output('uint32', 2).view
      }
    });
    rig.graph.add(fuzzy);
    const summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
    rig.graph.add(
      new GPUGlobalSpatialStatistics({
        weights: fuzzy.outputWeights,
        values: rig.input(new Float32Array([1, 2]), 'float32', 2),
        statistics: ['moran'],
        output: {summary: summary.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      candidate.buffer.write(new Uint32Array([1]));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
      const value = (
        await readFloat32(summary.buffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)
      )[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran];
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(-1, 6);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('final1 direct contiguity reports pair-only loss separately from sufficient final slots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const fixture = createPairStarvedContiguityFixture();
    const overflow = rig.output('uint32', 1);
    const candidate = rig.output('uint32', 1);
    const required = rig.output('uint32', 1);
    const output = rig.weightsOutput(2, fixture.options.capacity);
    const producer = new GPUContiguityWeights({
      ...fixture.options,
      polygons: rig.polygons(fixture.input),
      output: {
        ...output.spatialWeights,
        status: {
          overflow: overflow.view,
          candidateOverflow: candidate.view,
          requiredCount: required.view
        }
      }
    });
    rig.run(producer);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(candidate.buffer, 1)).toEqual([1]);
    expect(await readUint32(required.buffer, 1)).toEqual([1]);
    expect((await output.read()).neighbors).toEqual([1]);
  } finally {
    rig.destroy();
  }
});

it('final1 lattice clears a stale candidate word before global Moran on every submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const candidate = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const lattice = new GPULatticeWeights({
      width: 2,
      height: 2,
      criterion: 'rook',
      output: {
        ...rig.weightsOutput(4, 8).spatialWeights,
        status: {overflow: overflow.view, candidateOverflow: candidate.view}
      }
    });
    rig.graph.add(lattice);
    const summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
    rig.graph.add(
      new GPUGlobalSpatialStatistics({
        weights: lattice.outputWeights,
        values: rig.input(new Float32Array([1, 2, 4, 8]), 'float32', 4),
        statistics: ['moran'],
        output: {summary: summary.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        candidate.buffer.write(new Uint32Array([1]));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        const result = await readFloat32(
          summary.buffer,
          GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
        );
        expect(Number.isFinite(result[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran])).toBe(true);
        expect(result[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran]).toBeCloseTo(-9 / 115, 5);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

function expectClose(actual: number[], expected: number[], label: string, relative = 2e-5): void {
  expectNumericArrayClose(actual, expected, {
    absoluteTolerance: 1e-6,
    relativeTolerance: relative,
    label
  });
}

/** Squares on a grid; every ring is rotated and half are reversed so vertex order is varied. */
function createSquareGrid(columns: number, rows: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const polygons: OraclePolygons = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      let ring: [number, number][] = [
        [x, y],
        [x + 1, y],
        [x + 1, y + 1],
        [x, y + 1]
      ];
      const rotation = Math.floor(random() * 4);
      ring = [...ring.slice(rotation), ...ring.slice(0, rotation)];
      if (random() < 0.5) ring.reverse();
      polygons.push([ring]);
    }
  }
  return polygons;
}

/** Random axis-aligned integer rectangles: overlapping, T-junctions and partial shared edges. */
function createRandomRectangles(count: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const polygons: OraclePolygons = [];
  for (let index = 0; index < count; index++) {
    const x = Math.floor(random() * 16);
    const y = Math.floor(random() * 16);
    const width = 1 + Math.floor(random() * 3);
    const height = 1 + Math.floor(random() * 3);
    polygons.push([
      [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
      ]
    ]);
  }
  return polygons;
}

async function runContiguity(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  polygons: OraclePolygons,
  criterion: 'queen' | 'rook',
  options: {
    snapTolerance?: number;
    capacity: number;
    pairCapacity?: number;
    layout?: PolygonLayout;
    rowCount?: number;
  }
): Promise<{csr: OracleCSR; overflow: number; total: number; candidateOverflow: number}> {
  const layout = options.layout ?? flattenPolygons(polygons);
  const rows = options.rowCount ?? polygons.length;
  const rig = new WeightsRig(device);
  const output = rig.weightsOutput(rows, options.capacity);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('uint32', 1);
  const candidateOverflow = rig.output('uint32', 1);
  rig.run(
    new GPUContiguityWeights({
      criterion,
      polygons: rig.polygons(layout),
      snapTolerance: options.snapTolerance,
      pairCapacity: options.pairCapacity,
      output: {
        ...output.spatialWeights,
        status: {
          overflow: overflow.view,
          requiredCount: total.view,
          candidateOverflow: candidateOverflow.view
        }
      }
    })
  );
  const result = {
    csr: await output.read(),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    total: (await readUint32(total.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

it('GPUContiguityWeights matches the oracle on a square grid for queen and rook', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createSquareGrid(7, 5, 3);
  for (const criterion of ['queen', 'rook'] as const) {
    const expected = computeContiguityOracle(polygons, criterion);
    // Sanity: the interior cell of a 7x5 grid has 8 (queen) or 4 (rook) neighbors.
    const interior = 2 * 7 + 3;
    expect(expected.offsets[interior + 1] - expected.offsets[interior]).toBe(
      criterion === 'queen' ? 8 : 4
    );
    const {csr, overflow, total} = await runContiguity(device, polygons, criterion, {
      capacity: expected.neighbors.length + 16
    });
    assertValidCSR(csr, polygons.length, criterion);
    expect(csr.offsets, `${criterion} offsets`).toEqual(expected.offsets);
    expect(csr.neighbors, `${criterion} neighbors`).toEqual(expected.neighbors);
    expect(csr.weights.every(weight => weight === 1)).toBe(true);
    expect(overflow).toBe(0);
    expect(total).toBe(expected.neighbors.length);
    expect(isSymmetricPattern(csr)).toBe(true);
  }
});

it('GPUContiguityWeights sorts narrow ID keys correctly at polygon counts around powers of two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The pair and edge sorts compare only getSortKeyBits(count) bits; the invalid key then wraps to
  // the largest key value, so the largest valid ID must stay below it.
  for (const [columns, rows] of [
    [3, 5],
    [4, 4],
    [31, 1],
    [32, 1],
    [1, 33]
  ] as const) {
    const polygons = createSquareGrid(columns, rows, columns * 100 + rows);
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = computeContiguityOracle(polygons, criterion);
      const {csr, overflow} = await runContiguity(device, polygons, criterion, {
        capacity: expected.neighbors.length + 4
      });
      const label = `${columns}x${rows} ${criterion}`;
      assertValidCSR(csr, polygons.length, label);
      expect(csr.offsets, `${label} offsets`).toEqual(expected.offsets);
      expect(csr.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
      expect(overflow).toBe(0);
    }
  }
});

it('GPUContiguityWeights matches the oracle on random rectangles (T-junctions, overlaps)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seed of [1, 2, 3]) {
    const polygons = createRandomRectangles(80, seed);
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = computeContiguityOracle(polygons, criterion);
      expect(expected.neighbors.length).toBeGreaterThan(0);
      const {csr} = await runContiguity(device, polygons, criterion, {
        capacity: expected.neighbors.length + 8,
        pairCapacity: 20000
      });
      assertValidCSR(csr, polygons.length, `${criterion} seed ${seed}`);
      expect(csr.offsets, `${criterion} ${seed} offsets`).toEqual(expected.offsets);
      expect(csr.neighbors, `${criterion} ${seed} neighbors`).toEqual(expected.neighbors);
    }
  }
});

it('GPUContiguityWeights handles holes, corner contact and non-finite vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons: OraclePolygons = [
    // 0: square with a hole whose edge is shared with polygon 1
    [
      [
        [0, 0],
        [6, 0],
        [6, 6],
        [0, 6]
      ],
      [
        [2, 2],
        [4, 2],
        [4, 4],
        [2, 4]
      ]
    ],
    // 1: fills the hole (shares the whole hole ring)
    [
      [
        [2, 2],
        [4, 2],
        [4, 4],
        [2, 4]
      ]
    ],
    // 2: touches polygon 0 at the corner (6, 6) only
    [
      [
        [6, 6],
        [8, 6],
        [8, 8],
        [6, 8]
      ]
    ],
    // 3: isolated, with a NaN vertex that must be ignored
    [
      [
        [20, 20],
        [21, 20],
        [Number.NaN, 21],
        [20, 21]
      ]
    ]
  ];
  for (const criterion of ['queen', 'rook'] as const) {
    const expected = computeContiguityOracle(polygons, criterion);
    const {csr} = await runContiguity(device, polygons, criterion, {capacity: 32});
    assertValidCSR(csr, polygons.length, criterion);
    expect(csr.offsets).toEqual(expected.offsets);
    expect(csr.neighbors).toEqual(expected.neighbors);
  }
  const queen = computeContiguityOracle(polygons, 'queen');
  const rook = computeContiguityOracle(polygons, 'rook');
  expect(queen.neighbors.slice(queen.offsets[2], queen.offsets[3])).toEqual([0]);
  expect(rook.neighbors.slice(rook.offsets[2], rook.offsets[3])).toEqual([]);
});

it('GPUContiguityWeights collapses repeated polygon occurrences before emitting pairs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases: {criterion: 'queen' | 'rook'; polygons: OraclePolygons}[] = [
    {
      criterion: 'queen',
      polygons: [
        [
          [
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0]
          ]
        ],
        [
          [
            [0, 0],
            [0, 0],
            [2, 0]
          ]
        ]
      ]
    },
    {
      criterion: 'rook',
      // Every edge has the same undirected key, repeated four times by each polygon.
      polygons: [
        [
          [
            [0, 0],
            [1, 0],
            [0, 0],
            [1, 0]
          ]
        ],
        [
          [
            [0, 0],
            [1, 0],
            [0, 0],
            [1, 0]
          ]
        ]
      ]
    }
  ];
  for (const {criterion, polygons} of cases) {
    const expected = computeContiguityOracle(polygons, criterion);
    expect(expected.neighbors).toEqual([1, 0]);
    const result = await runContiguity(device, polygons, criterion, {
      capacity: 2,
      pairCapacity: 2
    });
    expect(result.csr.offsets, `${criterion} offsets`).toEqual(expected.offsets);
    expect(result.csr.neighbors, `${criterion} neighbors`).toEqual(expected.neighbors);
    expect(result.overflow, `${criterion} overflow`).toBe(0);
    expect(result.total, `${criterion} total`).toBe(2);
  }
});

it('GPUContiguityWeights snapTolerance joins near-coincident vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons: OraclePolygons = [
    [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ]
    ],
    [
      [
        [1.0004, 0.0003],
        [2, 0],
        [2, 1],
        [1.0004, 0.9997]
      ]
    ]
  ];
  const exact = await runContiguity(device, polygons, 'rook', {capacity: 8});
  expect(exact.csr.neighbors).toEqual([]);
  const snapped = await runContiguity(device, polygons, 'rook', {capacity: 8, snapTolerance: 0.01});
  expect(snapped.csr.neighbors).toEqual([1, 0]);
  expect(snapped.csr.neighbors).toEqual(computeContiguityOracle(polygons, 'rook', 0.01).neighbors);
});

// polygon-topology-03, geometry-measures-04: the snap key was floor(x / tolerance + 0.5) on the
// absolute coordinate, clamped at 2^31. Projected coordinates in the millions with a millimetre
// tolerance pass the clamp (5e6 / 1e-3 = 5e9), so every vertex got one key: queen made every
// feature a neighbor of every other and rook, whose edges all became degenerate, found none.
it('GPUContiguityWeights snaps on a grid local to the data at projected coordinates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Three 100 m squares in a row at UTM-like coordinates (exact in f32): A | B | C. A and C do
  // not touch. Expected by inspection: A-B and B-C, for queen and for rook.
  const square = (column: number): OraclePolygons[number] => [
    [
      [5000000 + column * 100, 4500000],
      [5000100 + column * 100, 4500000],
      [5000100 + column * 100, 4500100],
      [5000000 + column * 100, 4500100]
    ]
  ];
  const polygons: OraclePolygons = [square(0), square(1), square(2)];
  for (const criterion of ['queen', 'rook'] as const) {
    const {csr, overflow} = await runContiguity(device, polygons, criterion, {
      capacity: 16,
      snapTolerance: 1e-3
    });
    expect(overflow, criterion).toBe(0);
    expect(csr.offsets, `${criterion} offsets`).toEqual([0, 1, 3, 4]);
    expect(csr.neighbors, `${criterion} neighbors`).toEqual([1, 0, 2, 1]);
  }
  // Snapping still joins near-coincident vertices there: B moved 0.5 m to the right (the f32 step
  // at 5e6) with a tolerance of 2 m.
  const shiftedRing = square(1)[0].map(([x, y]): [number, number] => [x + 0.5, y]);
  const shifted: OraclePolygons = [square(0), [shiftedRing]];
  const shiftedResult = await runContiguity(device, shifted, 'queen', {
    capacity: 16,
    snapTolerance: 2
  });
  // Cells of pitch 2 anchored at x = 5000000: A's right corners at offset 100 are in cell 50, and
  // B's left corners at offset 100.5 are in floor(50.25 + 0.5) = 50 as well, so they are neighbors.
  expect(shiftedResult.csr.neighbors).toEqual([1, 0]);
});

// fr-geometry-08: from 2^23 snap steps the float32 quotient is already an integer, and adding 0.5
// to an odd one is a tie that rounds to even, so floor(steps + 0.5) gave step 8388609 the key of
// step 8388610. The key now uses the quotient as it is from 2^23 on (the coverage snap key).
it('GPUContiguityWeights keeps odd snap steps above 2^23 apart', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Tolerance 1 and the first vertex at the origin, so a coordinate is its snap step. A has a
  // corner at step 8388609 (odd, exact in f32), B a corner at 8388610 and no vertex of A or C,
  // C the corner 8388609 of A. Expected by inspection: A-C only.
  const polygons: OraclePolygons = [
    [
      [
        [0, 0],
        [8388609, 0],
        [0, 10]
      ]
    ],
    [
      [
        [8388610, 0],
        [8388620, 0],
        [8388620, 10]
      ]
    ],
    [
      [
        [8388609, 0],
        [8388609, 20],
        [8388600, 20]
      ]
    ]
  ];
  const {csr, overflow} = await runContiguity(device, polygons, 'queen', {
    capacity: 16,
    snapTolerance: 1
  });
  expect(overflow).toBe(0);
  expect(csr.offsets).toEqual([0, 1, 1, 2]);
  expect(csr.neighbors).toEqual([2, 0]);
});

it('GPUContiguityWeights reports capacity and pair overflow with consistent rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createSquareGrid(6, 6, 9);
  const expected = computeContiguityOracle(polygons, 'queen');
  const capacity = Math.floor(expected.neighbors.length / 2);
  const small = await runContiguity(device, polygons, 'queen', {capacity});
  expect(small.overflow).toBe(1);
  expect(small.candidateOverflow).toBe(0);
  expect(small.total).toBe(expected.neighbors.length);
  expect(small.csr.offsets).toEqual(expected.offsets.map(offset => Math.min(offset, capacity)));
  expect(small.csr.neighbors).toEqual(expected.neighbors.slice(0, capacity));
  const pairs = await runContiguity(device, polygons, 'queen', {
    capacity: expected.neighbors.length,
    pairCapacity: 10
  });
  expect(pairs.overflow).toBe(0);
  expect(pairs.candidateOverflow).toBe(1);
  for (let row = 0; row < polygons.length; row++) {
    expect(pairs.csr.offsets[row + 1]).toBeGreaterThanOrEqual(pairs.csr.offsets[row]);
  }
});

it('GPUContiguityWeights emits one row per multipart feature', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Feature 0 has two parts: [0,1]x[0,1] and [2,3]x[0,1]. Feature 1 is [1,2]x[0,1] and touches
  // both parts by an edge; feature 2 is [0,1]x[1,2] and touches part 0 of feature 0 only.
  const square = (x: number, y: number): [number, number][][] => [
    [
      [x, y],
      [x + 1, y],
      [x + 1, y + 1],
      [x, y + 1]
    ]
  ];
  const features: OraclePolygons[] = [[square(0, 0), square(2, 0)], [square(1, 0)], [square(0, 1)]];
  const layout = flattenFeatures(features);
  for (const criterion of ['queen', 'rook'] as const) {
    const expected = computeContiguityOracle(features.flat(), criterion, 0, [0, 0, 1, 2]);
    const result = await runContiguity(device, features.flat(), criterion, {
      capacity: 16,
      layout,
      rowCount: features.length
    });
    expect(result.overflow, criterion).toBe(0);
    expect(result.csr.offsets, criterion).toEqual(expected.offsets);
    expect(result.csr.neighbors, criterion).toEqual(expected.neighbors);
    expect(result.total, criterion).toBe(expected.neighbors.length);
  }
  // Rook: 0-1 and 0-2 share edges; 1-2 share only the corner (1, 1).
  expect(computeContiguityOracle(features.flat(), 'rook', 0, [0, 0, 1, 2]).neighbors).toEqual([
    1, 2, 0, 0
  ]);
});

it('GPULatticeWeights matches the oracle for rook/queen, radius, mask and distances', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 9;
  const height = 6;
  const random = createSeededRandom(11);
  const mask = new Uint32Array(width * height).map(() => (random() < 0.8 ? 1 : 0));
  const cases = [
    {criterion: 'rook', radius: 1},
    {criterion: 'queen', radius: 1},
    {criterion: 'rook', radius: 2},
    {criterion: 'queen', radius: 3, cellSize: [2, 0.5] as [number, number]},
    {criterion: 'queen', radius: 1, mask},
    {criterion: 'rook', radius: 2, mask}
  ] as const;
  for (const options of cases) {
    const expected = computeLatticeOracle({width, height, ...options} as never);
    const rig = new WeightsRig(device);
    const output = rig.weightsOutput(width * height, expected.neighbors.length + 5, true);
    const overflow = rig.output('uint32', 1);
    rig.run(
      new GPULatticeWeights({
        width,
        height,
        criterion: options.criterion,
        radius: options.radius,
        mask: 'mask' in options ? rig.input(options.mask, 'uint32', width * height) : undefined,
        cellSize: 'cellSize' in options ? options.cellSize : undefined,
        output: {...output.spatialWeights, status: {overflow: overflow.view}}
      })
    );
    const csr = await output.read();
    const label = JSON.stringify({...options, mask: 'mask' in options});
    assertValidCSR(csr, width * height, label);
    expect(csr.offsets, label).toEqual(expected.offsets);
    expect(csr.neighbors, label).toEqual(expected.neighbors);
    expectClose(csr.distances, expected.distances, `${label} distances`);
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    rig.destroy();
  }
  // lat2W sanity: interior rook = 4, queen = 8, corner rook = 2.
  const rook = computeLatticeOracle({width, height, criterion: 'rook'});
  expect(rook.offsets[1] - rook.offsets[0]).toBe(2);
  expect(rook.offsets[width + 2] - rook.offsets[width + 1]).toBe(4);
});

it('GPULatticeWeights flags capacity overflow and clamps offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const expected = computeLatticeOracle({width: 5, height: 5, criterion: 'queen'});
  const capacity = 30;
  const rig = new WeightsRig(device);
  const output = rig.weightsOutput(25, capacity);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('uint32', 1);
  rig.run(
    new GPULatticeWeights({
      width: 5,
      height: 5,
      criterion: 'queen',
      output: {
        ...output.spatialWeights,
        status: {overflow: overflow.view, requiredCount: total.view}
      }
    })
  );
  const csr = await output.read();
  expect((await readUint32(overflow.buffer, 1))[0]).toBe(1);
  expect((await readUint32(total.buffer, 1))[0]).toBe(expected.neighbors.length);
  expect(csr.offsets).toEqual(expected.offsets.map(offset => Math.min(offset, capacity)));
  rig.destroy();
});

const KERNELS: GPUSpatialWeightsKernel[] = [
  'gaussian',
  'triangular',
  'epanechnikov',
  'bisquare',
  'uniform'
];

it.each(['whole', 'chunked'] as const)('UST-15 preserves one-way support (%s)', async layout => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights({
      offsets: [0, 1, 1],
      neighbors: [1],
      weights: [1],
      distances: []
    });
    const output = rig.output('float32', 1);
    const operation = 'average-reciprocal';
    const producer =
      layout === 'whole'
        ? new GPUSpatialWeightsTransform({operation, weights, output: output.view})
        : new GPUChunkedSpatialWeightsTransform({
            operation,
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: [output.view]
          });
    const result = producer.outputWeights;
    const resultCsr = 'chunks' in result ? result.chunks[0].weights : result;
    rig.run(producer);
    // The directed edge is averaged with zero and its missing reverse slot stays absent.
    expect(await readUint32(rig.bufferOf(resultCsr.offsets), 3)).toEqual([0, 1, 1]);
    expect(await readUint32(rig.bufferOf(resultCsr.neighbors), 1)).toEqual([1]);
    expect(await readFloat32(output.buffer, 1)).toEqual([0.5]);
  } finally {
    rig.destroy();
  }
});

it('GPUSpatialWeightsTransform row, binary, kernel and average-reciprocal match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(5);
  const base = computeLatticeOracle({width: 8, height: 6, criterion: 'queen', radius: 2});
  const weighted: OracleCSR = {
    ...base,
    weights: base.weights.map(() => (random() < 0.15 ? 0 : 0.25 + random() * 4))
  };
  for (const outOfPlace of [true, false]) {
    for (const operation of ['row', 'binary', 'average-reciprocal'] as const) {
      if (operation === 'average-reciprocal' && !outOfPlace) continue;
      const rig = new WeightsRig(device);
      const upload = rig.uploadWeights(weighted, 3);
      const output = rig.output('float32', upload.neighbors.length);
      const target = outOfPlace ? output : undefined;
      rig.run(new GPUSpatialWeightsTransform({operation, weights: upload, output: target?.view}));
      const readBuffer = target ? target.buffer : rig.bufferOf(upload.weights);
      const actual = await readFloat32(readBuffer, weighted.neighbors.length);
      const expected = computeTransformOracle(weighted, operation);
      expectClose(actual, expected, `${operation} outOfPlace=${outOfPlace}`);
      if (operation === 'average-reciprocal') {
        expect(isSymmetricPattern(weighted)).toBe(true);
        // Result is symmetric: w'_ij == w'_ji.
        const symmetric = {...weighted, weights: actual};
        expectClose(computeTransformOracle(symmetric, 'average-reciprocal'), actual, 'idempotent');
      }
      expect(actual.some(value => value !== 0)).toBe(true);
      rig.destroy();
    }
  }
  for (const kernel of KERNELS) {
    for (const bandwidth of ['adaptive', 2.5] as const) {
      const rig = new WeightsRig(device);
      // The rows here stand in for kNN rows: 'adaptive' needs the kNN criterion label.
      const upload: GPUSpatialWeights = {
        ...rig.uploadWeights(base, 3),
        information: {
          rowCount: base.offsets.length - 1,
          transform: 'raw',
          symmetric: 'unknown',
          provenance: {source: 'test', criterion: 'knn'}
        }
      };
      const output = rig.output('float32', upload.neighbors.length);
      rig.run(
        new GPUSpatialWeightsTransform({
          operation: 'kernel',
          kernel,
          bandwidth,
          weights: upload,
          output: output.view
        })
      );
      const actual = await readFloat32(output.buffer, base.neighbors.length);
      const expected = computeTransformOracle(base, 'kernel', {kernel, bandwidth});
      expectClose(actual, expected, `${kernel} ${bandwidth}`, 1e-4);
      expect(actual.some(value => value > 0)).toBe(true);
      rig.destroy();
    }
  }
});

it('GPUSpatialLag matches the oracle, with mask and normalization', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(21);
  const csr = computeLatticeOracle({width: 10, height: 8, criterion: 'queen'});
  const rows = 80;
  const values = new Float32Array(rows).map(() => random() * 10 - 3);
  const mask = new Uint32Array(rows).map(() => (random() < 0.75 ? 1 : 0));
  for (const options of [
    {mask: false, normalize: false},
    {mask: true, normalize: false},
    {mask: false, normalize: true},
    {mask: true, normalize: true}
  ]) {
    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(csr);
    const output = rig.output('float32', rows);
    rig.run(
      new GPUSpatialLag({
        values: rig.input(values, 'float32', rows),
        weights,
        mask: options.mask ? rig.input(mask, 'uint32', rows) : undefined,
        normalize: options.normalize,
        output: output.view
      })
    );
    const actual = await readFloat32(output.buffer, rows);
    const expected = computeLagOracle(
      csr,
      values,
      options.mask ? mask : undefined,
      options.normalize
    );
    expectClose(actual, expected, JSON.stringify(options), 1e-4);
    expect(actual.some(value => value !== 0)).toBe(true);
    rig.destroy();
  }
});

it('contiguity -> transform -> lag chains in one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createSquareGrid(6, 4, 17);
  const layout = flattenPolygons(polygons);
  const expected = computeContiguityOracle(polygons, 'rook');
  const rows = polygons.length;
  const values = new Float32Array(rows).map((_, index) => (index * 7) % 5);
  const rig = new WeightsRig(device);
  const output = rig.weightsOutput(rows, expected.neighbors.length + 4);
  const lag = rig.output('float32', rows);
  rig.run(
    new GPUContiguityWeights({
      criterion: 'rook',
      polygons: rig.polygons(layout),
      output: {...output.spatialWeights, status: {overflow: rig.output('uint32', 1).view}}
    }),
    new GPUSpatialWeightsTransform({operation: 'row', weights: output.spatialWeights}),
    new GPUSpatialLag({
      values: rig.input(values, 'float32', rows),
      weights: output.spatialWeights,
      output: lag.view
    })
  );
  const csr = await output.read();
  const standardized = computeTransformOracle(expected, 'row');
  expectClose(csr.weights, standardized, 'row standardized');
  expectClose(
    await readFloat32(lag.buffer, rows),
    computeLagOracle({...expected, weights: standardized}, values),
    'lag',
    1e-4
  );
  rig.destroy();
});

/** Flattens libpysal rows (neighbors ascending by id) to the CSR offsets and neighbors. */
function flattenLibpysalRows(rows: LibpysalRows): {offsets: number[]; neighbors: number[]} {
  const offsets = [0];
  const neighbors: number[] = [];
  for (const row of rows.neighbors) {
    neighbors.push(...row);
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

it('GPUContiguityWeights equals libpysal Queen and Rook on the square grid, rectangles and holes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [name, scene] of Object.entries(LIBPYSAL_CONSTRUCTION_REFERENCE.contiguity)) {
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = flattenLibpysalRows(scene[criterion]);
      const label = `libpysal ${name} ${criterion}`;
      const {csr, overflow, total} = await runContiguity(device, scene.polygons, criterion, {
        capacity: expected.neighbors.length + 8,
        pairCapacity: 20000
      });
      assertValidCSR(csr, scene.polygons.length, label);
      expect(csr.offsets, `${label} offsets`).toEqual(expected.offsets);
      expect(csr.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
      // libpysal contiguity weights are binary.
      expect(csr.weights, `${label} weights`).toEqual(scene[criterion].weights.flat());
      expect(overflow).toBe(0);
      expect(total).toBe(expected.neighbors.length);
    }
  }
});

it('GPULatticeWeights equals libpysal lat2W for rook and queen', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {width, height} = LIBPYSAL_CONSTRUCTION_REFERENCE.lattice;
  for (const criterion of ['rook', 'queen'] as const) {
    const expected = flattenLibpysalRows(LIBPYSAL_CONSTRUCTION_REFERENCE.lattice[criterion]);
    const rig = new WeightsRig(device);
    const output = rig.weightsOutput(width * height, expected.neighbors.length + 5);
    const overflow = rig.output('uint32', 1);
    rig.run(
      new GPULatticeWeights({
        width,
        height,
        criterion,
        output: {...output.spatialWeights, status: {overflow: overflow.view}}
      })
    );
    const csr = await output.read();
    const label = `libpysal lat2W ${criterion}`;
    assertValidCSR(csr, width * height, label);
    expect(csr.offsets, `${label} offsets`).toEqual(expected.offsets);
    expect(csr.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
    expect(csr.weights, `${label} weights`).toEqual(
      LIBPYSAL_CONSTRUCTION_REFERENCE.lattice[criterion].weights.flat()
    );
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    rig.destroy();
  }
});
