// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer, VertexFormat} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT,
  GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK,
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  createInputBuffer,
  createNullWebGPUDevice,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Scenario = {
  name: string;
  positions: number[];
  observations: number[];
  supports: number[];
  supportFormat: 'float32x2' | 'float32x4';
  supportCount: number;
  props: Pick<
    GPULocalKrigingProps,
    | 'mode'
    | 'neighborCount'
    | 'searchRadius'
    | 'means'
    | 'variableCount'
    | 'coregionalization'
    | 'coKrigingMean'
    | 'targetVariable'
    | 'driftCount'
    | 'sampleIndexChunkQueryCount'
  >;
  /** Row-major float32 external drift columns, `driftCount` per sample and per query. */
  sampleDrift?: number[];
  queryDrift?: number[];
};

/** A 9 x 9 integer lattice: every query below has exact distance ties among its neighbors. */
function createLatticeSamples(usable: (row: number) => boolean) {
  const positions: number[] = [];
  const observations: number[] = [];
  for (let y = 0; y < 9; y++) {
    for (let x = 0; x < 9; x++) {
      const row = y * 9 + x;
      positions.push(x, y);
      const value = Math.sin(x * 0.7) + Math.cos(y * 0.4) + 0.05 * row;
      if (!usable(row)) {
        // Rotate through the four ways a sample is excluded: zero validity, NaN value,
        // negative error variance, NaN position.
        const reason = row % 4;
        observations.push(
          reason === 1 ? Number.NaN : value,
          reason === 2 ? -1 : 0.01,
          0,
          reason === 0 ? 0 : 1
        );
        if (reason === 3) {
          positions[positions.length - 2] = Number.NaN;
        }
      } else {
        observations.push(value, 0.01, 0, 1);
      }
    }
  }
  return {positions, observations};
}

// Lattice points, half-cell points (two- and four-way ties) and points outside the sample extent.
const QUERY_POINTS = [
  [4, 4],
  [2.5, 3],
  [3.5, 3.5],
  [0, 0],
  [8, 8],
  [-3, 4],
  [12, 11],
  [6.5, 1.5]
];

const masked = createLatticeSamples(row => row % 5 !== 2);
const sparse = createLatticeSamples(row => [3, 40, 41, 67, 80].includes(row));

const SCENARIOS: Scenario[] = [
  {
    name: 'ordinary, ties and masked samples',
    ...masked,
    supports: QUERY_POINTS.flat(),
    supportFormat: 'float32x2',
    supportCount: 1,
    props: {mode: 'ordinary', neighborCount: 8}
  },
  {
    name: 'universal, weighted two-point supports and a search radius on a tie',
    ...masked,
    supports: QUERY_POINTS.flatMap(([x, y]) => [x - 0.5, y, 1, 0, x + 0.5, y, 1, 0]),
    supportFormat: 'float32x4',
    supportCount: 2,
    props: {mode: 'universal', neighborCount: 12, searchRadius: 2}
  },
  {
    name: 'simple, fewer usable samples than neighborCount',
    ...sparse,
    supports: QUERY_POINTS.flat(),
    supportFormat: 'float32x2',
    supportCount: 1,
    props: {mode: 'simple', neighborCount: 8, means: [0.5]}
  }
];

async function runLocalKriging(
  scenario: Scenario,
  sampleIndex: 'auto' | 'brute-force' | 'grid',
  resolvedSampleIndex: 'brute-force' | 'grid' = sampleIndex === 'auto' ? 'grid' : sampleIndex
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let nextBuffer = 0;
  const input = <Format extends GPUVectorFormat>(
    data: readonly number[],
    format: Format,
    components: number
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? new Uint32Array(data) : new Float32Array(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${nextBuffer++}`,
      buffer,
      format,
      data.length / components
    );
  };
  const queryCount =
    scenario.supports.length /
    (scenario.supportFormat === 'float32x4' ? 4 : 2) /
    scenario.supportCount;
  const neighborCount = scenario.props.neighborCount ?? 16;
  const valuesBuffer = createOutputBuffer(device, queryCount * 4);
  const statusBuffer = createOutputBuffer(device, queryCount);
  const weightsBuffer = createOutputBuffer(device, queryCount * neighborCount);
  buffers.push(valuesBuffer, statusBuffer, weightsBuffer);
  const kriging = new GPULocalKriging({
    ...scenario.props,
    id: `local-kriging-${sampleIndex}`,
    sampleIndex,
    positions: input(scenario.positions, 'float32x2', 2),
    observations: input(scenario.observations, 'float32x4', 4),
    supports: input(
      scenario.supports,
      scenario.supportFormat,
      scenario.supportFormat === 'float32x4' ? 4 : 2
    ),
    supportCount: scenario.supportCount,
    covariance: {model: 'exponential', sill: 1.3, range: 4, nugget: 0.02},
    ...(scenario.sampleDrift && scenario.queryDrift
      ? {
          sampleDrift: input(scenario.sampleDrift, 'float32', 1),
          queryDrift: input(scenario.queryDrift, 'float32', 1)
        }
      : {}),
    output: {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32x4', queryCount),
      status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', queryCount),
      weights: importGraphBuffer(
        graph,
        'weights',
        weightsBuffer,
        'float32',
        queryCount * neighborCount
      )
    }
  });
  expect(kriging.sampleIndex).toBe(resolvedSampleIndex);
  graph.add(kriging);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      values: Array.from(await readFloat32(valuesBuffer, queryCount * 4)),
      status: Array.from(await readUint32(statusBuffer, queryCount)),
      weights: Array.from(await readFloat32(weightsBuffer, queryCount * neighborCount))
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const scenario of SCENARIOS) {
  it(`local kriging grid index matches the exhaustive scan: ${scenario.name}`, async () => {
    const bruteForce = await runLocalKriging(scenario, 'brute-force');
    const indexed = await runLocalKriging(scenario, 'grid');
    // Same neighbors in the same order give the same system, so the results are identical.
    expect(indexed.status).toEqual(bruteForce.status);
    expect(indexed.values).toEqual(bruteForce.values);
    expect(indexed.weights).toEqual(bruteForce.weights);
    expect(bruteForce.status.filter(status => status === 0).length).toBeGreaterThan(4);
  });
}

/** Deterministic uniform values in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

/** Squared distance as float32 arithmetic computes it. */
function getFloat32SquaredDistance(x: number, y: number): number {
  const fround = Math.fround;
  return fround(fround(fround(x) * fround(x)) + fround(fround(y) * fround(y)));
}

/** Random non-lattice samples with a smooth field, a few unusable rows and random queries. */
function createRandomScenario(
  name: string,
  sampleCount: number,
  queryCount: number,
  props: Scenario['props'],
  seed: number
): Scenario {
  const random = createRandom(seed);
  const positions: number[] = [];
  const observations: number[] = [];
  const variableCount = props.mode === 'co-kriging' ? (props.variableCount ?? 2) : 1;
  for (let row = 0; row < sampleCount; row++) {
    const x = random() * 20;
    const y = random() * 20;
    positions.push(x, y);
    const variable = row % variableCount;
    observations.push(
      Math.sin(x * 0.3) + Math.cos(y * 0.2) + variable * 0.5,
      0.01,
      variable,
      row % 23 === 7 ? 0 : 1
    );
  }
  const supports: number[] = [];
  for (let query = 0; query < queryCount; query++)
    supports.push(random() * 22 - 1, random() * 22 - 1);
  const scenario: Scenario = {
    name,
    positions,
    observations,
    supports,
    supportFormat: 'float32x2',
    supportCount: 1,
    props
  };
  if (props.driftCount) {
    const drift = (x: number, y: number) => [1, 0.1 * x - 0.05 * y].slice(0, props.driftCount);
    scenario.sampleDrift = [];
    scenario.queryDrift = [];
    for (let row = 0; row < sampleCount; row++)
      scenario.sampleDrift.push(...drift(positions[row * 2], positions[row * 2 + 1]));
    for (let query = 0; query < queryCount; query++)
      scenario.queryDrift.push(...drift(supports[query * 2], supports[query * 2 + 1]));
  }
  return scenario;
}

/**
 * Review input: rows 1 to 8 at `(±3, ±4)` and `(±4, ±3)` have float32 `d^2 = 25`; row 0 at
 * `(3 + 2^-22, 4)` has `d^2` one float32 step above 25, which `sqrt` may round back to 5. The scan
 * ranks by `distance()` with ties to the lower row; the index ranks by `d^2`.
 */
function createNearTieScenario(): Scenario {
  const positions = [3.000000238418579, 4, 3, 4, -3, 4, 3, -4, -3, -4, 4, 3, -4, 3, 4, -3, -4, -3];
  for (let row = 0; row < 6; row++) positions.push(7 + row, 1 + row * 0.5);
  const sampleCount = positions.length / 2;
  const observations: number[] = [];
  for (let row = 0; row < sampleCount; row++) observations.push(row * 0.37 - 1, 0.01, 0, 1);
  return {
    name: 'near-tie',
    positions,
    observations,
    supports: [0, 0],
    supportFormat: 'float32x2',
    supportCount: 1,
    props: {mode: 'ordinary', neighborCount: 4}
  };
}

it('local kriging grid index keeps the scan neighbors on a float32 near-tie', async () => {
  const scenario = createNearTieScenario();
  expect(getFloat32SquaredDistance(scenario.positions[0], scenario.positions[1])).toBeGreaterThan(
    25
  );
  expect(getFloat32SquaredDistance(scenario.positions[2], scenario.positions[3])).toBe(25);
  const bruteForce = await runLocalKriging(scenario, 'brute-force');
  const indexed = await runLocalKriging(scenario, 'grid');
  expect(bruteForce.status).toEqual([0]);
  expect(indexed.status).toEqual(bruteForce.status);
  expect(indexed.values).toEqual(bruteForce.values);
  expect(indexed.weights).toEqual(bruteForce.weights);
});

it('local kriging grid index matches the scan when a distance tie outnumbers its candidate margin', async () => {
  // Forty samples on a circle of radius 5 around the query: float32 rounding spreads their d^2
  // over a few steps, so neighbors tie or near-tie far beyond the index's extra candidates.
  const positions: number[] = [];
  const observations: number[] = [];
  for (let row = 0; row < 40; row++) {
    const angle = (2 * Math.PI * ((row * 17) % 40)) / 40;
    positions.push(5 * Math.cos(angle), 5 * Math.sin(angle));
    observations.push(Math.sin(row), 0.01, 0, 1);
  }
  for (let row = 0; row < 10; row++) {
    positions.push(9 + row * 0.3, -2 + row * 0.7);
    observations.push(row * 0.1, 0.01, 0, 1);
  }
  for (const neighborCount of [4, 16]) {
    const scenario: Scenario = {
      name: 'circle',
      positions,
      observations,
      supports: [0, 0, 0.5, 0],
      supportFormat: 'float32x2',
      supportCount: 1,
      props: {mode: 'ordinary', neighborCount}
    };
    const bruteForce = await runLocalKriging(scenario, 'brute-force');
    const indexed = await runLocalKriging(scenario, 'grid');
    expect(bruteForce.status).toEqual([0, 0]);
    expect(indexed.values).toEqual(bruteForce.values);
    expect(indexed.weights).toEqual(bruteForce.weights);
  }
});

const RANDOM_SCENARIOS: Scenario[] = [
  createRandomScenario('ordinary, non-lattice', 400, 64, {mode: 'ordinary', neighborCount: 16}, 11),
  createRandomScenario(
    'ordinary, non-lattice, queries in chunks of seven',
    400,
    64,
    {mode: 'ordinary', neighborCount: 12, sampleIndexChunkQueryCount: 7},
    12
  ),
  createRandomScenario(
    'co-kriging, two variables',
    400,
    48,
    {
      mode: 'co-kriging',
      variableCount: 2,
      coregionalization: [1, 0, 0.6, 0.7],
      coKrigingMean: 'ordinary',
      neighborCount: 20,
      targetVariable: 1
    },
    13
  ),
  createRandomScenario(
    'external drift, two columns',
    400,
    48,
    {mode: 'external-drift', driftCount: 2, neighborCount: 10, searchRadius: 4},
    14
  )
];

for (const scenario of RANDOM_SCENARIOS) {
  it(`local kriging grid index matches the exhaustive scan: ${scenario.name}`, async () => {
    const bruteForce = await runLocalKriging(scenario, 'brute-force');
    const indexed = await runLocalKriging(scenario, 'grid');
    expect(indexed.status).toEqual(bruteForce.status);
    expect(indexed.values).toEqual(bruteForce.values);
    expect(indexed.weights).toEqual(bruteForce.weights);
    expect(bruteForce.status.filter(status => status === 0).length).toBeGreaterThan(20);
  });
}

it('local kriging auto resolves to the grid above the sample cut-over and matches the scan', async () => {
  const scenario = createRandomScenario(
    'auto',
    GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT + 37,
    24,
    {mode: 'ordinary', neighborCount: 16},
    15
  );
  const bruteForce = await runLocalKriging(scenario, 'brute-force');
  const automatic = await runLocalKriging(scenario, 'auto', 'grid');
  expect(automatic.status).toEqual(bruteForce.status);
  expect(automatic.values).toEqual(bruteForce.values);
  expect(automatic.weights).toEqual(bruteForce.weights);
});

it('local kriging fewer-than-neighborCount scenario retains every usable sample', async () => {
  const result = await runLocalKriging(SCENARIOS[2], 'grid');
  for (let query = 0; query < QUERY_POINTS.length; query++) {
    expect(result.status[query]).toBe(0);
    expect(result.values[query * 4 + 3]).toBe(5);
  }
});

it('local kriging sampleIndex resolves auto by sample count and rejects contradictions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const view = <Format extends GPUVectorFormat>(name: string, format: Format, length: number) =>
    importGraphBuffer(graph, name, createOutputBuffer(device, length * 4), format, length);
  const create = (sampleCount: number, overrides: Partial<GPULocalKrigingProps> = {}) =>
    new GPULocalKriging({
      positions: view(`positions-${sampleCount}`, 'float32x2', sampleCount),
      observations: view(`observations-${sampleCount}`, 'float32x4', sampleCount),
      supports: view(`supports-${sampleCount}`, 'float32x2', 2),
      mode: 'ordinary',
      covariance: {model: 'exponential', sill: 1, range: 1},
      output: {
        values: view(`values-${sampleCount}`, 'float32x4', 2),
        status: view(`status-${sampleCount}`, 'uint32', 2)
      },
      ...overrides
    });
  expect(create(GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT).sampleIndex).toBe(
    'brute-force'
  );
  expect(create(GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT + 1).sampleIndex).toBe('grid');
  // External drift, per-frame parameters and weights: the grid kernel reads sample positions from
  // its candidate binding, so it binds no more buffers than the scan and 'auto' still takes the
  // grid above the cut-over instead of an uncapped scan.
  const crowdedCount = GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT + 3;
  expect(
    create(crowdedCount, {
      mode: 'external-drift',
      covariance: undefined,
      parameters: view('crowded-parameters', 'float32', 64),
      driftCount: 1,
      sampleDrift: view('crowded-sample-drift', 'float32', crowdedCount),
      queryDrift: view('crowded-query-drift', 'float32', 2),
      output: {
        values: view('crowded-values', 'float32x4', 2),
        status: view('crowded-status', 'uint32', 2),
        weights: view('crowded-weights', 'float32', 32)
      }
    }).sampleIndex
  ).toBe('grid');
  expect(
    create(GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT + 2, {sampleIndex: 'brute-force'})
      .sampleIndex
  ).toBe('brute-force');
  // Few samples but more scan tests than the work cap: 'auto' takes the grid too.
  const manyQueries = GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK / 4096 + 1;
  const busy = (sampleIndex: 'auto' | 'brute-force') =>
    new GPULocalKriging({
      positions: createTransientView(graph, `busy-positions-${sampleIndex}`, 'float32x2', 4096),
      observations: createTransientView(
        graph,
        `busy-observations-${sampleIndex}`,
        'float32x4',
        4096
      ),
      supports: createTransientView(
        graph,
        `busy-supports-${sampleIndex}`,
        'float32x2',
        manyQueries
      ),
      mode: 'ordinary',
      neighborCount: 4,
      covariance: {model: 'exponential', sill: 1, range: 1},
      sampleIndex,
      output: {
        values: createTransientView(graph, `busy-values-${sampleIndex}`, 'float32x4', manyQueries),
        status: createTransientView(graph, `busy-status-${sampleIndex}`, 'uint32', manyQueries)
      }
    });
  expect(busy('auto').sampleIndex).toBe('grid');
  expect(() => busy('brute-force')).toThrow(
    /^local-kriging sampleIndex 'brute-force' work \(samples times queries\) must be at most 4294967296$/
  );
  expect(() =>
    create(16, {
      sampleIndex: 'grid',
      neighborhoods: view('neighborhoods', 'uint32', 4),
      neighborhoodCapacity: 2
    })
  ).toThrow(/^local-kriging sampleIndex 'grid' needs neighborhoods to be omitted/);
  expect(() => create(17, {sampleIndex: 'nearest' as never})).toThrow(
    /^local-kriging sampleIndex must be 'auto', 'brute-force' or 'grid'/
  );
  expect(() =>
    create(18, {spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}})
  ).toThrow(/^local-kriging does not support longitude-latitude coordinates/);
  expect(
    create(19, {spatialContext: {coordinateSpace: 'planar', metric: 'native'}}).spatialContext
  ).toMatchObject({coordinateSpace: 'planar'});
});

it('local kriging explicit grid rejects bindings beyond the device limits by name', () => {
  // A device with the WebGPU defaults: eight storage buffers per stage and 128 MiB per binding.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const create = (sampleCount: number, queryCount: number, crowded: boolean) =>
    new GPULocalKriging({
      id: `local-kriging-${serial}`,
      positions: view('float32x2', sampleCount),
      observations: view('float32x4', sampleCount),
      supports: view('float32x2', queryCount),
      mode: crowded ? 'external-drift' : 'ordinary',
      neighborCount: 4,
      sampleIndex: 'grid',
      ...(crowded
        ? {
            parameters: view('float32', 64),
            driftCount: 1,
            sampleDrift: view('float32', sampleCount),
            queryDrift: view('float32', queryCount)
          }
        : {covariance: {model: 'exponential' as const, sill: 1, range: 1}}),
      output: {
        values: view('float32x4', queryCount),
        status: view('uint32', queryCount),
        ...(crowded ? {weights: view('float32', queryCount * 4)} : {})
      }
    });
  // External drift, parameters and weights: nine kriging bindings on an eight-binding device.
  expect(() => create(64, 8, true).getCommandNodes(graph)).toThrow(
    /^GPULocalKriging: kernel local-kriging-\d+ binds 9 storage buffers, more than the device limit of 8$/
  );
  // Samples plus queries beyond 2^24 float32x2 search targets: above one 128 MiB binding.
  expect(() => create(2 ** 23, 2 ** 23 + 1, false).getCommandNodes(graph)).toThrow(
    /^local-kriging-\d+ sampleIndex 'grid' search targets \(samples plus queries, float32x2\) needs 134217736 bytes in one binding \(limit 134217728\)/
  );
  // Within the limits the explicit grid builds; its candidate rows are chunked to fit.
  expect(create(64, 8, false).getCommandNodes(graph).length).toBeGreaterThan(1);
});
