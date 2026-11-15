// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS,
  GPUOPTICS,
  type GPUOPTICSProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import parallelFixture from './optics-parallel-fixtures.json';

type Scenario = {
  name: string;
  positions: number[];
  repeat?: boolean;
  times?: number[];
  /** Grid-path radius-neighbor slots, when the default per-row allowance is too small. */
  neighborCapacity?: number;
  props: Omit<GPUOPTICSProps, 'positions' | 'times' | 'output' | 'neighborIndex'>;
};

/** Deterministic pseudo-random sequence, so the clusters are the same on every run. */
function createRandom(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Clustered points snapped to a 1/8 lattice (exact ties and duplicates) plus invalid rows. */
function createClusteredPositions(count: number, seed: number): number[] {
  const random = createRandom(seed);
  const centers = [
    [2, 2],
    [9, 3],
    [5, 9],
    [14, 12]
  ];
  const positions: number[] = [];
  for (let row = 0; row < count; row++) {
    if (row % 97 === 13) {
      positions.push(Number.NaN, 0);
      continue;
    }
    const [x, y] = row % 11 === 0 ? [random() * 16, random() * 16] : centers[row % centers.length];
    const spread = row % 11 === 0 ? 0 : 1.5;
    positions.push(
      Math.round((x + (random() - 0.5) * spread) * 8) / 8,
      Math.round((y + (random() - 0.5) * spread) * 8) / 8
    );
  }
  return positions;
}

const clustered = createClusteredPositions(600, 7);

const SCENARIOS: Scenario[] = [
  {
    name: 'planar clusters, duplicates, noise and invalid rows (DBSCAN extraction)',
    positions: clustered,
    props: {minimumSamples: 5, maximumDistance: 0.6, extractionDistance: 0.4, frontierBatchSize: 37}
  },
  {
    name: 'planar, a radius covering whole clusters',
    positions: clustered,
    neighborCapacity: 300 * 300,
    props: {minimumSamples: 3, maximumDistance: 3, extractionDistance: 1, frontierBatchSize: 64}
  },
  {
    name: 'planar, stopped at maximumIterations',
    positions: clustered,
    props: {
      minimumSamples: 4,
      maximumDistance: 0.8,
      extractionDistance: 0.5,
      maximumIterations: 211,
      frontierBatchSize: 50
    }
  },
  {
    name: 'planar xi extraction',
    positions: clustered.slice(0, 400),
    props: {minimumSamples: 6, maximumDistance: 1.5, extractionMethod: 'xi', xi: 0.05}
  },
  {
    name: 'space-time OPTICS',
    positions: clustered,
    times: Array.from({length: clustered.length / 2}, (_, row) => (row * 7) % 50),
    props: {
      minimumSamples: 4,
      maximumDistance: 1,
      extractionDistance: 0.7,
      maximumTimeDistance: 12,
      temporalContext: {
        encoding: 'float32-relative',
        unit: 'second',
        epoch: 0n,
        axis: {kind: 'irregular'}
      }
    }
  },
  {
    name: 'great-circle positions across the antimeridian',
    neighborCapacity: 300 * 300,
    positions: clustered.map((value, index) =>
      index % 2 === 0 ? ((value * 0.05 + 179.6 + 180) % 360) - 180 : value * 0.05 + 40
    ),
    props: {
      minimumSamples: 4,
      maximumDistance: 4000,
      extractionDistance: 2500,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
    }
  }
];

/** Every output word of one OPTICS run, read back to the CPU. */
type OPTICSRun = {
  ordering: number[];
  coreDistances: number[];
  reachability: number[];
  predecessors: number[];
  labels: number[];
  status: number[];
  clusterIntervals?: number[];
  clusterSummary?: number[];
  candidateOverflow?: number[];
  iterationCount: number[];
  converged: number[];
  iterationLimitReached: number[];
};

async function runOPTICS(
  scenario: Scenario,
  neighborIndex: 'brute-force' | 'grid',
  orderingMethod: 'parallel' | 'sequential' = 'sequential'
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const rowCount = scenario.positions.length / 2;
  const input = <Format extends 'float32' | 'float32x2'>(
    values: number[],
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, Float32Array.from(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, `optics-input-${buffers.length}`, buffer, format, length);
  };
  const output = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    format: Format,
    length = rowCount,
    components = 1
  ) => {
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      words: length * components,
      view: importGraphBuffer(graph, `optics-output-${buffers.length}`, buffer, format, length)
    };
  };
  const xi = scenario.props.extractionMethod === 'xi';
  const outputs = {
    ordering: output('uint32'),
    coreDistances: output('float32'),
    reachability: output('float32'),
    predecessors: output('uint32'),
    labels: output('uint32'),
    status: output('uint32', 3),
    ...(xi
      ? {clusterIntervals: output('uint32x2', rowCount, 2), clusterSummary: output('uint32', 2)}
      : {})
  };
  const iterationPorts = {
    iterationCount: output('uint32', 1),
    converged: output('uint32', 1),
    iterationLimitReached: output('uint32', 1)
  };
  const candidateOverflow = neighborIndex === 'grid' ? output('uint32', 1) : undefined;
  const optics = new GPUOPTICS({
    ...scenario.props,
    id: `optics-${neighborIndex}`,
    neighborIndex,
    orderingMethod,
    neighborCapacity: neighborIndex === 'grid' ? scenario.neighborCapacity : undefined,
    positions: input(scenario.positions, 'float32x2', rowCount),
    times: scenario.times ? input(scenario.times, 'float32', rowCount) : undefined,
    output: {
      ...Object.fromEntries(Object.entries(outputs).map(([name, field]) => [name, field.view])),
      ports: {
        status: {
          ...Object.fromEntries(
            Object.entries(iterationPorts).map(([name, field]) => [name, field.view])
          ),
          ...(candidateOverflow ? {candidateOverflow: candidateOverflow.view} : {})
        }
      }
    } as GPUOPTICSProps['output']
  });
  expect(optics.neighborIndex).toBe(neighborIndex);
  graph.add(optics);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    if (scenario.repeat) submitGraph(device, compiled, undefined);
    const result: Record<string, number[]> = {};
    for (const [name, field] of Object.entries(outputs)) {
      result[name] =
        field.view.format === 'float32'
          ? await readFloat32(field.buffer, field.words)
          : await readUint32(field.buffer, field.words);
    }
    for (const [name, field] of Object.entries(iterationPorts)) {
      result[name] = await readUint32(field.buffer, 1);
    }
    if (candidateOverflow) {
      result['candidateOverflow'] = await readUint32(candidateOverflow.buffer, 1);
    }
    return result as OPTICSRun;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const scenario of SCENARIOS) {
  it(`OPTICS grid reachability matches the exhaustive step: ${scenario.name}`, async () => {
    const bruteForce = await runOPTICS(scenario, 'brute-force');
    const grid = await runOPTICS(scenario, 'grid');
    const parallel = await runOPTICS(scenario, 'grid', 'parallel');
    expect(parallel).toEqual(grid);
    expect(grid.candidateOverflow).toEqual([0]);
    for (const [name, expected] of Object.entries(bruteForce)) {
      expect(grid[name as keyof OPTICSRun], name).toEqual(expected);
    }
    // The scenario exercises the frontier: several rows visited, some with finite reachability.
    expect(bruteForce.status[0]).toBeGreaterThan(100);
    expect(bruteForce.reachability.filter(Number.isFinite).length).toBeGreaterThan(50);
  });
}

it('OPTICS grid path reports candidate overflow and leaves the ordering incomplete', async () => {
  const scenario = SCENARIOS[1];
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rowCount = scenario.positions.length / 2;
  const buffers: Buffer[] = [];
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number,
    data?: Float32Array
  ) => {
    const words = length * (format === 'float32x2' ? 2 : 1);
    const buffer = data ? createInputBuffer(device, data) : createOutputBuffer(device, words);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `overflow-${buffers.length}`, buffer, format, length)
    };
  };
  const status = view('uint32', 3);
  const overflow = view('uint32', 1);
  graph.add(
    new GPUOPTICS({
      ...scenario.props,
      neighborIndex: 'grid',
      neighborCapacity: rowCount,
      positions: view('float32x2', rowCount, Float32Array.from(scenario.positions)).view,
      output: {
        ordering: view('uint32', rowCount).view,
        coreDistances: view('float32', rowCount).view,
        reachability: view('float32', rowCount).view,
        predecessors: view('uint32', rowCount).view,
        labels: view('uint32', rowCount).view,
        status: status.view,
        ports: {status: {candidateOverflow: overflow.view}}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
    expect((await readUint32(status.buffer, 3)).slice(0, 2)).toEqual([0, 0]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('OPTICS neighborIndex resolves auto by row count and names its limits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  let viewCount = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) =>
    importGraphBuffer(
      graph,
      `${name}-${viewCount++}`,
      createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1)),
      format,
      length
    );
  const create = (rows: number, overrides: Partial<GPUOPTICSProps> = {}) =>
    new GPUOPTICS({
      positions: view(`positions-${rows}-${Object.keys(overrides).join()}`, 'float32x2', rows),
      minimumSamples: 4,
      maximumDistance: 1,
      extractionDistance: 0.5,
      output: {
        ordering: view(`ordering-${rows}-${Object.keys(overrides).join()}`, 'uint32', rows),
        coreDistances: view(`core-${rows}-${Object.keys(overrides).join()}`, 'float32', rows),
        reachability: view(`reach-${rows}-${Object.keys(overrides).join()}`, 'float32', rows),
        predecessors: view(`pred-${rows}-${Object.keys(overrides).join()}`, 'uint32', rows),
        labels: view(`labels-${rows}-${Object.keys(overrides).join()}`, 'uint32', rows),
        status: view(`status-${rows}-${Object.keys(overrides).join()}`, 'uint32', 3)
      },
      ...overrides
    });
  expect(create(GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS).neighborIndex).toBe('brute-force');
  expect(create(GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS + 1).neighborIndex).toBe('grid');
  for (const neighborIndex of ['brute-force', 'grid'] as const) {
    const optics = create(4096, {
      id: `count-${neighborIndex}`,
      neighborIndex,
      frontierBatchSize: 3
    });
    expect(optics.getCommandNodes(graph).length).toBeLessThanOrEqual(
      optics.estimatedCommandNodeCount
    );
  }
  expect(() => create(65536, {frontierBatchSize: 1})).toThrow(
    /^optics estimated command node count must be at most 32768/
  );
  expect(() =>
    create(GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS + 2, {neighborIndex: 'brute-force'})
  ).toThrow(/^optics positions rows must be an integer in \[1, 4096\]/);
  expect(() => create(65537)).toThrow(/^optics positions rows must be an integer in \[1, 65536\]/);
  expect(() =>
    create(GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS + 3, {
      extractionMethod: 'xi',
      extractionDistance: undefined
    })
  ).not.toThrow();
  expect(() => create(64, {neighborCapacity: 100})).toThrow(
    /^optics neighborCapacity and output.ports.status.candidateOverflow need neighborIndex 'grid'/
  );
  expect(() => create(65, {neighborIndex: 'kd-tree' as never})).toThrow(
    /^optics neighborIndex must be 'auto', 'brute-force' or 'grid'/
  );
});

it('OPTICS above the exhaustive cap orders every valid row through the grid', async () => {
  const positions = createClusteredPositions(20000, 11).map(value => value * 4);
  const scenario: Scenario = {
    name: 'large',
    positions,
    // Many duplicate lattice sites: about 300 rows lie within the radius of a clustered row.
    neighborCapacity: 20000 * 512,
    props: {
      minimumSamples: 5,
      maximumDistance: 0.5,
      extractionDistance: 0.3,
      frontierBatchSize: 256
    }
  };
  const result = await runOPTICS(scenario, 'grid');
  const rowCount = positions.length / 2;
  const invalidCount = Array.from({length: rowCount}, (_, row) => positions[2 * row]).filter(
    value => !Number.isFinite(value)
  ).length;
  expect(result.candidateOverflow).toEqual([0]);
  expect(result.status).toEqual([rowCount - invalidCount, 1, invalidCount]);
  const ordered = result.ordering.slice(0, rowCount - invalidCount);
  expect(new Set(ordered).size).toBe(ordered.length);
  const rank = new Map(ordered.map((row, position) => [row, position]));
  for (const row of ordered) {
    const predecessor = result.predecessors[row];
    if (Number.isFinite(result.reachability[row])) {
      // Reachability comes from an earlier, core row and is at least its core distance.
      expect(rank.get(predecessor)!).toBeLessThan(rank.get(row)!);
      expect(result.reachability[row]).toBeGreaterThanOrEqual(result.coreDistances[predecessor]);
    } else {
      expect(predecessor).toBe(0xffffffff);
    }
  }
});

for (const predecessorCorrection of [true, false]) {
  it(`OPTICS xi above 4096 matches sklearn and the sequential frontier (correction=${predecessorCorrection})`, async () => {
    const scenario: Scenario = {
      name: 'sklearn-large',
      positions: parallelFixture.positions,
      props: {...parallelFixture.options, extractionMethod: 'xi', predecessorCorrection}
    };
    const sequential = await runOPTICS(scenario, 'grid');
    const parallel = await runOPTICS(scenario, 'grid', 'parallel');
    expect(parallel).toEqual(sequential);
    const extraction = parallelFixture.xi.find(
      value => value.predecessorCorrection === predecessorCorrection
    )!;
    expect(parallel.ordering).toEqual(parallelFixture.ordering);
    expect(parallel.predecessors).toEqual(parallelFixture.predecessors);
    expect(parallel.coreDistances).toEqual(parallelFixture.coreDistances);
    expect(parallel.reachability).toEqual(
      parallelFixture.reachability.map(value => (value === 'Infinity' ? Infinity : value))
    );
    expect(parallel.labels).toEqual(extraction.labels);
    expect(parallel.clusterSummary).toEqual([extraction.intervals.length, 0]);
    expect(parallel.clusterIntervals!.slice(0, extraction.intervals.length * 2)).toEqual(
      extraction.intervals.flat()
    );
    expect(
      parallel
        .clusterIntervals!.slice(extraction.intervals.length * 2)
        .every(value => value === 0xffffffff)
    ).toBe(true);
    expect(parallel.iterationCount).toEqual([parallelFixture.ordering.length]);
    expect(parallel.converged).toEqual([1]);
    expect(parallel.iterationLimitReached).toEqual([0]);
  }, 600_000);
}

for (const extractionMethod of ['xi', 'dbscan'] as const) {
  for (const scene of [
    'all-invalid',
    'ties-and-padding',
    'visit-budget',
    'candidate-overflow'
  ] as const) {
    it(`parallel OPTICS handles ${scene}, ${extractionMethod}`, async () => {
      const positions = Array.from({length: 259}, (_, row) => [
        scene === 'all-invalid' || row % 17 === 0 ? NaN : Math.floor(row / 7),
        0
      ]).flat();
      const scenario: Scenario = {
        name: scene,
        positions,
        neighborCapacity: scene === 'candidate-overflow' ? 1 : 259 * 32,
        props: {
          minimumSamples: 2,
          maximumDistance: 1,
          extractionDistance: 1,
          extractionMethod,
          ...(scene === 'visit-budget' ? {maximumIterations: 19} : {})
        }
      };
      const reference = await runOPTICS(scenario, 'grid');
      const result = await runOPTICS(scenario, 'grid', 'parallel');
      expect(result).toEqual(reference);
      const overflow = scene === 'candidate-overflow';
      const incomplete = overflow || scene === 'visit-budget';
      expect(result.converged).toEqual([incomplete ? 0 : 1]);
      expect(result.iterationLimitReached).toEqual([scene === 'visit-budget' ? 1 : 0]);
      expect(result.iterationCount).toEqual([result.status[0]]);
      if (incomplete || scene === 'all-invalid') {
        expect(result.labels.every(value => value === 0xffffffff)).toBe(true);
        if (extractionMethod === 'xi') {
          expect(result.clusterSummary).toEqual([0, 0]);
          expect(result.clusterIntervals!.every(value => value === 0xffffffff)).toBe(true);
        }
      }
    });
  }
}

it('parallel OPTICS retains every distance bit before applying row-index ties', async () => {
  const scenario: Scenario = {
    name: 'near-tie',
    repeat: true,
    positions: [0, 0, 1 + 2 ** -20, 0, 1, 0],
    props: {minimumSamples: 2, maximumDistance: 2, extractionDistance: 2}
  };
  const result = await runOPTICS(scenario, 'grid', 'parallel');
  expect(result).toEqual(await runOPTICS(scenario, 'grid'));
  expect(result.ordering).toEqual([0, 2, 1]);
  expect(result.predecessors).toEqual([0xffffffff, 2, 0]);
});
