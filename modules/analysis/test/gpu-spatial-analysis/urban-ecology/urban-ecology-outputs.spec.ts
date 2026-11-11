// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addPopulationSupportAlignmentRecipe,
  addResistanceConnectivityRecipe,
  addSegregationCounterfactualRecipe,
  addSegregationDecompositionRecipe,
  addSparseResistanceConnectivityRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {
  addBuildingAdjacencyRecipe,
  addBuildingTessellationRecipe,
  addIsovistsRecipe,
  addStreetProfilesRecipe,
  addVisibilityGraphRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology.json';
import advancedReference from '../../oracles/fixtures/urban-ecology-advanced.json';

type AnyView = GraphDataView;
type Mirrored = {view: AnyView; buffer: ReturnType<RecipeTestFixture['output']>['buffer']};

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

function getComponentCount(format: string): number {
  return format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
}

/** Every graph view in a nested output description, by path. */
function collectViews(value: unknown, path: string, into: Map<string, AnyView>) {
  if (value instanceof GraphDataView) into.set(path, value);
  else if (Array.isArray(value)) {
    value.forEach((entry, index) => collectViews(entry, `${path}[${index}]`, into));
  } else if (value && typeof value === 'object') {
    for (const [key, entry] of Object.entries(value)) collectViews(entry, `${path}.${key}`, into);
  }
}

/** The same nested shape with a caller-owned fixture output in place of each view. */
function mirrorViews(
  fixture: RecipeTestFixture,
  value: unknown,
  path: string,
  into: Map<string, Mirrored>
): unknown {
  if (value instanceof GraphDataView) {
    const output = fixture.output(`caller${path}`, value.format as 'float32', value.length);
    into.set(path, output as Mirrored);
    return output.view;
  }
  if (Array.isArray(value)) {
    return value.map((entry, index) => mirrorViews(fixture, entry, `${path}[${index}]`, into));
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [
        key,
        mirrorViews(fixture, entry, `${path}.${key}`, into)
      ])
    );
  }
  return value;
}

function publishView(fixture: RecipeTestFixture, name: string, source: AnyView) {
  const output = fixture.output(name, source.format as 'float32', source.length);
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'urban-ecology-readback',
      invocationCount: source.length * getComponentCount(source.format),
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}

type OutputCase = {
  name: string;
  /** Builds the recipe and returns its results shaped like its `output` prop. */
  build: (fixture: RecipeTestFixture, output: never) => unknown;
  /** Paths whose content is compared; unwritten scratch in transient views is not. */
  compare?: (path: string) => boolean;
  /** Paths that must hold a nonzero word after the run. */
  expectWritten: string[];
};

/**
 * Runs the recipe twice: with graph-owned transient views, then with a caller-owned buffer for
 * every result. The recipe must use the given views (identity), write the same words there, and
 * leave none of them untouched.
 */
async function expectCallerOwnedOutputs(testCase: OutputCase) {
  const compare = testCase.compare ?? (() => true);
  const expected = new Map<string, number[]>();
  let template: unknown;
  const transient = await createFixture(`${testCase.name}-transient`);
  try {
    template = testCase.build(transient, undefined as never);
    const structure = new Map<string, AnyView>();
    collectViews(template, '', structure);
    expect(structure.size).toBeGreaterThan(0);
    const published = new Map<string, ReturnType<typeof publishView>>();
    for (const [path, view] of structure) {
      if (compare(path)) published.set(path, publishView(transient, `copy${path}`, view));
    }
    transient.run();
    for (const [path, output] of published) {
      const view = structure.get(path)!;
      expected.set(
        path,
        await transient.readUint32(output, view.length * getComponentCount(view.format))
      );
    }
  } finally {
    transient.destroy();
  }

  const caller = await createFixture(`${testCase.name}-caller`);
  try {
    const mirrored = new Map<string, Mirrored>();
    // The transient results only describe the shape: formats and lengths of every result.
    const outputProp = mirrorViews(caller, template, '', mirrored);
    const results = testCase.build(caller, outputProp as never);
    const used = new Map<string, AnyView>();
    collectViews(results, '', used);
    for (const [path, mirror] of mirrored) {
      expect(used.get(path), `${testCase.name} ${path} is the caller-owned view`).toBe(mirror.view);
    }
    caller.run();
    const read = async (path: string) => {
      const mirror = mirrored.get(path)!;
      return caller.readUint32(mirror, mirror.view.length * getComponentCount(mirror.view.format));
    };
    for (const [path, words] of expected) {
      expect(await read(path), `${testCase.name} ${path}`).toEqual(words);
    }
    for (const path of testCase.expectWritten) {
      expect(
        (await read(path)).some(word => word !== 0),
        `${testCase.name} ${path} was written`
      ).toBe(true);
    }
  } finally {
    caller.destroy();
  }
}

const BOX_SEGMENTS = new Float32Array([-2, -2, 2, -2, 2, -2, 2, 2, 2, 2, -2, 2, -2, 2, -2, -2]);
function getSharedWalls(fixture: RecipeTestFixture) {
  return {
    runs: {
      positions: fixture.input('positions', new Float32Array([0, 0, 0, 2]), 'float32x2', 2),
      offsets: fixture.input('offsets', new Uint32Array([0, 2]), 'uint32', 2),
      leftSourceIds: fixture.input('left', new Uint32Array([0]), 'uint32', 1),
      rightSourceIds: fixture.input('right', new Uint32Array([1]), 'uint32', 1),
      forward: fixture.input('forward', new Uint32Array([0]), 'uint32', 1)
    },
    status: {count: fixture.input('count', new Uint32Array([1]), 'uint32', 1)}
  };
}

it('isovists write every result into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'isovists',
    build: (fixture, output) =>
      addIsovistsRecipe(fixture.graph, {
        positions: fixture.input('positions', new Float32Array([0, 0, 2, 0]), 'float32x2', 2),
        segments: fixture.input('segments', BOX_SEGMENTS, 'float32x4', 4),
        rayCount: 8,
        maximumDistance: 10,
        output
      }).outputs,
    expectWritten: ['.endpoints', '.distances', '.measures', '.angles', '.validity']
  });
});

it('street profiles write every result, including exact facades, into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'street-profiles',
    build: (fixture, output) => {
      const result = addStreetProfilesRecipe(fixture.graph, {
        positions: fixture.input('positions', new Float32Array([0, 0]), 'float32x2', 1),
        normals: fixture.input('normals', new Float32Array([0, 1]), 'float32x2', 1),
        segments: fixture.input(
          'segments',
          new Float32Array([-10, 2, 10, 2, -10, 3, 10, 3, -10, -3, 10, -3]),
          'float32x4',
          3
        ),
        segmentBuildingIds: fixture.input('owners', new Uint32Array([0, 1, 2]), 'uint32', 3),
        buildingHeights: fixture.input('heights', new Float32Array([6, 7, 9]), 'float32', 3),
        maximumDistance: 10,
        sampleSpacing: 1,
        streetSegment: fixture.input('street', new Float32Array([-4, 0, 4, 0]), 'float32x4', 1),
        output
      });
      return {
        ...result.outputs,
        facadeLengths: result.facades!.lengths,
        facadeFrontage: result.facades!.frontage
      };
    },
    expectWritten: ['.profiles', '.canyon', '.frontage', '.facadeLengths', '.facadeFrontage']
  });
});

it('building adjacency writes every result into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'building-adjacency',
    build: (fixture, output) => {
      const result = addBuildingAdjacencyRecipe(fixture.graph, {
        sharedPaths: getSharedWalls(fixture),
        lineBuildingIds: fixture.input('owners', new Uint32Array([0, 1, 2]), 'uint32', 3),
        orientations: fixture.input(
          'orientations',
          new Float32Array([Math.PI * 0.49, -Math.PI * 0.49, 0]),
          'float32',
          3
        ),
        output
      });
      return {...result.outputs, complete: result.status.complete};
    },
    expectWritten: ['.pairs', '.lengths', '.morphology', '.complete']
  });
});

it('visibility graph writes every result into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'visibility-graph',
    build: (fixture, output) => {
      const result = addVisibilityGraphRecipe(fixture.graph, {
        positions: fixture.input('positions', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
        segments: fixture.input('segments', new Float32Array([0, 10, 2, 10]), 'float32x4', 1),
        maximumDistance: 1.1,
        edgeCapacity: 4,
        output
      });
      return {...result.outputs, status: result.status};
    },
    expectWritten: ['.adjacency', '.edges', '.metrics', '.choice', '.integration', '.status']
  });
});

it('building tessellation writes its samples, cells and status into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'building-tessellation',
    build: (fixture, output) => {
      const result = addBuildingTessellationRecipe(fixture.graph, {
        segments: fixture.input(
          'segments',
          Float32Array.from(reference.tessellation.segments),
          'float32x4',
          8
        ),
        segmentBuildingIds: fixture.input(
          'owners',
          Uint32Array.from(reference.tessellation.owners),
          'uint32',
          8
        ),
        buildingCount: 2,
        enclosure: fixture.input(
          'enclosure',
          Float32Array.from(reference.tessellation.enclosure),
          'float32x2',
          4
        ),
        sampleSpacing: 1,
        sampleCapacity: 8,
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 256, nodedSegments: 256, vertices: 128, rings: 32},
        output
      });
      return {
        overlays: result.outputs,
        ...result.intermediates,
        samplingStatus: result.status.sampling,
        complete: result.status.complete
      };
    },
    // The overlay scratch beyond its counts is not defined; its counts and the sampling are.
    compare: path =>
      !path.startsWith('.overlays') || path.endsWith('.status.count') || path.endsWith('.overflow'),
    expectWritten: [
      '.samples',
      '.owners',
      '.fragments',
      '.fragmentCounts',
      '.samplingStatus',
      '.complete',
      '.overlays[0].status.count',
      '.overlays[1].status.count',
      '.overlays[0].geometry.positions'
    ]
  });
});

it('landscape metrics write every result, including reconciled labels, into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'landscape-metrics',
    build: (fixture, output) => {
      const result = addLandscapeMetricsRecipe(fixture.graph, {
        // Patches 1 and 2 are the same class and touch, so they merge into root 1.
        patches: fixture.input('patches', new Uint32Array([0, 0, 1, 2, 0, 0, 1, 2]), 'uint32', 8),
        patchClasses: fixture.input('classes', new Uint32Array([1, 0, 0]), 'uint32', 3),
        width: 4,
        height: 2,
        classCount: 2,
        cellSize: [1, 1],
        connectivity: 4,
        reconcilePatches: true,
        output
      });
      return {
        ...result.outputs,
        reconciledPatches: result.provenance.reconciledPatches,
        patchParents: result.provenance.patchParents
      };
    },
    expectWritten: [
      '.patchMetrics',
      '.classMetrics',
      '.classAdjacency',
      '.landscape',
      '.landscapeCohesion',
      '.reconciledPatches',
      '.patchParents'
    ]
  });
});

it('patch proximity writes every result into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'patch-proximity',
    build: (fixture, output) => {
      const result = addPatchProximityRecipe(fixture.graph, {
        patches: fixture.input('patches', new Uint32Array([0, 0, 0xffffffff, 1, 1]), 'uint32', 5),
        patchClasses: fixture.input('classes', new Uint32Array([0, 0]), 'uint32', 2),
        width: 5,
        height: 1,
        cellSize: [1, 1],
        edgeDepth: 0.4,
        searchRadius: 2,
        output
      });
      return {...result.outputs, ...result.intermediates};
    },
    expectWritten: ['.core', '.distances', '.metrics', '.horizontal', '.fields']
  });
});

it('dense resistance connectivity writes every result into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'resistance-connectivity',
    build: (fixture, output) => {
      const result = addResistanceConnectivityRecipe(fixture.graph, {
        edges: fixture.input('edges', Uint32Array.from(reference.resistance.edges), 'uint32x2', 3),
        conductances: fixture.input(
          'conductances',
          Float32Array.from(reference.resistance.conductances),
          'float32',
          3
        ),
        vertexCount: 3,
        source: 0,
        sink: 2,
        output
      });
      return {...result.outputs, status: result.status};
    },
    expectWritten: ['.potentials', '.currents', '.diagnostics', '.status']
  });
});

it('sparse resistance connectivity writes every result into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'sparse-resistance-connectivity',
    build: (fixture, output) => {
      const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
        edges: fixture.input('edges', Uint32Array.from(reference.resistance.edges), 'uint32x2', 3),
        conductances: fixture.input(
          'conductances',
          Float32Array.from(reference.resistance.conductances),
          'float32',
          3
        ),
        vertexCount: 3,
        source: 0,
        sink: 2,
        output
      });
      return {...result.outputs, status: result.status};
    },
    expectWritten: ['.potentials', '.currents', '.vertexCurrents', '.diagnostics', '.status']
  });
});

it('segregation decomposition writes its indices and randomization into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'segregation-decomposition',
    build: (fixture, output) => {
      const baseline = fixture.input(
        'baseline',
        Float32Array.from(advancedReference.counterfactual.baseline),
        'float32x2',
        4
      );
      const final = fixture.input(
        'final',
        Float32Array.from(advancedReference.counterfactual.final),
        'float32x2',
        4
      );
      const result = addSegregationDecompositionRecipe(fixture.graph, {
        populations: [baseline, baseline, final, final],
        supportId: 'same-four-units',
        counterfactualMethod: 'supplied',
        randomization: {seed: 7, replicates: 16},
        output
      });
      return {
        ...result.outputs,
        distribution: result.randomization!.distribution,
        inference: result.randomization!.inference
      };
    },
    expectWritten: ['.indices', '.validity', '.decomposition', '.distribution', '.inference']
  });
});

it('segregation counterfactual writes its populations and decomposition into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'segregation-counterfactual',
    build: (fixture, output) => {
      const result = addSegregationCounterfactualRecipe(fixture.graph, {
        baseline: fixture.input(
          'baseline',
          Float32Array.from(advancedReference.counterfactual.baseline),
          'float32x2',
          4
        ),
        final: fixture.input(
          'final',
          Float32Array.from(advancedReference.counterfactual.final),
          'float32x2',
          4
        ),
        supportId: 'same-four-units',
        method: 'composition',
        output
      });
      return {...result.outputs, ...result.counterfactuals};
    },
    expectWritten: ['.compositionOnly', '.spatialOnly', '.indices', '.validity', '.decomposition']
  });
});

it('population support alignment writes its populations and certificate into caller-owned views', async () => {
  await expectCallerOwnedOutputs({
    name: 'population-support-alignment',
    build: (fixture, output) => {
      const result = addPopulationSupportAlignmentRecipe(fixture.graph, {
        populations: fixture.input('populations', new Float32Array([2, 8, 6, 4]), 'float32x2', 2),
        sourceIds: fixture.input('source', new Uint32Array([20, 10]), 'uint32', 2),
        targetIds: fixture.input('target', new Uint32Array([10, 20, 30]), 'uint32', 3),
        output
      });
      return {...result.outputs, status: result.status};
    },
    expectWritten: ['.populations', '.validity', '.status']
  });
});

function createLCG(seed: number) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

async function readAll(fixture: RecipeTestFixture, views: Record<string, AnyView>) {
  const published = Object.entries(views).map(
    ([name, view]) => [name, view, publishView(fixture, `read-${name}`, view)] as const
  );
  fixture.run();
  const words: Record<string, number[]> = {};
  for (const [name, view, output] of published) {
    words[name] = await fixture.readUint32(output, view.length * getComponentCount(view.format));
  }
  return words;
}

it('isovists find the same hits with the segment BVH as with every segment', async () => {
  const next = createLCG(11);
  // Connected random-walk polylines share endpoints, so vertex rays tie between segments.
  const segments: number[] = [];
  for (let polyline = 0; polyline < 40; polyline++) {
    let x = 100 * next();
    let y = 100 * next();
    for (let step = 0; step < 6; step++) {
      const nextX = x + 12 * (next() - 0.5);
      const nextY = y + 12 * (next() - 0.5);
      segments.push(x, y, nextX, nextY);
      x = nextX;
      y = nextY;
    }
  }
  const positions: number[] = [];
  for (let viewpoint = 0; viewpoint < 12; viewpoint++) positions.push(100 * next(), 100 * next());
  const results: Record<string, Record<string, number[]>> = {};
  for (const obstacleIndex of ['brute-force', 'bvh'] as const) {
    const fixture = await createFixture(`isovist-${obstacleIndex}`);
    try {
      const result = addIsovistsRecipe(fixture.graph, {
        positions: fixture.input('positions', Float32Array.from(positions), 'float32x2', 12),
        segments: fixture.input('segments', Float32Array.from(segments), 'float32x4', 240),
        rayCount: 32,
        sampling: 'vertices',
        angularEpsilon: 1e-4,
        maximumDistance: 45,
        obstacleIndex
      });
      expect(result.provenance.obstacleIndex).toBe(obstacleIndex);
      results[obstacleIndex] = await readAll(fixture, result.outputs);
    } finally {
      fixture.destroy();
    }
  }
  const bruteForce = results['brute-force'];
  expect(bruteForce['segments'].some(hit => hit !== 0xffffffff)).toBe(true);
  expect(bruteForce['validity'].some(valid => valid === 1)).toBe(true);
  for (const name of Object.keys(bruteForce)) {
    expect(results['bvh'][name], name).toEqual(bruteForce[name]);
  }
});

it('isovists agree between the segment BVH and every segment for segments one to two tolerances off a ray', async () => {
  const tolerance = 0.1;
  const diagonal = Math.SQRT1_2;
  // Near-parallel segments whose start is inside the tolerance and whose far end is up to twice
  // the tolerance from the ray: the brute-force collinear branch accepts them. Rays leave the
  // origin along the axes and the diagonals.
  const segments = [
    ...[3, 0.09, 3.5, 0.18],
    ...[6, 0.09, 4, 0.17],
    ...[3, -0.09, 3.5, -0.19],
    ...[0.09, 3, 0.18, 3.5],
    ...[
      3 * diagonal - 0.09 * diagonal,
      3 * diagonal + 0.09 * diagonal,
      3.5 * diagonal - 0.18 * diagonal,
      3.5 * diagonal + 0.18 * diagonal
    ],
    ...[-5, 0.5, -5, -0.5]
  ];
  const results: Record<string, Record<string, number[]>> = {};
  for (const obstacleIndex of ['brute-force', 'bvh'] as const) {
    const fixture = await createFixture(`isovist-tolerance-${obstacleIndex}`);
    try {
      const result = addIsovistsRecipe(fixture.graph, {
        positions: fixture.input('positions', Float32Array.from([0, 0]), 'float32x2', 1),
        segments: fixture.input(
          'segments',
          Float32Array.from(segments),
          'float32x4',
          segments.length / 4
        ),
        rayCount: 8,
        sampling: 'uniform',
        maximumDistance: 20,
        tolerance,
        obstacleIndex
      });
      results[obstacleIndex] = await readAll(fixture, result.outputs);
    } finally {
      fixture.destroy();
    }
  }
  // Some ray is blocked by one of the tolerance-band segments.
  expect(results['brute-force']['segments'].some(hit => hit !== 0xffffffff && hit < 5)).toBe(true);
  for (const name of Object.keys(results['brute-force'])) {
    expect(results['bvh'][name], name).toEqual(results['brute-force'][name]);
  }
});

it('street profiles and exact facades match between the segment BVH and every segment', async () => {
  const next = createLCG(23);
  const segments: number[] = [];
  const owners: number[] = [];
  const heights: number[] = [];
  for (let building = 0; building < 36; building++) {
    const x = -48 + 96 * next();
    const y = (next() < 0.5 ? -1 : 1) * (3 + 22 * next());
    const width = 2 + 5 * next();
    const depth = 2 + 5 * next();
    segments.push(x, y, x + width, y, x + width, y, x + width, y + depth);
    segments.push(x + width, y + depth, x, y + depth, x, y + depth, x, y);
    owners.push(building, building, building, building);
    heights.push(3 + 20 * next());
  }
  const sampleCount = 50;
  const positions: number[] = [];
  const normals: number[] = [];
  for (let sample = 0; sample < sampleCount; sample++) {
    positions.push(-49 + (98 * sample) / (sampleCount - 1), 0);
    normals.push(0, 1);
  }
  const results: Record<string, Record<string, number[]>> = {};
  for (const obstacleIndex of ['brute-force', 'bvh'] as const) {
    const fixture = await createFixture(`street-${obstacleIndex}`);
    try {
      const result = addStreetProfilesRecipe(fixture.graph, {
        positions: fixture.input(
          'positions',
          Float32Array.from(positions),
          'float32x2',
          sampleCount
        ),
        normals: fixture.input('normals', Float32Array.from(normals), 'float32x2', sampleCount),
        segments: fixture.input(
          'segments',
          Float32Array.from(segments),
          'float32x4',
          segments.length / 4
        ),
        segmentBuildingIds: fixture.input(
          'owners',
          Uint32Array.from(owners),
          'uint32',
          owners.length
        ),
        buildingHeights: fixture.input('heights', Float32Array.from(heights), 'float32', 36),
        maximumDistance: 30,
        sampleSpacing: 98 / (sampleCount - 1),
        streetSegment: fixture.input('street', new Float32Array([-49, 0, 49, 0]), 'float32x4', 1),
        obstacleIndex
      });
      expect(result.provenance.obstacleIndex).toBe(obstacleIndex);
      results[obstacleIndex] = await readAll(fixture, {
        ...result.outputs,
        facadeLengths: result.facades!.lengths,
        facadeFrontage: result.facades!.frontage
      });
    } finally {
      fixture.destroy();
    }
  }
  const bruteForce = results['brute-force'];
  expect(bruteForce['validity'].some(flags => (flags & 3) !== 0)).toBe(true);
  expect(bruteForce['facadeLengths'].some(word => word !== 0)).toBe(true);
  for (const name of Object.keys(bruteForce)) {
    expect(results['bvh'][name], name).toEqual(bruteForce[name]);
  }
});
