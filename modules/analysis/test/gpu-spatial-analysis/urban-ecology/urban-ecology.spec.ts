// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addResistanceConnectivityRecipe,
  addSegregationDecompositionRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {
  addStreetProfilesRecipe,
  addIsovistsRecipe,
  addVisibilityGraphRecipe,
  addBuildingAdjacencyRecipe,
  addBuildingTessellationRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectUrbanFloat32Output as expectClose} from './urban-ecology-assertions';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology.json';
import directReference from '../../oracles/fixtures/urban-ecology-advanced.json';

type Format = 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x2' | 'uint32x4';
async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}
function publish(fixture: RecipeTestFixture, name: string, source: GraphDataView<Format>) {
  const output = fixture.output(name, source.format, source.length);
  const components = source.format.endsWith('x4') ? 4 : source.format.endsWith('x2') ? 2 : 1;
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'urban-ecology-readback',
      invocationCount: source.length * components,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}
const BOX_SEGMENTS = new Float32Array([-2, -2, 2, -2, 2, -2, 2, 2, 2, 2, -2, 2, -2, 2, -2, -2]);

it('landscape class aggregation, contagion, mesh and cohesion match analytical definitions', async () => {
  const fixture = await createFixture('landscape');
  try {
    const result = addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array([0, 0, 1, 1, 0, 0, 1, 1]), 'uint32', 8),
      patchClasses: fixture.input('classes', new Uint32Array([0, 1]), 'uint32', 2),
      width: 4,
      height: 2,
      classCount: 2,
      cellSize: [1, 1],
      connectivity: 4
    });
    const classes = publish(fixture, 'classes-output', result.outputs.classMetrics);
    const adjacency = publish(fixture, 'adjacency-output', result.outputs.classAdjacency);
    const aggregation = publish(fixture, 'aggregation-output', result.outputs.aggregation);
    const landscape = publish(fixture, 'landscape-output', result.outputs.landscape);
    fixture.run();
    await expectClose(fixture, classes, reference.landscape.classMetrics);
    expect(await fixture.readUint32(adjacency, 4)).toEqual(reference.landscape.classAdjacency);
    await expectClose(fixture, aggregation, reference.landscape.aggregation);
    await expectClose(fixture, landscape, reference.landscape.landscape);
  } finally {
    fixture.destroy();
  }
});

it('patch proximity excludes self and preserves physical edge depth and background barriers', async () => {
  const fixture = await createFixture('patch-proximity');
  try {
    const result = addPatchProximityRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array([0, 0, 0xffffffff, 1, 1]), 'uint32', 5),
      patchClasses: fixture.input('classes', new Uint32Array([0, 0]), 'uint32', 2),
      width: 5,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0.4,
      searchRadius: 2
    });
    const metrics = publish(fixture, 'metrics', result.outputs.metrics);
    const core = publish(fixture, 'core', result.outputs.core);
    fixture.run();
    await expectClose(fixture, metrics, [2, 2, 1, 2, 2, 2, 1, 2]);
    expect(await fixture.readUint32(core, 5)).toEqual([1, 1, 0, 1, 1]);
  } finally {
    fixture.destroy();
  }
});

it('grounded resistance and edge currents agree with independent SciPy Laplacian solve', async () => {
  const fixture = await createFixture('resistance');
  try {
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
      sink: 2
    });
    const potentials = publish(fixture, 'potentials', result.outputs.potentials);
    const currents = publish(fixture, 'currents', result.outputs.currents);
    const status = publish(fixture, 'status', result.status);
    const diagnostics = publish(fixture, 'diagnostics', result.outputs.diagnostics);
    fixture.run();
    await expectClose(fixture, potentials, reference.resistance.potentials);
    await expectClose(fixture, currents, reference.resistance.currents);
    expect((await fixture.readUint32(status, 4))[0]).toBe(1);
    expect((await fixture.readFloat32(diagnostics, 4))[0]).toBeCloseTo(
      reference.resistance.resistance,
      5
    );
  } finally {
    fixture.destroy();
  }
});

it.each([
  {conductance: 0, iterations: 8, invalid: 0},
  {conductance: -1, iterations: 8, invalid: 1}
])('resistance rejects disconnected and invalid conductance $conductance', async ({
  conductance,
  iterations,
  invalid
}) => {
  const fixture = await createFixture('resistance-invalid');
  try {
    const result = addResistanceConnectivityRecipe(fixture.graph, {
      edges: fixture.input('edges', new Uint32Array([0, 1]), 'uint32x2', 1),
      conductances: fixture.input('conductance', new Float32Array([conductance]), 'float32', 1),
      vertexCount: 2,
      source: 0,
      sink: 1,
      maximumIterations: iterations
    });
    const status = publish(fixture, 'status', result.status);
    const diagnostics = publish(fixture, 'diagnostics', result.outputs.diagnostics);
    fixture.run();
    expect(await fixture.readUint32(status, 4)).toEqual([0, 0, 0, invalid]);
    expect((await fixture.readFloat32(diagnostics, 4))[0]).toBeNaN();
  } finally {
    fixture.destroy();
  }
});

it('segregation counterfactual components conserve observed change', async () => {
  const fixture = await createFixture('segregation-decomposition');
  try {
    const populations = reference.decomposition.populations.map((values, index) =>
      fixture.input(`population-${index}`, Float32Array.from(values), 'float32x2', 2)
    );
    const result = addSegregationDecompositionRecipe(fixture.graph, {
      populations: [populations[0], populations[1], populations[2], populations[3]],
      supportId: 'two-aligned-zones',
      counterfactualMethod: 'explicit-independent-fixture'
    });
    const indices = publish(fixture, 'indices', result.outputs.indices);
    const decomposition = publish(fixture, 'decomposition', result.outputs.decomposition);
    fixture.run();
    await expectClose(fixture, indices, reference.decomposition.indices);
    await expectClose(fixture, decomposition, reference.decomposition.decomposition);
  } finally {
    fixture.destroy();
  }
});

it('street profiles retain side assignment and canyon height-width ratio', async () => {
  const fixture = await createFixture('street-profile');
  try {
    const result = addStreetProfilesRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0]), 'float32x2', 1),
      normals: fixture.input('normals', new Float32Array([0, 1]), 'float32x2', 1),
      segments: fixture.input(
        'segments',
        new Float32Array([-10, 2, 10, 2, -10, -3, 10, -3]),
        'float32x4',
        2
      ),
      segmentBuildingIds: fixture.input('owners', new Uint32Array([0, 1]), 'uint32', 2),
      buildingHeights: fixture.input('heights', new Float32Array([6, 9]), 'float32', 2),
      maximumDistance: 10,
      sampleSpacing: 1
    });
    const profiles = publish(fixture, 'profiles', result.outputs.profiles);
    const canyon = publish(fixture, 'canyon', result.outputs.canyon);
    const frontage = publish(fixture, 'frontage', result.outputs.frontage);
    const validity = publish(fixture, 'validity', result.outputs.validity);
    fixture.run();
    await expectClose(fixture, profiles, [2, 3, 6, 9]);
    await expectClose(fixture, canyon, [5, 7.5, 1.5, 5]);
    await expectClose(fixture, frontage, [1, 0, 0, 1]);
    expect(await fixture.readUint32(validity, 1)).toEqual([3]);
  } finally {
    fixture.destroy();
  }
});

it('sampled isovist agrees with independent Shapely ray intersections', async () => {
  const fixture = await createFixture('isovist');
  try {
    const result = addIsovistsRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0, 2, 0]), 'float32x2', 2),
      segments: fixture.input('segments', BOX_SEGMENTS, 'float32x4', 4),
      rayCount: 8,
      maximumDistance: 10
    });
    const endpoints = publish(fixture, 'endpoints', result.outputs.endpoints);
    const measures = publish(fixture, 'measures', result.outputs.localMeasures);
    fixture.run();
    await expectClose(fixture, endpoints, [...reference.isovist.endpoints, ...Array(16).fill(NaN)]);
    await expectClose(fixture, measures, [...reference.isovist.measures, NaN, NaN, NaN, 0]);
  } finally {
    fixture.destroy();
  }
});

it('visibility hop integration and choice agree with NetworkX and expose edge truncation', async () => {
  const fixture = await createFixture('visibility');
  try {
    const result = addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: fixture.input('segments', new Float32Array([0, 10, 2, 10]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 1
    });
    const choice = publish(fixture, 'choice', result.outputs.choice);
    const metrics = publish(fixture, 'metrics', result.outputs.metrics);
    const status = publish(fixture, 'status', result.status);
    fixture.run();
    await expectClose(fixture, choice, reference.visibility.choice);
    await expectClose(fixture, metrics, reference.visibility.metrics.flat());
    expect(await fixture.readUint32(status, 4)).toEqual([1, 2, 1, 0]);
  } finally {
    fixture.destroy();
  }
});

it('shared wall and axial orientation distinguish positive wall length from point contact', async () => {
  const fixture = await createFixture('shared-wall');
  try {
    const result = addBuildingAdjacencyRecipe(fixture.graph, {
      sharedPaths: {
        runs: {
          positions: fixture.input('positions', new Float32Array([0, 0, 0, 2]), 'float32x2', 2),
          offsets: fixture.input('offsets', new Uint32Array([0, 2]), 'uint32', 2),
          leftSourceIds: fixture.input('left', new Uint32Array([0]), 'uint32', 1),
          rightSourceIds: fixture.input('right', new Uint32Array([1]), 'uint32', 1),
          forward: fixture.input('forward', new Uint32Array([0]), 'uint32', 1)
        },
        status: {
          count: fixture.input('count', new Uint32Array([1]), 'uint32', 1),
          overflow: fixture.input('shared-path-loss', new Uint32Array([0]), 'uint32', 1)
        }
      },
      lineBuildingIds: fixture.input('owners', new Uint32Array([0, 1, 2]), 'uint32', 3),
      orientations: fixture.input(
        'orientations',
        new Float32Array([Math.PI * 0.49, -Math.PI * 0.49, 0]),
        'float32',
        3
      )
    });
    const pairs = publish(fixture, 'pairs', result.outputs.pairs);
    fixture.run();
    const actual = await fixture.readFloat32(pairs, 36);
    expect(actual[4]).toBeCloseTo(2, 5);
    expect(actual[5]).toBeCloseTo(0.02 * Math.PI, 5);
    expect(actual[6]).toBe(1);
    expect(actual[8]).toBe(0);
    for (let building = 0; building < 3; building++) {
      let shared = 0;
      for (let neighbor = 0; neighbor < 3; neighbor++)
        shared += actual[4 * (3 * building + neighbor)];
      expect(shared).toBeCloseTo(directReference.sharedWalls.lengths[building], 5);
    }
  } finally {
    fixture.destroy();
  }
});

it('boundary-sampled building tessellation dissolves fragments into complementary enclosed cells', async () => {
  const fixture = await createFixture('building-tessellation');
  try {
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
      capacity: {intersections: 256, nodedSegments: 256, vertices: 128, rings: 32}
    });
    const complete = publish(fixture, 'complete', result.status.complete);
    const readbacks = result.outputs.map((output, index) => ({
      count: publish(fixture, `count-${index}`, output.status.count),
      positions: publish(fixture, `positions-${index}`, output.geometry.positions),
      features: publish(fixture, `features-${index}`, output.geometry.featureOffsets),
      polygons: publish(fixture, `polygons-${index}`, output.geometry.polygonOffsets),
      rings: publish(fixture, `rings-${index}`, output.geometry.ringOffsets)
    }));
    fixture.run();
    expect(await fixture.readUint32(complete, 2)).toEqual([1, 1]);
    for (const [building, output] of readbacks.entries()) {
      const [count] = await fixture.readUint32(output.count, 1);
      expect(count).toBeGreaterThan(0);
      const features = await fixture.readUint32(output.features, count + 1);
      const polygons = await fixture.readUint32(output.polygons, features[count] + 1);
      const ringCount = polygons.at(-1)!;
      const rings = await fixture.readUint32(output.rings, ringCount + 1);
      const positions = await fixture.readFloat32(output.positions, rings[ringCount] * 2);
      let area = 0;
      for (let ring = 0; ring < ringCount; ring++)
        for (let vertex = rings[ring]; vertex < rings[ring + 1]; vertex++) {
          const next = vertex + 1 === rings[ring + 1] ? rings[ring] : vertex + 1;
          area +=
            0.5 *
            (positions[2 * vertex] * positions[2 * next + 1] -
              positions[2 * vertex + 1] * positions[2 * next]);
        }
      expect(Math.abs(area)).toBeCloseTo(reference.tessellation.areas[building], 2);
    }
  } finally {
    fixture.destroy();
  }
});

it('resistance iteration exhaustion keeps candidate voltage but withholds a resistance certificate', async () => {
  const fixture = await createFixture('resistance-exhausted');
  try {
    const result = addResistanceConnectivityRecipe(fixture.graph, {
      edges: fixture.input('edges', new Uint32Array([0, 1, 1, 2, 2, 3]), 'uint32x2', 3),
      conductances: fixture.input('conductances', new Float32Array([1, 1, 1]), 'float32', 3),
      vertexCount: 4,
      source: 0,
      sink: 3,
      maximumIterations: 1
    });
    const status = publish(fixture, 'status', result.status);
    const diagnostics = publish(fixture, 'diagnostics', result.outputs.diagnostics);
    fixture.run();
    expect(await fixture.readUint32(status, 4)).toEqual([0, 1, 1, 0]);
    const actual = await fixture.readFloat32(diagnostics, 4);
    expect(actual[0]).toBeNaN();
    expect(actual[1]).toBeGreaterThan(0.1);
  } finally {
    fixture.destroy();
  }
});

it('street missing sides remain censored and boundary viewpoints never produce observed canyon width', async () => {
  const fixture = await createFixture('street-censored');
  try {
    const result = addStreetProfilesRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0, 0, 2]), 'float32x2', 2),
      normals: fixture.input('normals', new Float32Array([0, 1, 0, 1]), 'float32x2', 2),
      segments: fixture.input('segments', new Float32Array([-10, 2, 10, 2]), 'float32x4', 1),
      segmentBuildingIds: fixture.input('owners', new Uint32Array([0]), 'uint32', 1),
      buildingHeights: fixture.input('heights', new Float32Array([6]), 'float32', 1),
      maximumDistance: 10,
      sampleSpacing: 1
    });
    const canyon = publish(fixture, 'canyon', result.outputs.canyon);
    const validity = publish(fixture, 'validity', result.outputs.validity);
    const frontage = publish(fixture, 'frontage', result.outputs.frontage);
    fixture.run();
    await expectClose(fixture, canyon, [NaN, NaN, NaN, 12, NaN, NaN, NaN, NaN]);
    expect(await fixture.readUint32(validity, 2)).toEqual([1, 7]);
    await expectClose(fixture, frontage, [1, 0]);
  } finally {
    fixture.destroy();
  }
});

it('closed obstacles block visibility while a path around their endpoint remains connected', async () => {
  const fixture = await createFixture('visibility-obstacle');
  try {
    const result = addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([-1, 0, 1, 0, 0, 1]), 'float32x2', 3),
      segments: fixture.input('segments', new Float32Array([0, -1, 0, 0.5]), 'float32x4', 1),
      maximumDistance: 10,
      edgeCapacity: 3
    });
    const adjacency = publish(fixture, 'adjacency', result.outputs.adjacency);
    const choice = publish(fixture, 'choice', result.outputs.choice);
    fixture.run();
    expect(await fixture.readUint32(adjacency, 9)).toEqual([0, 0, 1, 0, 0, 1, 1, 1, 0]);
    await expectClose(fixture, choice, [0, 0, 1]);
  } finally {
    fixture.destroy();
  }
});

it('tessellation sample overflow withholds completeness and exposes the exact requirement', async () => {
  const fixture = await createFixture('tessellation-overflow');
  try {
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
      sampleCapacity: 4,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 64, vertices: 32, rings: 8}
    });
    const sampling = publish(fixture, 'sampling', result.status.sampling);
    const complete = publish(fixture, 'complete', result.status.complete);
    fixture.run();
    expect(await fixture.readUint32(sampling, 4)).toEqual([4, 8, 1, 0]);
    expect(await fixture.readUint32(complete, 2)).toEqual([0, 0]);
  } finally {
    fixture.destroy();
  }
});

// URB-105: independently count orthogonal directed contacts on interior-background:
// class areas 5,6, adjacency [[8,2],[2,14]]. Area weighting gives probabilities
// [(5/11)*(8/10),(5/11)*(2/10),(6/11)*(2/16),(6/11)*(14/16)]; joint adjacency
// gives [8/26,2/26,2/26,14/26]. Evaluate 100*(1+sum(p*ln(p))/(2*ln(2))) in
// Python binary64. The explicit adjacency value also matches landscapemetrics 2.2.1.
it.each([
  [undefined, 19.066547909632227],
  ['adjacency', 21.32987737911165]
] as const)('URB-105 interior-background contagion has independent %s weighting', async (contagionWeighting, expected) => {
  const fixture = await createFixture('contagion-weighting');
  try {
    const landscape = fixture.output('landscape', 'float32x4', 1);
    const adjacency = fixture.output('adjacency', 'uint32', 4);
    addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input(
        'patches',
        new Uint32Array([0, 0, 1, 1, 0, 0xffffffff, 1, 1, 0, 0, 1, 1]),
        'uint32',
        12
      ),
      patchClasses: fixture.input('classes', new Uint32Array([0, 1]), 'uint32', 2),
      width: 4,
      height: 3,
      classCount: 2,
      cellSize: [1, 1],
      connectivity: 4,
      contagionWeighting,
      output: {landscape: landscape.view, classAdjacency: adjacency.view}
    });
    fixture.run();
    expect(await fixture.readUint32(adjacency, 4)).toEqual([8, 2, 2, 14]);
    await expectClose(fixture, landscape, [expected, 5.545454545454546, 11, 2], 1e-5);
  } finally {
    fixture.destroy();
  }
});

it('URB-105 vertex-event isovist has distinct rectangle area and perimeter', async () => {
  const fixture = await createFixture('vertex-square');
  try {
    const measures = fixture.output('measures', 'float32x4', 1);
    addIsovistsRecipe(fixture.graph, {
      positions: fixture.input('position', new Float32Array([1.5, 3.5]), 'float32x2', 1),
      segments: fixture.input(
        'segments',
        new Float32Array([0, 0, 3, 0, 3, 0, 3, 7, 3, 7, 0, 7, 0, 7, 0, 0]),
        'float32x4',
        4
      ),
      sampling: 'vertices',
      rayCount: 3,
      maximumDistance: 10,
      output: {localMeasures: measures.view}
    });
    fixture.run();
    // The 3-by-7 rectangle has area 21, perimeter 20 and corner distance sqrt(14.5).
    // All corners are endpoint events; the three base rays alone cannot recover its area.
    await expectClose(fixture, measures, [21, 20, 3.8078865529319543, 1], 1e-5);
  } finally {
    fixture.destroy();
  }
});

it('URB-105 seeded Philox exchanges match independent integer counter/key draws and rational inference', async () => {
  const fixture = await createFixture('seeded-exchange');
  try {
    const baseline = fixture.input(
      'baseline',
      new Float32Array([12, 3, 2, 9, 7, 4, 3, 11]),
      'float32x2',
      4
    );
    const final = fixture.input(
      'final',
      new Float32Array([5, 8, 6, 3, 8, 7, 4, 5]),
      'float32x2',
      4
    );
    const distribution = fixture.output('distribution', 'float32', 16);
    const inference = fixture.output('inference', 'float32x4', 1);
    addSegregationDecompositionRecipe(fixture.graph, {
      populations: [baseline, baseline, final, final],
      supportId: 'four-units',
      counterfactualMethod: 'supplied',
      randomization: {seed: 29, replicates: 16}, // No supplied masks: exercise the shader generator.
      output: {distribution: distribution.view, inference: inference.view}
    });
    fixture.run();
    // Independent Python arbitrary-precision products implement Random123 Philox4x32-10,
    // counter=(unit,replicate,0,0), key=(29,604). Verified against Random123's zero
    // vector 6627e8d5 e169c58d bc57ac4c 9b00dbd8. Ten rounds:
    // a=0xd2511f53*c0; b=0xcd9e8d57*c2;
    // c=(hi(b)^c1^k0,lo(b),hi(a)^c3^k1,lo(a));
    // k+=(0x9e3779b9,0xbb67ae85) modulo 2^32; exchange iff c0 & 1.
    // Four-bit masks (unit 0 first) are 0000,0110,1000,1110,1010,0010,1111,1000,
    // 1110,1010,0110,1110,0001,0010,1111,1001. Fraction arithmetic evaluates
    // D = 1/2 sum |focal/sum(focal)-complement/sum(complement)| after exchange.
    // Replicates 0,6,14 tie |observed|; all others are strictly smaller: p=(3+1)/(16+1).
    await expectClose(
      fixture,
      distribution,
      [
        -0.3584943639291465, -0.10573239021514884, -0.0642156862745098, 0.13513061650992686,
        -0.07296113847837986, -0.22575757575757577, 0.3584943639291465, -0.0642156862745098,
        0.13513061650992686, -0.07296113847837986, -0.10573239021514884, 0.13513061650992686,
        -0.13513061650992686, -0.22575757575757577, 0.3584943639291465, 0.10573239021514884
      ],
      1e-6
    );
    await expectClose(fixture, inference, [-0.3584943639291465, 0.23529411764705882, 16, 3], 1e-6);
  } finally {
    fixture.destroy();
  }
});

it('URB-105 anisotropic cells reconcile diagonal tile seams with 8-connectivity', async () => {
  const fixture = await createFixture('anisotropic-diagonal-seams');
  try {
    const result = addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]), 'uint32', 9),
      patchClasses: fixture.input(
        'classes',
        new Uint32Array([0, 1, 2, 1, 0, 2, 2, 2, 0]),
        'uint32',
        9
      ),
      width: 3,
      height: 3,
      cellSize: [2, 3],
      classCount: 3,
      connectivity: 8,
      reconcilePatches: true
    });
    const labels = publish(fixture, 'labels', result.provenance.reconciledPatches!);
    const parents = publish(fixture, 'parents', result.provenance.patchParents!);
    const patches = publish(fixture, 'patch-metrics', result.outputs.patchMetrics);
    const classes = publish(fixture, 'class-metrics', result.outputs.classMetrics);
    const adjacency = publish(fixture, 'adjacency', result.outputs.classAdjacency);
    const landscape = publish(fixture, 'landscape', result.outputs.landscape);
    const horizontal = addLandscapeMetricsRecipe(fixture.graph, {
      id: 'horizontal',
      patches: fixture.input('horizontal-labels', new Uint32Array([0, 0]), 'uint32', 2),
      patchClasses: fixture.input('horizontal-class', new Uint32Array([0]), 'uint32', 1),
      width: 2,
      height: 1,
      cellSize: [2, 3],
      classCount: 1,
      connectivity: 8
    });
    const horizontalMetrics = publish(
      fixture,
      'horizontal-metrics',
      horizontal.outputs.patchMetrics
    );
    fixture.run();
    // Two horizontally adjacent cells form a 4-by-3 rectangle: area 12, perimeter 14,
    // two cells, two directed like-side contacts. Exchanging cell axes would give perimeter 16.
    await expectClose(fixture, horizontalMetrics, [12, 14, 2, 2]);
    // Hand union of eight-neighbour same-class cells gives roots 0,1,2. Class 0 joins
    // ONLY at corners, as do class-1 cells; 4-connectivity would keep those tiles apart.
    // Areas are 3,2,4 cells times 6. Each cell has perimeter 10; class 2 shares one
    // vertical side (length 2) and one horizontal side (length 3): perimeter=40-2*(2+3)=30.
    // Directed orthogonal contacts remain [[0,4,4],[4,0,2],[4,2,4]], irrespective of
    // connectivity. Python ln/sqrt on these counts gives the literals below:
    // mesh_i=6*n_i^2/9; cohesion_i=100*(1-1/sqrt(n_i))/(1-1/sqrt(9)).
    expect(await fixture.readUint32(labels, 9)).toEqual([0, 1, 2, 1, 0, 2, 2, 2, 0]);
    expect(await fixture.readUint32(parents, 9)).toEqual([0, 1, 2, 1, 0, 2, 2, 2, 0]);
    expect(await fixture.readUint32(adjacency, 9)).toEqual([0, 4, 4, 4, 0, 2, 4, 2, 4]);
    await expectClose(fixture, patches, [
      18,
      30,
      3,
      0,
      12,
      20,
      2,
      0,
      24,
      30,
      4,
      4,
      ...new Array(24).fill(0)
    ]);
    await expectClose(
      fixture,
      classes,
      [
        18, 30, 6, 63.39745962155612, 12, 20, 2.6666666666666665, 43.933982822017875, 24, 30,
        10.666666666666666, 75
      ]
    );
    await expectClose(fixture, landscape, [13.426828194365381, 19.333333333333332, 54, 3]);
  } finally {
    fixture.destroy();
  }
});

it('URB-104 absent side retains the no-owner sentinel and finite range-censored distance', async () => {
  const fixture = await createFixture('no-facade-hit');
  try {
    const result = addStreetProfilesRecipe(fixture.graph, {
      positions: fixture.input('position', new Float32Array([0, 0]), 'float32x2', 1),
      normals: fixture.input('normal', new Float32Array([0, 1]), 'float32x2', 1),
      segments: fixture.input('segments', new Float32Array([-2, 20, 2, 20]), 'float32x4', 1),
      segmentBuildingIds: fixture.input('owner', new Uint32Array([0]), 'uint32', 1),
      buildingHeights: fixture.input('height', new Float32Array([6]), 'float32', 1),
      streetSegment: fixture.input('street', new Float32Array([-4, 0, 4, 0]), 'float32x4', 1),
      maximumDistance: 5,
      sampleSpacing: 1
    });
    const profiles = publish(fixture, 'profiles', result.outputs.profiles);
    const buildings = publish(fixture, 'buildings', result.outputs.buildings);
    const lengths = publish(fixture, 'facades', result.facades!.lengths);
    const frontage = publish(fixture, 'frontage', result.facades!.frontage);
    fixture.run();
    // Wall y=20 lies outside the +/-5 corridor. Neither ray has an owner.
    await expectClose(fixture, profiles, [5, 5, NaN, NaN]);
    expect(await fixture.readUint32(buildings, 2)).toEqual([0xffffffff, 0xffffffff]);
    await expectClose(fixture, lengths, [0, 0]);
    await expectClose(fixture, frontage, [0, 0]);
  } finally {
    fixture.destroy();
  }
});
