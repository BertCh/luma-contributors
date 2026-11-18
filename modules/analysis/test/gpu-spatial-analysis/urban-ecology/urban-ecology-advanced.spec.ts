// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  addSegregationCounterfactualRecipe,
  addPopulationSupportAlignmentRecipe,
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addSegregationDecompositionRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {
  addStreetProfilesRecipe,
  addIsovistsRecipe,
  addVisibilityGraphRecipe,
  addBuildingTessellationRecipe,
  addBuildingAdjacencyRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology-advanced.json';
import depthmapReference from '../../oracles/fixtures/urban-depthmap.json';
import landscapeReference from '../../oracles/fixtures/urban-landscape.json';

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
async function expectClose(
  fixture: RecipeTestFixture,
  output: ReturnType<typeof publish>,
  values: number[],
  tolerance = 1e-4
) {
  const actual = await fixture.readFloat32(output, values.length);
  values.forEach((expected, index) => {
    if (Number.isNaN(expected)) expect(actual[index]).toBeNaN();
    else
      expect(Math.abs(actual[index] - expected)).toBeLessThanOrEqual(
        tolerance * (1 + Math.abs(expected))
      );
  });
}

it('vertex-event isovist captures a narrow obstacle missed by uniform angular rays', async () => {
  const fixture = await createFixture('vertex-isovist');
  try {
    const result = addIsovistsRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0]), 'float32x2', 1),
      segments: fixture.input('segments', new Float32Array([2, 0.2, 2, 0.3]), 'float32x4', 1),
      rayCount: 8,
      sampling: 'vertices',
      maximumDistance: 5
    });
    const hits = publish(fixture, 'hits', result.outputs.segments);
    const angles = publish(fixture, 'angles', result.outputs.angles);
    fixture.run();
    const values = await fixture.readUint32(hits, 14);
    expect(values.filter(value => value === 0).length).toBeGreaterThanOrEqual(2);
    const ordered = await fixture.readFloat32(angles, 14);
    expect(ordered).toEqual([...ordered].sort((a, b) => a - b));
  } finally {
    fixture.destroy();
  }
});

it.each([
  'composition',
  'dual-composition',
  'share'
] as const)('counterfactual %s matches direct segregation 2.5.3 including zero populations and tied ranks', async method => {
  const fixture = await createFixture(`counterfactual-${method}`);
  try {
    const result = addSegregationCounterfactualRecipe(fixture.graph, {
      baseline: fixture.input(
        'baseline',
        Float32Array.from(reference.counterfactual.baseline),
        'float32x2',
        4
      ),
      final: fixture.input(
        'final',
        Float32Array.from(reference.counterfactual.final),
        'float32x2',
        4
      ),
      supportId: 'same-four-units',
      method
    });
    const first = publish(fixture, 'first', result.counterfactuals.compositionOnly);
    const second = publish(fixture, 'second', result.counterfactuals.spatialOnly);
    const decomposition = publish(fixture, 'decomposition', result.outputs.decomposition);
    fixture.run();
    const expected =
      reference.counterfactual.methods[method === 'dual-composition' ? 'dual_composition' : method];
    await expectClose(fixture, first, expected.populations[0]);
    await expectClose(fixture, second, expected.populations[1]);
    await expectClose(fixture, decomposition, expected.decomposition);
  } finally {
    fixture.destroy();
  }
});
it('paired label randomization replays all assignments against direct segregation statistics', async () => {
  const fixture = await createFixture('segregation-randomization');
  try {
    const baseline = fixture.input(
      'baseline',
      Float32Array.from(reference.counterfactual.baseline),
      'float32x2',
      4
    );
    const final = fixture.input(
      'final',
      Float32Array.from(reference.counterfactual.final),
      'float32x2',
      4
    );
    const result = addSegregationDecompositionRecipe(fixture.graph, {
      populations: [baseline, baseline, final, final],
      supportId: 'same-four-units',
      counterfactualMethod: 'supplied',
      randomization: {
        seed: 7,
        replicates: 16,
        swaps: fixture.input('swaps', Uint32Array.from(reference.randomization.swaps), 'uint32', 64)
      }
    });
    const distribution = publish(fixture, 'distribution', result.randomization!.distribution);
    const inference = publish(fixture, 'inference', result.randomization!.inference);
    fixture.run();
    await expectClose(fixture, distribution, reference.randomization.distribution);
    expect((await fixture.readFloat32(inference, 4))[1]).toBeCloseTo(
      reference.randomization.pValue,
      5
    );
  } finally {
    fixture.destroy();
  }
});
it('support alignment reorders explicit IDs and rejects absent/duplicated support', async () => {
  const fixture = await createFixture('support-alignment');
  try {
    const result = addPopulationSupportAlignmentRecipe(fixture.graph, {
      populations: fixture.input('populations', new Float32Array([2, 8, 6, 4]), 'float32x2', 2),
      sourceIds: fixture.input('source', new Uint32Array([20, 10]), 'uint32', 2),
      targetIds: fixture.input('target', new Uint32Array([10, 20, 30]), 'uint32', 3)
    });
    const populations = publish(fixture, 'populations-output', result.outputs.populations);
    const alignmentCode = publish(fixture, 'alignment-code', result.outputs.alignmentCode);
    const matchedMask = publish(fixture, 'matched-mask', result.outputs.matchedMask);
    const status = publish(fixture, 'status', result.packedStatus);
    const stageCounts = result.status.stages.map(stage =>
      publish(fixture, `status-${stage.stage}`, stage.status.invalidCount)
    );
    fixture.run();
    await expectClose(fixture, populations, [6, 4, 2, 8, NaN, NaN]);
    expect(await fixture.readUint32(alignmentCode, 3)).toEqual([0, 0, 1]);
    expect(await fixture.readUint32(matchedMask, 3)).toEqual([1, 1, 0]);
    expect(await fixture.readUint32(status, 4)).toEqual([0, 1, 0, 0]);
    for (const [index, count] of stageCounts.entries()) {
      expect(await fixture.readUint32(count, 1)).toEqual([index === 0 ? 1 : 0]);
    }
  } finally {
    fixture.destroy();
  }
});
it('exact facade intervals distinguish visible facade arc length from sampled frontage', async () => {
  const fixture = await createFixture('facade-frontage');
  try {
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
      streetSegment: fixture.input('street', new Float32Array([-4, 0, 4, 0]), 'float32x4', 1)
    });
    const frontage = publish(fixture, 'facades', result.facades!.frontage);
    const canyon = publish(fixture, 'canyon', result.outputs.canyon);
    fixture.run();
    await expectClose(fixture, frontage, [8, 0, 0, 0, 0, 8]);
    const values = await fixture.readFloat32(canyon, 4);
    expect(values[0]).toBeCloseTo(reference.streetProfile.width, 5);
  } finally {
    fixture.destroy();
  }
});
it('landscape perimeter policies distinguish domain edges from background contacts', async () => {
  const fixture = await createFixture('landscape-boundary-policy');
  try {
    const result = addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array([0, 0, 1]), 'uint32', 3),
      patchClasses: fixture.input('classes', new Uint32Array([0, 1]), 'uint32', 2),
      width: 3,
      height: 1,
      classCount: 2,
      cellSize: [1, 1],
      connectivity: 4,
      backgroundClass: 1,
      countBoundary: false,
      countBackground: true
    });
    const metrics = publish(fixture, 'patch-metrics', result.outputs.patchMetrics);
    const landscape = publish(fixture, 'landscape', result.outputs.landscape);
    fixture.run();
    expect((await fixture.readFloat32(metrics, 8))[1]).toBe(1);
    expect((await fixture.readFloat32(landscape, 4))[2]).toBe(2);
  } finally {
    fixture.destroy();
  }
});
it('nonconvex enclosed tessellation agrees with direct momepy areas', async () => {
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
        6
      ),
      enclosureMode: 'simple',
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

it('same-class patch stitching reconciles tile seams before core and proximity summaries', async () => {
  const fixture = await createFixture('patch-seam-reconciliation');
  try {
    const patches = fixture.input(
      'patches',
      new Uint32Array([0, 0, 1, 1, 0, 0, 1, 1]),
      'uint32',
      8
    );
    const patchClasses = fixture.input('classes', new Uint32Array([0, 0]), 'uint32', 2);
    const landscape = addLandscapeMetricsRecipe(fixture.graph, {
      patches,
      patchClasses,
      width: 4,
      height: 2,
      classCount: 1,
      cellSize: [1, 1],
      connectivity: 4,
      reconcilePatches: true
    });
    const proximity = addPatchProximityRecipe(fixture.graph, {
      patches,
      patchClasses,
      width: 4,
      height: 2,
      cellSize: [1, 1],
      edgeDepth: 0.4,
      searchRadius: 5,
      reconcilePatches: true,
      connectivity: 4
    });
    const metrics = publish(fixture, 'metrics', landscape.outputs.patchMetrics);
    const core = publish(fixture, 'core', proximity.outputs.core);
    fixture.run();
    await expectClose(fixture, metrics, [8, 12, 8, 20, 0, 0, 0, 0]);
    expect(await fixture.readUint32(core, 8)).toEqual(new Array(8).fill(1));
  } finally {
    fixture.destroy();
  }
});

it('visibility reports Hillier-Hanson integration separately from normalized undirected choice', async () => {
  const fixture = await createFixture('visibility-normalization');
  try {
    const result = addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([-1, 0, 1, 0, 0, 1]), 'float32x2', 3),
      segments: fixture.input('segments', new Float32Array([0, -1, 0, 0.5]), 'float32x4', 1),
      maximumDistance: 10,
      edgeCapacity: 3
    });
    const integration = publish(fixture, 'integration', result.outputs.integration);
    const choice = publish(fixture, 'normalized-choice', result.outputs.normalizedChoice);
    fixture.run();
    const values = await fixture.readFloat32(integration, 12);
    expect(values[0]).toBeCloseTo(1.5, 5);
    expect(values[3]).toBeCloseTo(3 * Math.log2(5 / 3) - 2, 5);
    expect(values[8]).toBeCloseTo(1, 5);
    expect(values[11]).toBe(Infinity);
    await expectClose(fixture, choice, [0, 0, 1]);
  } finally {
    fixture.destroy();
  }
});
it('enclosure holes are preserved against direct momepy reference', async () => {
  const fixture = await createFixture('building-tessellation');
  try {
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input(
        'segments',
        Float32Array.from(reference.holedTessellation.segments),
        'float32x4',
        8
      ),
      segmentBuildingIds: fixture.input(
        'owners',
        Uint32Array.from(reference.holedTessellation.owners),
        'uint32',
        8
      ),
      buildingCount: 2,
      enclosure: fixture.input(
        'enclosure',
        Float32Array.from(reference.holedTessellation.enclosure),
        'float32x2',
        8
      ),
      enclosureMode: 'simple',
      enclosureRingOffsets: fixture.input(
        'enclosure-rings',
        new Uint32Array([0, 4, 8]),
        'uint32',
        3
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
      expect(Math.abs(area)).toBeCloseTo(reference.holedTessellation.areas[building], 2);
    }
  } finally {
    fixture.destroy();
  }
});

for (const oracle of landscapeReference.cases) {
  it(`landscapemetrics 2.2.1 direct reference: ${oracle.name}`, async () => {
    const fixture = await createFixture(`landscape-direct-${oracle.name}`);
    try {
      const props = {
        patches: fixture.input(
          'patches',
          Uint32Array.from(oracle.patches),
          'uint32',
          oracle.patches.length
        ),
        patchClasses: fixture.input('classes', Uint32Array.from(oracle.patchClasses), 'uint32', 2),
        width: oracle.width,
        height: oracle.height,
        classCount: 2,
        connectivity: 4 as const,
        cellSize: [oracle.cellSize[0], oracle.cellSize[1]] as const
      };
      const result = addLandscapeMetricsRecipe(fixture.graph, {
        ...props,
        id: 'all-edges',
        contagionWeighting: 'adjacency'
      });
      const internal = addLandscapeMetricsRecipe(fixture.graph, {
        ...props,
        id: 'internal-edges',
        countBoundary: false,
        countBackground: false
      });
      const patches = publish(fixture, 'patches-result', result.outputs.patchMetrics);
      const classes = publish(fixture, 'classes-result', result.outputs.classMetrics);
      const internalClasses = publish(fixture, 'internal-result', internal.outputs.classMetrics);
      const aggregation = publish(fixture, 'aggregation-result', result.outputs.aggregation);
      const landscape = publish(fixture, 'landscape-result', result.outputs.landscape);
      const landscapeAggregation = publish(
        fixture,
        'landscape-aggregation',
        result.outputs.landscapeAggregation
      );
      const landscapeCohesion = publish(
        fixture,
        'landscape-cohesion',
        result.outputs.landscapeCohesion
      );
      fixture.run();
      const patchValues = await fixture.readFloat32(patches, 8);
      const classValues = await fixture.readFloat32(classes, 8);
      const internalValues = await fixture.readFloat32(internalClasses, 8);
      for (let category = 0; category < 2; category++) {
        expect(patchValues[4 * category]).toBeCloseTo(oracle.area[category], 4);
        expect(patchValues[4 * category + 1]).toBeCloseTo(oracle.perimeter[category], 4);
        expect(classValues[4 * category + 1]).toBeCloseTo(oracle.boundaryEdge[category], 4);
        expect(classValues[4 * category + 2]).toBeCloseTo(oracle.mesh[category], 4);
        expect(classValues[4 * category + 3]).toBeCloseTo(oracle.cohesion[category], 3);
        expect(internalValues[4 * category + 1]).toBeCloseTo(oracle.internalEdge[category], 4);
      }
      await expectClose(
        fixture,
        aggregation,
        oracle.aggregation.map(value => value ?? NaN)
      );
      await expectClose(fixture, landscapeAggregation, [oracle.landscapeAggregation]);
      await expectClose(fixture, landscapeCohesion, [oracle.landscapeCohesion]);
      const summary = await fixture.readFloat32(landscape, 4);
      expect(summary[0]).toBeCloseTo(oracle.contagion, 3);
      expect(summary[1]).toBeCloseTo(oracle.landscapeMesh, 4);
    } finally {
      fixture.destroy();
    }
  });
}

for (const boundary of ['edge', 'open'] as const) {
  it(`direct landscapemetrics raster core and ENN with ${boundary} domain boundary`, async () => {
    const fixture = await createFixture(`core-enn-${boundary}`);
    try {
      const oracle = landscapeReference.proximity;
      const result = addPatchProximityRecipe(fixture.graph, {
        patches: fixture.input(
          'patches',
          Uint32Array.from(oracle.patches),
          'uint32',
          oracle.patches.length
        ),
        patchClasses: fixture.input('classes', new Uint32Array([0, 0]), 'uint32', 2),
        width: oracle.width,
        height: oracle.height,
        cellSize: [1, 1],
        edgeDepth: 1,
        searchRadius: 4,
        distanceModel: 'cell-center',
        coreModel: 'raster-erosion',
        coreBoundary: boundary
      });
      const metrics = publish(fixture, 'result', result.outputs.metrics);
      fixture.run();
      const expected = oracle.area.flatMap((area, index) => [
        area,
        (boundary === 'edge' ? oracle.coreEdge : oracle.coreOpen)[index],
        oracle.nearest[index],
        oracle.composedProximity[index]
      ]);
      await expectClose(fixture, metrics, expected);
    } finally {
      fixture.destroy();
    }
  });
}

it('P-value, Teklenburg and Penn normalization match compiled pinned depthmapX helpers', async () => {
  const fixture = await createFixture('depthmap-normalizations');
  try {
    const result = addVisibilityGraphRecipe(fixture.graph, {
      positions: fixture.input(
        'positions',
        Float32Array.from(depthmapReference.path.positions),
        'float32x2',
        5
      ),
      segments: fixture.input('obstacles', new Float32Array([100, 0, 100, 1]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 8
    });
    const extended = publish(fixture, 'extended', result.outputs.extendedIntegration);
    const integration = publish(fixture, 'integration', result.outputs.integration);
    fixture.run();
    await expectClose(
      fixture,
      extended,
      depthmapReference.path.rows.flatMap((row, index) => [
        row[1],
        row[2],
        row[3],
        depthmapReference.path.maximumDepth[index]
      ])
    );
    const values = await fixture.readFloat32(integration, 20);
    depthmapReference.path.rows.forEach((row, index) =>
      expect(values[4 * index + 3]).toBeCloseTo(row[0], 4)
    );
  } finally {
    fixture.destroy();
  }
});

for (const fillRule of ['even-odd', 'nonzero'] as const) {
  for (const freeSpace of ['inside', 'outside'] as const) {
    it(`filled ring ${fillRule} ${freeSpace} classifies holes, boundaries and nonfinite viewpoints`, async () => {
      const fixture = await createFixture(`filled-region-${fillRule}-${freeSpace}`);
      try {
        const positions = fixture.input(
          'positions',
          new Float32Array([1.5, 0, 0, 0, 3, 0, 2, 0, NaN, 0]),
          'float32x2',
          5
        );
        const segments = fixture.input(
          'segments',
          new Float32Array([
            -2, -2, 2, -2, 2, -2, 2, 2, 2, 2, -2, 2, -2, 2, -2, -2, -1, -1, -1, 1, -1, 1, 1, 1, 1,
            1, 1, -1, 1, -1, -1, -1
          ]),
          'float32x4',
          8
        );
        const result = addIsovistsRecipe(fixture.graph, {
          positions,
          segments,
          rayCount: 8,
          maximumDistance: 5,
          region: {fillRule, freeSpace}
        });
        const classifications = publish(
          fixture,
          'classifications',
          result.outputs.viewpointClassifications
        );
        const validity = publish(fixture, 'validity', result.outputs.validity);
        fixture.run();
        const expected = freeSpace === 'outside' ? [1, 0, 0, 2, 3] : [0, 1, 1, 2, 3];
        expect(await fixture.readUint32(classifications, 5)).toEqual(expected);
        expect(await fixture.readUint32(validity, 40)).toEqual(
          expected.flatMap(value => Array(8).fill(value === 0 ? 1 : 0))
        );
      } finally {
        fixture.destroy();
      }
    });
  }
}

it.each([
  'valid',
  'overflow',
  'uncertain',
  'invalid-contiguity'
] as const)('building component exterior walls preserve direct momepy results and reject incomplete topology: %s', async evidence => {
  const fixture = await createFixture('building-structures');
  try {
    const oracle = reference.structures;
    const result = addBuildingAdjacencyRecipe(fixture.graph, {
      sharedPaths: {
        runs: {
          positions: fixture.input('wall', new Float32Array([2, 0, 2, 2]), 'float32x2', 2),
          offsets: fixture.input('wall-offsets', new Uint32Array([0, 2]), 'uint32', 2),
          leftSourceIds: fixture.input('left', new Uint32Array([0]), 'uint32', 1),
          rightSourceIds: fixture.input('right', new Uint32Array([1]), 'uint32', 1),
          forward: fixture.input('forward', new Uint32Array([0]), 'uint32', 1)
        },
        status: {
          count: fixture.input('count', new Uint32Array([1]), 'uint32', 1),
          overflow: fixture.input(
            'overflow',
            new Uint32Array([Number(evidence === 'overflow')]),
            'uint32',
            1
          ),
          uncertainCount: fixture.input(
            'uncertain',
            new Uint32Array([Number(evidence === 'uncertain')]),
            'uint32',
            1
          )
        }
      },
      lineBuildingIds: fixture.input('owners', new Uint32Array([0, 1, 2]), 'uint32', 3),
      orientations: fixture.input(
        'orientations',
        Float32Array.from(oracle.orientations),
        'float32',
        3
      ),
      contiguity: fixture.input(
        'queen',
        new Uint32Array([0, evidence === 'invalid-contiguity' ? 3 : 1]),
        'uint32x2',
        1
      ),
      structures: {
        geometry: {
          kind: 'polygons',
          positions: fixture.input(
            'positions',
            Float32Array.from(oracle.positions),
            'float32x2',
            oracle.positions.length / 2
          ),
          ringOffsets: fixture.input(
            'rings',
            Uint32Array.from(oracle.ringOffsets),
            'uint32',
            oracle.ringOffsets.length
          ),
          polygonOffsets: fixture.input(
            'polygons',
            Uint32Array.from(oracle.polygonOffsets),
            'uint32',
            oracle.polygonOffsets.length
          ),
          featureOffsets: fixture.input(
            'features',
            Uint32Array.from(oracle.featureOffsets),
            'uint32',
            oracle.featureOffsets.length
          )
        },
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 64, nodedSegments: 128, vertices: 64, rings: 16}
      }
    });
    const morphology = publish(fixture, 'morphology', result.outputs.morphology);
    const components = publish(fixture, 'components', result.outputs.components);
    const perimeters = publish(fixture, 'structure-perimeters', result.structures!.perimeters);
    fixture.run();
    if (evidence !== 'valid') {
      expect(await fixture.readUint32(components, 3)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
      expect((await fixture.readFloat32(perimeters, 3)).every(Number.isNaN)).toBe(true);
      const values = await fixture.readFloat32(morphology, 12);
      for (let building = 0; building < 3; building++) {
        expect(values[4 * building + 2]).toBeNaN();
        expect(values[4 * building + 3]).toBeNaN();
      }
      return;
    }
    await expectClose(
      fixture,
      morphology,
      oracle.perimeterWall.flatMap((perimeter, index) => [
        [2, 2, 0][index],
        oracle.cardinal[index],
        oracle.alignment[index] ?? NaN,
        perimeter
      ])
    );
    expect(await fixture.readUint32(components, 3)).toEqual([0, 0, 2]);
  } finally {
    fixture.destroy();
  }
});

it('isovist rays preserve misses when the finite range cannot represent range plus one', async () => {
  const fixture = await createFixture('large-range-isovist');
  try {
    const result = addIsovistsRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0]), 'float32x2', 1),
      segments: fixture.input('segments', new Float32Array([1, 1, 2, 1]), 'float32x4', 1),
      rayCount: 4,
      maximumDistance: 2 ** 25
    });
    const segments = publish(fixture, 'hits', result.outputs.segments);
    const distances = publish(fixture, 'distances', result.outputs.distances);
    const validity = publish(fixture, 'validity', result.outputs.validity);
    fixture.run();
    expect(await fixture.readUint32(segments, 4)).toEqual(Array(4).fill(0xffffffff));
    expect(await fixture.readFloat32(distances, 4)).toEqual(Array(4).fill(2 ** 25));
    expect(await fixture.readUint32(validity, 4)).toEqual([1, 1, 1, 1]);
  } finally {
    fixture.destroy();
  }
});

it('paired randomization withholds inference when a swapped finite population overflows', async () => {
  const fixture = await createFixture('segregation-randomization-overflow');
  try {
    const baseline = fixture.input(
      'baseline',
      new Float32Array([2e38, 1e38, 1, 1]),
      'float32x2',
      2
    );
    const final = fixture.input('final', new Float32Array([1, 1, 2e38, 1e38]), 'float32x2', 2);
    const result = addSegregationDecompositionRecipe(fixture.graph, {
      populations: [baseline, baseline, final, final],
      supportId: 'same-two-units',
      counterfactualMethod: 'supplied',
      randomization: {
        seed: 7,
        replicates: 2,
        swaps: fixture.input('swaps', new Uint32Array([0, 0, 0, 1]), 'uint32', 4)
      }
    });
    const distribution = publish(fixture, 'distribution', result.randomization!.distribution);
    const inference = publish(fixture, 'inference', result.randomization!.inference);
    const validity = publish(fixture, 'validity', result.outputs.validity);
    fixture.run();
    expect(await fixture.readUint32(validity, 4)).toEqual([1, 1, 1, 1]);
    const samples = await fixture.readFloat32(distribution, 2);
    expect(samples[0]).toBeCloseTo(0, 6);
    expect(samples[1]).toBeNaN();
    const statistics = await fixture.readFloat32(inference, 4);
    expect(statistics[0]).toBeCloseTo(0, 6);
    expect(statistics[1]).toBeNaN();
    expect(statistics[2]).toBe(1);
  } finally {
    fixture.destroy();
  }
});

it.each([
  {
    name: 'background-only',
    labels: [0xffffffff, 0xffffffff],
    countBoundary: true,
    aggregation: NaN,
    cohesion: NaN
  },
  {
    name: 'single-cell',
    labels: [0, 0xffffffff],
    countBoundary: true,
    aggregation: 0,
    cohesion: NaN
  },
  {
    name: 'excluded-perimeter',
    labels: [0, 0],
    countBoundary: false,
    aggregation: 100,
    cohesion: NaN
  },
  {name: 'connected-pair', labels: [0, 0], countBoundary: true, aggregation: 100, cohesion: 100}
])('landscape aggregation and cohesion retain $name denominator conventions', async ({
  name,
  labels,
  countBoundary,
  aggregation,
  cohesion
}) => {
  const fixture = await createFixture(`landscape-denominators-${name}`);
  try {
    const result = addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', Uint32Array.from(labels), 'uint32', 2),
      patchClasses: fixture.input('classes', new Uint32Array([0]), 'uint32', 1),
      width: 2,
      height: 1,
      cellSize: [1, 1],
      classCount: 1,
      connectivity: 4,
      countBoundary
    });
    const landscapeAggregation = publish(
      fixture,
      'aggregation',
      result.outputs.landscapeAggregation
    );
    const landscapeCohesion = publish(fixture, 'cohesion', result.outputs.landscapeCohesion);
    fixture.run();
    await expectClose(fixture, landscapeAggregation, [aggregation]);
    await expectClose(fixture, landscapeCohesion, [cohesion]);
  } finally {
    fixture.destroy();
  }
});
