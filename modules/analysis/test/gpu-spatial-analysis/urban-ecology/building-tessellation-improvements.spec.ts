// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {
  addBuildingTessellationIndex,
  getBuildingTessellationIndexDeclarations,
  getBuildingTessellationGridWalk
} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-index';
import {getBuildingTessellationCellKernel} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-cells';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology-advanced.json';
import {publishBuildingTessellationOutput as publish} from './building-tessellation-test-utils';

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

/** Float64 first-vertex fan area, independent of GPU clipping and reference overlay. */
function getBuildingAreas(
  positions: number[],
  cells: number[],
  vertexCapacity: number,
  buildingCount: number
) {
  const areas = new Array<number>(buildingCount).fill(0);
  for (let piece = 0; piece < cells.length / 4; piece++) {
    const count = cells[4 * piece];
    const owner = cells[4 * piece + 2];
    const base = 2 * piece * vertexCapacity;
    for (let vertex = 1; vertex + 1 < count; vertex++) {
      const first = base + 2 * vertex;
      const second = first + 2;
      areas[owner] +=
        0.5 *
        ((positions[first] - positions[base]) * (positions[second + 1] - positions[base + 1]) -
          (positions[first + 1] - positions[base + 1]) * (positions[second] - positions[base]));
    }
  }
  return areas;
}

it.each([
  {
    name: 'nonconvex L',
    enclosure: [0, 0, 8, 0, 8, 4, 4, 4, 4, 8, 0, 8],
    offsets: [0, 6],
    expected: [28, 20]
  },
  {
    name: 'courtyard hole',
    enclosure: [0, 0, 8, 0, 8, 8, 0, 8, 3, 3, 5, 3, 5, 5, 3, 5],
    offsets: [0, 4, 8],
    expected: [27, 33]
  }
])('bounded clipping supports $name and matches closed-form areas and the reference', async ({
  enclosure,
  offsets,
  expected
}) => {
  const fixture = await createFixture('local-simple-enclosure');
  try {
    const props = {
      segments: fixture.input(
        'walls',
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
      enclosure: fixture.input(
        'enclosure',
        Float32Array.from(enclosure),
        'float32x2',
        enclosure.length / 2
      ),
      enclosureRingOffsets: fixture.input(
        'rings',
        Uint32Array.from(offsets),
        'uint32',
        offsets.length
      ),
      enclosureMode: 'simple' as const,
      buildingCount: 2,
      sampleSpacing: 1,
      sampleCapacity: 8,
      vertexTolerance: 1 / 1024
    };
    const local = addBuildingTessellationRecipe(fixture.graph, {
      ...props,
      id: 'local',
      candidateRadius: 32,
      maximumSamplesPerBuilding: 4
    });
    const exact = addBuildingTessellationRecipe(fixture.graph, {
      ...props,
      id: 'reference',
      mode: 'reference',
      capacity: {intersections: 256, nodedSegments: 256, vertices: 128, rings: 32}
    });
    const positions = publish(fixture, 'positions', local.outputs.positions);
    const cells = publish(fixture, 'cells', local.outputs.cells);
    const status = publish(fixture, 'status', local.outputs.buildingStatus);
    const complete = publish(fixture, 'complete', exact.status.complete);
    const overlays = exact.outputs.map((output, building) => ({
      count: publish(fixture, `exact-count-${building}`, output.status.count),
      features: publish(fixture, `exact-features-${building}`, output.geometry.featureOffsets),
      polygons: publish(fixture, `exact-polygons-${building}`, output.geometry.polygonOffsets),
      rings: publish(fixture, `exact-rings-${building}`, output.geometry.ringOffsets),
      positions: publish(fixture, `exact-positions-${building}`, output.geometry.positions)
    }));
    fixture.run();
    expect(await fixture.readUint32(status, 8)).toEqual([1, 0, 0, 0, 1, 0, 0, 0]);
    expect(await fixture.readUint32(complete, 2)).toEqual([1, 1]);
    const coordinates = await fixture.readFloat32(positions, local.outputs.positions.length * 2);
    const records = await fixture.readUint32(cells, local.outputs.cells.length * 4);
    const areas = getBuildingAreas(coordinates, records, 12, 2);
    for (const [building, area] of areas.entries()) {
      expect(area).toBeCloseTo(expected[building], 4);
      const output = overlays[building];
      const [count] = await fixture.readUint32(output.count, 1);
      const features = await fixture.readUint32(output.features, count + 1);
      const polygons = await fixture.readUint32(output.polygons, features[count] + 1);
      const ringCount = polygons.at(-1)!;
      const rings = await fixture.readUint32(output.rings, ringCount + 1);
      const points = await fixture.readFloat32(output.positions, 2 * rings[ringCount]);
      let exactArea = 0;
      for (let ring = 0; ring < ringCount; ring++) {
        const first = rings[ring];
        for (let vertex = first + 1; vertex + 1 < rings[ring + 1]; vertex++) {
          exactArea +=
            0.5 *
            ((points[2 * vertex] - points[2 * first]) *
              (points[2 * vertex + 3] - points[2 * first + 1]) -
              (points[2 * vertex + 1] - points[2 * first + 1]) *
                (points[2 * vertex + 2] - points[2 * first]));
        }
      }
      expect(Math.abs(area - exactArea)).toBeLessThan(0.01);
    }
    for (let piece = 0; piece < records.length / 4; piece++) {
      const count = records[4 * piece];
      const base = 2 * piece * 12;
      if (count === 0) continue;
      let horizontal = 0;
      let vertical = 0;
      for (let vertex = 0; vertex < count; vertex++) {
        horizontal += coordinates[base + 2 * vertex] / count;
        vertical += coordinates[base + 2 * vertex + 1] / count;
      }
      // Every convex piece centroid must lie inside the requested region.
      if (offsets.length === 3)
        expect(horizontal > 3 && horizontal < 5 && vertical > 3 && vertical < 5).toBe(false);
      else expect(horizontal > 4 && vertical > 4).toBe(false);
    }
  } finally {
    fixture.destroy();
  }
});

it('2,500 clustered buildings retain certified unit areas with a distant outlier', async () => {
  const fixture = await createFixture('local-clustered-outlier');
  try {
    const side = 50;
    const clusterCount = side * side;
    const buildingCount = clusterCount + 1;
    const segments = new Float32Array(buildingCount * 16);
    const owners = new Uint32Array(buildingCount * 4);
    for (let building = 0; building < buildingCount; building++) {
      const horizontal = building === clusterCount ? 1000000 : (building % side) + 0.5;
      const vertical = building === clusterCount ? 1000000 : Math.floor(building / side) + 0.5;
      const corners = [
        [horizontal - 0.2, vertical - 0.2],
        [horizontal + 0.2, vertical - 0.2],
        [horizontal + 0.2, vertical + 0.2],
        [horizontal - 0.2, vertical + 0.2]
      ];
      for (let edge = 0; edge < 4; edge++) {
        segments.set([...corners[edge], ...corners[(edge + 1) % 4]], building * 16 + edge * 4);
        owners[building * 4 + edge] = building;
      }
    }
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input('walls', segments, 'float32x4', buildingCount * 4),
      segmentBuildingIds: fixture.input('owners', owners, 'uint32', buildingCount * 4),
      enclosure: fixture.input(
        'enclosure',
        new Float32Array([0, 0, side, 0, side, side, 0, side]),
        'float32x2',
        4
      ),
      buildingCount,
      sampleCapacity: buildingCount * 4,
      sampleSpacing: 1,
      candidateRadius: 2.75,
      vertexTolerance: 1 / 1024,
      maximumSamplesPerBuilding: 4
    });
    const status = publish(fixture, 'status', result.outputs.buildingStatus);
    const positions = publish(fixture, 'positions', result.outputs.positions);
    const cells = publish(fixture, 'cells', result.outputs.cells);
    fixture.run();
    const flags = await fixture.readUint32(status, 4 * buildingCount);
    for (let building = 0; building < clusterCount; building++)
      expect(flags.slice(4 * building, 4 * building + 4), `building ${building}`).toEqual([
        1, 0, 0, 0
      ]);
    const areas = getBuildingAreas(
      await fixture.readFloat32(positions, 2 * result.outputs.positions.length),
      await fixture.readUint32(cells, 4 * result.outputs.cells.length),
      12,
      buildingCount
    );
    for (let building = 0; building < clusterCount; building++)
      expect(areas[building], `building ${building}`).toBeCloseTo(1, 4);
    // The outlier cannot certify a cell across the whole distant study domain at this radius.
    expect(flags.slice(-4)).toEqual([0, 0, 1, 0]);
  } finally {
    fixture.destroy();
  }
}, 120000);

it.each([-1, 0, 1])('reference sampling saturates at 2^32 %+i requested samples', async delta => {
  const fixture = await createFixture('reference-sample-prefix-overflow');
  try {
    const segmentCount = delta === 1 ? 257 : 256;
    const segments = new Float32Array(segmentCount * 4);
    for (let segment = 0; segment < segmentCount; segment++)
      segments.set(
        [0, 0, segment === 256 ? 1 : segment === 255 && delta === -1 ? 16777215 : 16777216, 0],
        4 * segment
      );
    const result = addBuildingTessellationRecipe(fixture.graph, {
      mode: 'reference',
      segments: fixture.input('walls', segments, 'float32x4', segmentCount),
      segmentBuildingIds: fixture.input(
        'owners',
        new Uint32Array(segmentCount),
        'uint32',
        segmentCount
      ),
      enclosure: fixture.input(
        'enclosure',
        new Float32Array([0, 0, 2, 0, 2, 2, 0, 2]),
        'float32x2',
        4
      ),
      buildingCount: 1,
      sampleSpacing: 1,
      sampleCapacity: 128,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16}
    });
    const samples = publish(fixture, 'samples', result.intermediates.samples);
    const status = publish(fixture, 'sampling', result.status.sampling);
    fixture.run();
    expect(await fixture.readUint32(status, 4)).toEqual([128, 0xffffffff, 1, 0]);
    expect(await fixture.readFloat32(samples, 256)).toEqual(
      Array.from({length: 256}, (_, component) => (component % 2 === 0 ? component / 2 : 0))
    );
  } finally {
    fixture.destroy();
  }
});

it('rank-grid and interval queries match a CPU radius oracle with tied coordinates, signed zeros and unused capacity', async () => {
  const fixture = await createFixture('adaptive-tessellation-index');
  try {
    const points = [
      [-0, -0],
      [0, 0],
      [0, 1],
      [0, -1],
      [1, 0],
      [-1, 0],
      [2, 2],
      [-2, -2],
      [1000000, 1000000],
      [-1000000, -1000000],
      [0.5, 0.5],
      [0.5, -0.5]
    ];
    const values = new Float32Array(32);
    values.set(points.flat());
    // Unused rows deliberately lie in the query circle and must not be indexed.
    const positions = fixture.input('points', values, 'float32x2', 16);
    const count = fixture.input(
      'count',
      new Uint32Array([points.length, points.length, 0, 0]),
      'uint32x4',
      1
    );
    const pointIndex = addBuildingTessellationIndex(fixture.graph, {id: 'index', positions, count});
    const queries = [
      [0, 0, 1],
      [0, 0, 0],
      [0.5, 0, 0.5],
      [1000000, 1000000, 1],
      [5, 5, 0.125],
      [-0.5, -0.5, 0.75]
    ];
    const queryRows = fixture.input(
      'queries',
      Float32Array.from(queries.flatMap(query => [...query, 0])),
      'float32x4',
      queries.length
    );
    const output = fixture.output('query-results', 'uint32x4', queries.length);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'query',
        operation: 'TessellationIndexOracle',
        invocationCount: queries.length,
        bindings: [
          {name: 'queries', view: queryRows, type: 'f32', access: 'read'},
          {name: 'pointIndex', view: pointIndex.data, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'output', view: output.view, type: 'u32', access: 'read_write'}
        ],
        declarations:
          getBuildingTessellationIndexDeclarations(pointIndex) +
          `
fn acceptCandidate(candidate:u32,center:vec2<f32>,radius:f32)->bool{
 let point=vec2<f32>(positions[positionsOffset+2u*candidate],positions[positionsOffset+2u*candidate+1u]);
 return distance(point,center)<=radius;
}
fn queryGrid(center:vec2<f32>,searchRadius:f32)->vec2<u32>{
 var count=0u;var bits=0u;
 ${getBuildingTessellationGridWalk(pointIndex, 'if(acceptCandidate(candidate,center,searchRadius)){count++;bits|=1u<<candidate;}', '')}
 return vec2<u32>(bits,count);
}
fn queryInterval(center:vec2<f32>,radius:f32)->vec2<u32>{
 let interval=getCandidateInterval(center,radius);var count=0u;var bits=0u;
 for(var slot=interval.x;slot<interval.y;slot++){
  let candidate=pointIndex[pointIndexOffset+interval.z+slot];
  if(acceptCandidate(candidate,center,radius)){count++;bits|=1u<<candidate;}
 }
 return vec2<u32>(bits,count);
}
`,
        body: `let base=queriesOffset+4u*index;let center=vec2<f32>(queries[base],queries[base+1u]);let radius=queries[base+2u];
let gridResult=queryGrid(center,radius);let intervalResult=queryInterval(center,radius);
output[outputOffset+4u*index]=gridResult.x;output[outputOffset+4u*index+1u]=intervalResult.x;
output[outputOffset+4u*index+2u]=gridResult.y;output[outputOffset+4u*index+3u]=intervalResult.y;`
      })
    );
    fixture.run();
    const actual = await fixture.readUint32(output, queries.length * 4);
    for (const [queryIndex, [horizontal, vertical, radius]] of queries.entries()) {
      let bits = 0;
      let count = 0;
      points.forEach(([pointHorizontal, pointVertical], point) => {
        if (
          Math.fround(Math.hypot(pointHorizontal - horizontal, pointVertical - vertical)) <= radius
        ) {
          bits |= 1 << point;
          count++;
        }
      });
      expect(actual.slice(4 * queryIndex, 4 * queryIndex + 4)).toEqual([bits, bits, count, count]);
    }
  } finally {
    fixture.destroy();
  }
});

it('simple enclosure winding, degenerate holes and contents may change on repeated encoding', async () => {
  const fixture = await createFixture('local-enclosure-reencoding');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const enclosureBuffer = fixture.parameterBuffer('mutable-enclosure', 'float32', 16);
    const enclosure = fixture.graph.createDataView(
      enclosureBuffer.importToGraph(fixture.graph).buffer,
      {format: 'float32x2', length: 8}
    );
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input(
        'walls',
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
      enclosure,
      enclosureMode: 'simple',
      enclosureRingOffsets: fixture.input('rings', new Uint32Array([0, 4, 8]), 'uint32', 3),
      buildingCount: 2,
      sampleSpacing: 1,
      sampleCapacity: 8,
      candidateRadius: 32,
      vertexTolerance: 1 / 1024,
      maximumSamplesPerBuilding: 4
    });
    const positions = publish(fixture, 'positions', result.outputs.positions);
    const cells = publish(fixture, 'cells', result.outputs.cells);
    const status = publish(fixture, 'status', result.outputs.buildingStatus);
    compiled = fixture.graph.compile();
    const shell = [
      [0, 0],
      [8, 0],
      [8, 8],
      [0, 8]
    ];
    const hole = [
      [3, 3],
      [5, 3],
      [5, 5],
      [3, 5]
    ];
    const cases = [
      {shell, hole, areas: [27, 33]},
      {shell: [...shell].reverse(), hole: [...hole].reverse(), areas: [27, 33]},
      {
        shell,
        hole: [
          [3, 3],
          [3, 3],
          [3, 3],
          [3, 3]
        ],
        areas: [28, 36]
      },
      {shell: [...shell].reverse(), hole, areas: [27, 33]}
    ];
    for (const testCase of cases) {
      enclosureBuffer.write(Float32Array.from([...testCase.shell, ...testCase.hole].flat()));
      submitGraph(fixture.device, compiled, undefined);
      expect(await fixture.readUint32(status, 8)).toEqual([1, 0, 0, 0, 1, 0, 0, 0]);
      const areas = getBuildingAreas(
        await fixture.readFloat32(positions, 2 * result.outputs.positions.length),
        await fixture.readUint32(cells, 4 * result.outputs.cells.length),
        12,
        2
      );
      areas.forEach((area, building) => expect(area).toBeCloseTo(testCase.areas[building], 4));
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it.each([
  'horizontal',
  'vertical'
] as const)('production rank-grid queries fall back to the %s interval at the visit-budget boundary', async axis => {
  const fixture = await createFixture(`production-fallback-${axis}`);
  try {
    // Each coordinate-rank bin contains four rows. The queried cell holds the two true
    // candidates plus one distant false positive: grid visits=3, shorter interval visits=2.
    const points = [
      [0, 0],
      [0.9, 0],
      [100, 0],
      ...Array.from({length: 13}, (_, index) => [10 + index, 10 + index])
    ].map(point => (axis === 'horizontal' ? point : [point[1], point[0]]));
    const values = Float32Array.from(points.flat());
    const positions = fixture.input('samples', values, 'float32x2', 16);
    const sampleData = new Uint32Array(48);
    sampleData.set(new Uint32Array(values.buffer));
    sampleData.fill(1, 32);
    sampleData[32] = 0;
    const buildingIndex = new Uint32Array(23);
    buildingIndex[0] = 16;
    buildingIndex.set([0, 1, 16], 4);
    buildingIndex.set(
      Array.from({length: 16}, (_, row) => row),
      7
    );
    const pointIndex = addBuildingTessellationIndex(fixture.graph, {
      id: 'point-index',
      positions,
      count: fixture.input('count', new Uint32Array([16, 16, 0, 0]), 'uint32x4', 1)
    });
    const parameters = fixture.input(
      'parameters',
      new Float32Array([1, 1, 1 / 1024]),
      'float32',
      3
    );
    const enclosure = fixture.input(
      'enclosure',
      new Float32Array([-0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25]),
      'float32x2',
      4
    );
    const samples = fixture.input('sample-data', sampleData, 'uint32', 48);
    const owners = fixture.input('building-index', buildingIndex, 'uint32', 23);
    const outputs = [1, 2, 3].map(maximumGridVisits => {
      const outputPositions = fixture.output(`positions-${maximumGridVisits}`, 'float32x2', 16);
      const cells = fixture.output(`cells-${maximumGridVisits}`, 'uint32x4', 2);
      const status = fixture.output(`status-${maximumGridVisits}`, 'uint32x4', 2);
      fixture.graph.add(
        getBuildingTessellationCellKernel(fixture.graph, {
          id: `query-${maximumGridVisits}`,
          buildingCount: 2,
          maximumSamplesPerBuilding: 1,
          candidateCapacity: 2,
          cellVertexCapacity: 8,
          maximumGridVisits,
          parameters,
          enclosure,
          pointIndex,
          simpleEnclosure: false,
          enclosurePieceCount: 1,
          sampleData: samples,
          buildingIndex: owners,
          positions: outputPositions.view,
          cells: cells.view,
          buildingStatus: status.view
        })
      );
      return {maximumGridVisits, positions: outputPositions, cells, status};
    });
    fixture.run();
    const oracle = points.filter(([horizontal, vertical]) => Math.hypot(horizontal, vertical) <= 1);
    expect(oracle).toHaveLength(2);
    for (const output of outputs) {
      const status = await fixture.readUint32(output.status, 8);
      expect(status.slice(0, 4)).toEqual(
        output.maximumGridVisits === 1 ? [0, 0, 1, 0] : [1, 0, 0, 0]
      );
      const areas = getBuildingAreas(
        await fixture.readFloat32(output.positions, 32),
        await fixture.readUint32(output.cells, 8),
        8,
        2
      );
      // The other site is 0.9 away, so its bisector at 0.45 cannot cut this 0.5-square.
      expect(areas[0]).toBeCloseTo(output.maximumGridVisits === 1 ? 0 : 0.25, 6);
    }
  } finally {
    fixture.destroy();
  }
});
