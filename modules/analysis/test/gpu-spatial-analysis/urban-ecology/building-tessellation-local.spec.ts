// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUReduction} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {
  getGPUBuildingTessellationLocalParameterValues,
  type GPUBuildingTessellationLocalProps
} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-local';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology-advanced.json';
import {publishBuildingTessellationOutput as publish} from './building-tessellation-test-utils';
import {
  createClusteredTessellationInput,
  getClusteredTessellationOracle
} from './building-tessellation-cluster-test-utils';

type Point = [number, number];
function getArea(polygon: Point[]): number {
  if (polygon.length < 3) return 0;
  const anchor = polygon[0];
  let twice = 0;
  for (let vertex = 1; vertex + 1 < polygon.length; vertex++) {
    const first = polygon[vertex];
    const second = polygon[vertex + 1];
    twice +=
      (first[0] - anchor[0]) * (second[1] - anchor[1]) -
      (first[1] - anchor[1]) * (second[0] - anchor[0]);
  }
  return 0.5 * twice;
}
/** Independent separating-axis oracle: convex interiors cannot overlap across a separating projection. */
function hasInteriorOverlap(first: Point[], second: Point[], tolerance: number): boolean {
  if (
    Math.abs(getArea(first)) < tolerance * tolerance ||
    Math.abs(getArea(second)) < tolerance * tolerance
  )
    return false;
  for (const polygon of [first, second]) {
    for (let edge = 0; edge < polygon.length; edge++) {
      const start = polygon[edge];
      const end = polygon[(edge + 1) % polygon.length];
      const horizontal = end[0] - start[0];
      const vertical = end[1] - start[1];
      const length = Math.hypot(horizontal, vertical);
      if (length === 0) continue;
      const project = (point: Point) =>
        ((point[0] - start[0]) * -vertical + (point[1] - start[1]) * horizontal) / length;
      const firstValues = first.map(project);
      const secondValues = second.map(project);
      const overlap =
        Math.min(Math.max(...firstValues), Math.max(...secondValues)) -
        Math.max(Math.min(...firstValues), Math.min(...secondValues));
      if (overlap <= tolerance) return false;
    }
  }
  return true;
}

function getPieces(
  positions: number[],
  records: number[],
  cellVertexCapacity: number
): {source: number; points: Point[]}[] {
  const pieces: {source: number; points: Point[]}[] = [];
  for (let piece = 0; piece < records.length / 4; piece++) {
    const count = records[4 * piece];
    if (count === 0) continue;
    expect(count).toBeLessThanOrEqual(cellVertexCapacity);
    const points: Point[] = [];
    for (let vertex = 0; vertex < count; vertex++) {
      const offset = 2 * (piece * cellVertexCapacity + vertex);
      points.push([positions[offset], positions[offset + 1]]);
    }
    pieces.push({source: records[4 * piece + 2], points});
  }
  return pieces;
}
async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}
function createSmallProps(fixture: RecipeTestFixture): GPUBuildingTessellationLocalProps {
  return {
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
      new Float32Array([0, 0, 8, 0, 8, 8, 0, 8]),
      'float32x2',
      4
    ),
    buildingCount: 2,
    sampleSpacing: 1,
    sampleCapacity: 8,
    candidateRadius: 32,
    vertexTolerance: 1 / 1024,
    maximumSamplesPerBuilding: 8,
    cellVertexCapacity: 12
  };
}
function createStatusOutputs(fixture: RecipeTestFixture, name: string) {
  return Object.fromEntries(
    ['count', 'requiredCount', 'overflow', 'candidateOverflow', 'invalidCount'].map(field => [
      field,
      fixture.output(`${name}-${field}`, 'uint32', 1)
    ])
  ) as Record<
    'count' | 'requiredCount' | 'overflow' | 'candidateOverflow' | 'invalidCount',
    ReturnType<RecipeTestFixture['output']>
  >;
}

it('local sample-cell unions match the explicit reference and closed-form building areas', async () => {
  const fixture = await createFixture('local-tessellation-reference');
  try {
    const props = createSmallProps(fixture);
    const local = addBuildingTessellationRecipe(fixture.graph, {...props, id: 'local'});
    const exact = addBuildingTessellationRecipe(fixture.graph, {
      id: 'reference',
      mode: 'reference',
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      buildingCount: 2,
      enclosure: props.enclosure,
      sampleSpacing: 1,
      sampleCapacity: 8,
      vertexTolerance: 1 / 1024,
      searchStructure: 'brute-force',
      capacity: {intersections: 256, nodedSegments: 256, vertices: 128, rings: 32}
    });
    const positions = publish(fixture, 'local-positions', local.outputs.positions);
    const cells = publish(fixture, 'local-records', local.outputs.cells);
    const status = publish(fixture, 'local-status', local.outputs.buildingStatus);
    const complete = publish(fixture, 'reference-complete', exact.status.complete);
    const referenceOutputs = exact.outputs.map((output, building) => ({
      count: publish(fixture, `reference-count-${building}`, output.status.count),
      features: publish(fixture, `reference-features-${building}`, output.geometry.featureOffsets),
      polygons: publish(fixture, `reference-polygons-${building}`, output.geometry.polygonOffsets),
      rings: publish(fixture, `reference-rings-${building}`, output.geometry.ringOffsets),
      positions: publish(fixture, `reference-positions-${building}`, output.geometry.positions)
    }));
    fixture.run();
    expect(await fixture.readUint32(status, 8)).toEqual([1, 0, 0, 0, 1, 0, 0, 0]);
    expect(await fixture.readUint32(complete, 2)).toEqual([1, 1]);
    const pieces = getPieces(
      await fixture.readFloat32(positions, local.outputs.positions.length * 2),
      await fixture.readUint32(cells, local.outputs.cells.length * 4),
      12
    );
    const areas = [0, 0];
    for (const piece of pieces) {
      areas[piece.source] += getArea(piece.points);
      // Pointwise oracle for this fixture: the two reference regions are rectangles separated at x=3.5.
      for (const [horizontal, vertical] of piece.points) {
        expect(horizontal).toBeGreaterThanOrEqual((piece.source === 0 ? 0 : 3.5) - 1 / 1024);
        expect(horizontal).toBeLessThanOrEqual((piece.source === 0 ? 3.5 : 8) + 1 / 1024);
        expect(vertical).toBeGreaterThanOrEqual(-1 / 1024);
        expect(vertical).toBeLessThanOrEqual(8 + 1 / 1024);
      }
    }
    // Congruent rectangles [1,2] and [5,6] have an exact vertical bisector at x=3.5.
    expect(areas[0]).toBeCloseTo(28, 4);
    expect(areas[1]).toBeCloseTo(36, 4);
    for (const [building, output] of referenceOutputs.entries()) {
      const [count] = await fixture.readUint32(output.count, 1);
      const features = await fixture.readUint32(output.features, count + 1);
      const polygons = await fixture.readUint32(output.polygons, features[count] + 1);
      const ringCount = polygons.at(-1)!;
      const rings = await fixture.readUint32(output.rings, ringCount + 1);
      const coordinates = await fixture.readFloat32(output.positions, 2 * rings[ringCount]);
      let area = 0;
      for (let ring = 0; ring < ringCount; ring++) {
        const points: Point[] = [];
        for (let vertex = rings[ring]; vertex < rings[ring + 1]; vertex++)
          points.push([coordinates[2 * vertex], coordinates[2 * vertex + 1]]);
        area += getArea(points);
      }
      expect(Math.abs(areas[building] - area)).toBeLessThanOrEqual(0.01);
    }
    for (let first = 0; first < pieces.length; first++)
      for (let second = first + 1; second < pieces.length; second++)
        expect(hasInteriorOverlap(pieces[first].points, pieces[second].points, 1 / 1024)).toBe(
          false
        );
  } finally {
    fixture.destroy();
  }
});

it.each([
  {name: 'candidate capacity', overrides: {candidateCapacity: 4}},
  {name: 'own sample capacity', overrides: {maximumSamplesPerBuilding: 3}},
  {name: 'radius certification', overrides: {candidateRadius: 1}},
  {
    name: 'uncertified cells with a small output',
    overrides: {candidateRadius: 1, cellVertexCapacity: 3}
  },
  {name: 'grid visits', overrides: {maximumGridVisits: 1}},
  {name: 'global sampling capacity', overrides: {sampleCapacity: 4}}
])('$name reports candidateOverflow without output overflow or partial polygons', async ({
  overrides
}) => {
  const fixture = await createFixture('local-tessellation-candidate-overflow');
  try {
    const props = {...createSmallProps(fixture), ...overrides};
    const outputStatus = createStatusOutputs(fixture, 'status');
    const result = addBuildingTessellationRecipe(fixture.graph, {
      ...props,
      output: {
        status: Object.fromEntries(
          Object.entries(outputStatus).map(([name, output]) => [name, output.view])
        )
      }
    });
    const cells = publish(fixture, 'cells', result.outputs.cells);
    fixture.run();
    expect(await fixture.readUint32(outputStatus.count, 1)).toEqual([0]);
    expect(await fixture.readUint32(outputStatus.requiredCount, 1)).toEqual([2]);
    expect(await fixture.readUint32(outputStatus.overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(outputStatus.candidateOverflow, 1)).toEqual([1]);
    expect(await fixture.readUint32(outputStatus.invalidCount, 1)).toEqual([0]);
    const records = await fixture.readUint32(cells, result.outputs.cells.length * 4);
    expect(records.filter((_, index) => index % 4 === 0).every(count => count === 0)).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('a final vertex capacity shortfall preserves vertex requirements and sets only overflow', async () => {
  const fixture = await createFixture('local-tessellation-output-overflow');
  try {
    const result = addBuildingTessellationRecipe(fixture.graph, {
      ...createSmallProps(fixture),
      cellVertexCapacity: 3
    });
    const status = publish(fixture, 'status', result.outputs.buildingStatus);
    const cells = publish(fixture, 'cells', result.outputs.cells);
    fixture.run();
    expect(await fixture.readUint32(status, 8)).toEqual([0, 1, 0, 0, 0, 1, 0, 0]);
    const records = await fixture.readUint32(cells, result.outputs.cells.length * 4);
    expect(records.filter((_, index) => index % 4 === 0).every(count => count === 0)).toBe(true);
    expect(
      records.filter((_, index) => index % 4 === 1).some(count => count > 3 && count !== 0xffffffff)
    ).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('a nonconvex study ring is invalid on the local route and remains available to the reference', async () => {
  const fixture = await createFixture('local-tessellation-nonconvex');
  try {
    const props = createSmallProps(fixture);
    const enclosure = fixture.input(
      'nonconvex',
      Float32Array.from(reference.tessellation.enclosure),
      'float32x2',
      6
    );
    const result = addBuildingTessellationRecipe(fixture.graph, {...props, enclosure});
    const status = publish(fixture, 'status', result.outputs.buildingStatus);
    fixture.run();
    expect(await fixture.readUint32(status, 8)).toEqual([0, 0, 0, 1, 0, 0, 0, 1]);
  } finally {
    fixture.destroy();
  }
});

it('spacing, radius and input contents change success → overflow → success in one compiled graph', async () => {
  const fixture = await createFixture('local-tessellation-parameters');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const props = createSmallProps(fixture);
    const walls = fixture.parameterBuffer('mutable-walls', 'float32', 32);
    walls.write(Float32Array.from(reference.tessellation.segments));
    props.segments = fixture.graph.createDataView(walls.importToGraph(fixture.graph).buffer, {
      format: 'float32x4',
      length: 8
    });
    const parameters = fixture.parameterBuffer('parameters', 'float32', 3);
    parameters.write(getGPUBuildingTessellationLocalParameterValues(props));
    const buildingStatus = fixture.output('building-status', 'uint32x4', 2);
    addBuildingTessellationRecipe(fixture.graph, {
      ...props,
      parameters: parameters.importToGraph(fixture.graph),
      output: {buildingStatus: buildingStatus.view}
    });
    compiled = fixture.graph.compile();
    for (const radius of [32, 1, 32]) {
      const shifted = Float32Array.from(reference.tessellation.segments);
      if (radius === 1)
        for (let component = 0; component < shifted.length; component++)
          shifted[component] += 0.125;
      walls.write(shifted);
      parameters.write(
        getGPUBuildingTessellationLocalParameterValues({
          ...props,
          candidateRadius: radius,
          sampleSpacing: radius === 1 ? 0.125 : 1
        })
      );
      submitGraph(fixture.device, compiled, undefined);
      expect(await fixture.readUint32(buildingStatus, 8)).toEqual(
        radius === 1 ? [0, 0, 1, 0, 0, 0, 1, 0] : [1, 0, 0, 0, 1, 0, 0, 0]
      );
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('100k jittered rectangular buildings partition the study area without sampled cell overlaps', async () => {
  const fixture = await createFixture('local-tessellation-city');
  try {
    const columns = 500;
    const rows = 200;
    const buildingCount = columns * rows;
    const segments = new Float32Array(buildingCount * 16);
    const owners = new Uint32Array(buildingCount * 4);
    let state = 17;
    const next = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
    for (let building = 0; building < buildingCount; building++) {
      const centerX = (building % columns) + 0.5 + 0.08 * (next() - 0.5);
      const centerY = Math.floor(building / columns) + 0.5 + 0.08 * (next() - 0.5);
      const halfWidth = 0.18 + 0.04 * next();
      const halfHeight = 0.18 + 0.04 * next();
      const corners = [
        [centerX - halfWidth, centerY - halfHeight],
        [centerX + halfWidth, centerY - halfHeight],
        [centerX + halfWidth, centerY + halfHeight],
        [centerX - halfWidth, centerY + halfHeight]
      ];
      for (let edge = 0; edge < 4; edge++) {
        const start = corners[edge];
        const end = corners[(edge + 1) % 4];
        segments.set([...start, ...end], building * 16 + edge * 4);
        owners[building * 4 + edge] = building;
      }
    }
    const outputStatus = createStatusOutputs(fixture, 'city-status');
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input('city-walls', segments, 'float32x4', buildingCount * 4),
      segmentBuildingIds: fixture.input('city-owners', owners, 'uint32', buildingCount * 4),
      enclosure: fixture.input(
        'study',
        new Float32Array([0, 0, columns, 0, columns, rows, 0, rows]),
        'float32x2',
        4
      ),
      buildingCount,
      sampleSpacing: 1,
      sampleCapacity: buildingCount * 4,
      candidateRadius: 2.75,
      vertexTolerance: 1 / 1024,
      output: {
        status: Object.fromEntries(
          Object.entries(outputStatus).map(([name, output]) => [name, output.view])
        )
      }
    });
    const maximumSamplesPerBuilding = result.provenance.maximumSamplesPerBuilding;
    const cellVertexCapacity = result.provenance.cellVertexCapacity;
    const buildingAreas = fixture.output('building-areas', 'float32', buildingCount);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'measure-city-area',
        operation: 'tessellation-city-area',
        invocationCount: buildingCount,
        bindings: [
          {name: 'positions', view: result.outputs.positions, type: 'f32', access: 'read'},
          {name: 'cells', view: result.outputs.cells, type: 'u32', access: 'read'},
          {name: 'areas', view: buildingAreas.view, type: 'f32', access: 'read_write'}
        ],
        body: `var total=0.0;
for(var part=0u;part<${maximumSamplesPerBuilding}u;part++){
 let piece=index*${maximumSamplesPerBuilding}u+part;let count=cells[cellsOffset+4u*piece];
 let base=positionsOffset+2u*piece*${cellVertexCapacity}u;
 let anchor=vec2<f32>(positions[base],positions[base+1u]);
 for(var vertex=1u;vertex+1u<count;vertex++){
  let first=vec2<f32>(positions[base+2u*vertex],positions[base+2u*vertex+1u])-anchor;
  let second=vec2<f32>(positions[base+2u*(vertex+1u)],positions[base+2u*(vertex+1u)+1u])-anchor;
  total+=0.5*(first.x*second.y-first.y*second.x);
 }
}
areas[areasOffset+index]=total;`
      })
    );
    const totalArea = fixture.output('total-area', 'float32', 1);
    fixture.graph.add(
      new GPUReduction({
        id: 'total-area',
        input: buildingAreas.view,
        output: totalArea.view,
        operation: 'sum'
      }).getCommandNodes(fixture.graph)
    );
    const selected = new Set<number>();
    for (const building of [
      0,
      columns - 1,
      buildingCount - columns,
      buildingCount - 1,
      ...Array.from({length: 32}, () => Math.floor(next() * buildingCount))
    ]) {
      const column = building % columns;
      const row = Math.floor(building / columns);
      for (let vertical = Math.max(0, row - 1); vertical <= Math.min(rows - 1, row + 1); vertical++)
        for (
          let horizontal = Math.max(0, column - 1);
          horizontal <= Math.min(columns - 1, column + 1);
          horizontal++
        )
          selected.add(vertical * columns + horizontal);
    }
    const selectedRows = [...selected].sort((first, second) => first - second);
    const selection = fixture.input(
      'selected-buildings',
      Uint32Array.from(selectedRows),
      'uint32',
      selectedRows.length
    );
    const sampledCells = fixture.output(
      'sampled-cells',
      'uint32x4',
      selectedRows.length * maximumSamplesPerBuilding
    );
    const sampledPositions = fixture.output(
      'sampled-positions',
      'float32x2',
      selectedRows.length * maximumSamplesPerBuilding * cellVertexCapacity
    );
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'sample-city-cells',
        operation: 'tessellation-city-sample',
        invocationCount: selectedRows.length * maximumSamplesPerBuilding,
        bindings: [
          {name: 'selection', view: selection, type: 'u32', access: 'read'},
          {name: 'cells', view: result.outputs.cells, type: 'u32', access: 'read'},
          {name: 'positions', view: result.outputs.positions, type: 'f32', access: 'read'},
          {name: 'sampledCells', view: sampledCells.view, type: 'u32', access: 'read_write'},
          {name: 'sampledPositions', view: sampledPositions.view, type: 'f32', access: 'read_write'}
        ],
        body: `let building=selection[selectionOffset+index/${maximumSamplesPerBuilding}u];
let source=building*${maximumSamplesPerBuilding}u+index%${maximumSamplesPerBuilding}u;
for(var word=0u;word<4u;word++){sampledCells[sampledCellsOffset+4u*index+word]=cells[cellsOffset+4u*source+word];}
for(var vertex=0u;vertex<${cellVertexCapacity * 2}u;vertex++){sampledPositions[sampledPositionsOffset+index*${cellVertexCapacity * 2}u+vertex]=positions[positionsOffset+source*${cellVertexCapacity * 2}u+vertex];}`
      })
    );
    fixture.run();
    expect(await fixture.readUint32(outputStatus.count, 1)).toEqual([buildingCount]);
    expect(await fixture.readUint32(outputStatus.requiredCount, 1)).toEqual([buildingCount]);
    expect(await fixture.readUint32(outputStatus.overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(outputStatus.candidateOverflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(outputStatus.invalidCount, 1)).toEqual([0]);
    const [area] = await fixture.readFloat32(totalArea, 1);
    expect(Math.abs(area - columns * rows)).toBeLessThanOrEqual(columns * rows * 1e-4);
    const areas = await fixture.readFloat32(buildingAreas, buildingCount);
    expect(areas.every(value => value > 0.7 && value < 1.3)).toBe(true);
    const pieces = getPieces(
      await fixture.readFloat32(sampledPositions, sampledPositions.view.length * 2),
      await fixture.readUint32(sampledCells, sampledCells.view.length * 4),
      cellVertexCapacity
    );
    expect(new Set(pieces.map(piece => piece.source)).size).toBe(selectedRows.length);
    for (const piece of pieces)
      for (const [horizontal, vertical] of piece.points) {
        expect(horizontal).toBeGreaterThanOrEqual(-1 / 1024);
        expect(horizontal).toBeLessThanOrEqual(columns + 1 / 1024);
        expect(vertical).toBeGreaterThanOrEqual(-1 / 1024);
        expect(vertical).toBeLessThanOrEqual(rows + 1 / 1024);
      }
    // Compare adjacent and same-building pieces; distant grid rows cannot be neighbours in this fixture.
    for (let first = 0; first < pieces.length; first++)
      for (let second = first + 1; second < pieces.length; second++) {
        const firstBuilding = pieces[first].source;
        const secondBuilding = pieces[second].source;
        if (
          Math.abs((firstBuilding % columns) - (secondBuilding % columns)) > 1 ||
          Math.abs(Math.floor(firstBuilding / columns) - Math.floor(secondBuilding / columns)) > 1
        )
          continue;
        expect(
          hasInteriorOverlap(pieces[first].points, pieces[second].points, 4 / 1024),
          `${firstBuilding} / ${secondBuilding}`
        ).toBe(false);
      }
  } finally {
    fixture.destroy();
  }
}, 120000);

it('finite extreme coordinates either preserve the bisector or explicitly invalidate geometry', async () => {
  const fixture = await createFixture('local-extreme-arithmetic');
  try {
    const segments: number[] = [];
    for (const [lowerX, upperX, lowerY, upperY] of [
      [0, 1, 1, 1.5],
      [1e19, 1e19 + 1e13, 0, 0.5]
    ]) {
      const corners = [
        [lowerX, lowerY],
        [upperX, lowerY],
        [upperX, upperY],
        [lowerX, upperY]
      ];
      corners.forEach((corner, edge) => segments.push(...corner, ...corners[(edge + 1) % 4]));
    }
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input('walls', Float32Array.from(segments), 'float32x4', 8),
      segmentBuildingIds: fixture.input(
        'owners',
        new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1]),
        'uint32',
        8
      ),
      buildingCount: 2,
      enclosure: fixture.input(
        'study',
        new Float32Array([-2e19, -1, 2e19, -1, 2e19, 2, -2e19, 2]),
        'float32x2',
        4
      ),
      sampleSpacing: 1e20,
      sampleCapacity: 8,
      candidateRadius: 1e21,
      vertexTolerance: 1 / 1024
    });
    const status = publish(fixture, 'extreme-status', result.outputs.buildingStatus);
    const positions = publish(fixture, 'extreme-positions', result.outputs.positions);
    const records = publish(fixture, 'extreme-records', result.outputs.cells);
    fixture.run();
    const flags = await fixture.readUint32(status, 8);
    const cells = await fixture.readUint32(records, result.outputs.cells.length * 4);
    const pieces = getPieces(
      await fixture.readFloat32(positions, result.outputs.positions.length * 2),
      cells,
      12
    );
    for (let building = 0; building < 2; building++) {
      if (flags[4 * building] === 0) {
        expect(flags.slice(4 * building, 4 * building + 4)).toEqual([0, 0, 0, 1]);
        expect(pieces.filter(piece => piece.source === building)).toHaveLength(0);
      } else {
        expect(flags.slice(4 * building, 4 * building + 4)).toEqual([1, 0, 0, 0]);
        const owned = pieces.filter(piece => piece.source === building);
        const horizontal = owned.flatMap(piece => piece.points.map(point => point[0]));
        expect(
          Math.abs((building === 0 ? Math.max(...horizontal) : Math.min(...horizontal)) - 5e18)
        ).toBeLessThan(2e13);
        expect(owned.reduce((area, piece) => area + getArea(piece.points), 0)).toBeCloseTo(
          building === 0 ? 7.5e19 : 4.5e19,
          -15
        );
      }
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  10000, 100000
])('clustered %i buildings retain certified cells and flag uncertified boundary/outlier rows', async buildingCount => {
  const fixture = await createFixture(`local-cluster-${buildingCount}`);
  try {
    const {segments, owners} = createClusteredTessellationInput(buildingCount);
    const oracle = getClusteredTessellationOracle(buildingCount);
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input('walls', segments, 'float32x4', buildingCount * 4),
      segmentBuildingIds: fixture.input('owners', owners, 'uint32', buildingCount * 4),
      buildingCount,
      enclosure: fixture.input(
        'study',
        new Float32Array([0, 0, 1000001, 0, 1000001, 1000001, 0, 1000001]),
        'float32x2',
        4
      ),
      sampleSpacing: 1,
      sampleCapacity: buildingCount * 4,
      candidateRadius: 2.75,
      vertexTolerance: 1 / 1024
    });
    const statuses = Object.fromEntries(
      Object.entries(result.status)
        .filter(([name]) => name !== 'sampling')
        .map(([name, view]) => [name, publish(fixture, `status-${name}`, view)])
    );
    const rows = publish(fixture, 'rows', result.outputs.buildingStatus),
      cells = publish(fixture, 'cells', result.outputs.cells),
      sampling = publish(fixture, 'sampling', result.status.sampling);
    fixture.run();
    expect(await fixture.readUint32(sampling, 4)).toEqual([
      buildingCount * 4,
      buildingCount * 4,
      0,
      0
    ]);
    for (const [name, value] of Object.entries(oracle.status))
      expect(await fixture.readUint32(statuses[name], 1), name).toEqual([value]);
    const flags = await fixture.readUint32(rows, buildingCount * 4);
    const records = await fixture.readUint32(cells, result.outputs.cells.length * 4);
    const piecesPerBuilding = result.provenance.piecesPerBuilding;
    let firstIncorrectRow: unknown;
    let firstIncorrectPiece: unknown;
    for (let building = 0; building < buildingCount; building++) {
      const certified = oracle.isCertified(building);
      const row = flags.slice(building * 4, building * 4 + 4);
      const expected = certified ? [1, 0, 0, 0] : [0, 0, 1, 0];
      if (firstIncorrectRow === undefined && row.some((value, field) => value !== expected[field]))
        firstIncorrectRow = {building, row, expected};
      for (let part = 0; part < piecesPerBuilding; part++) {
        const record = 4 * (building * piecesPerBuilding + part);
        const vertexCount = records[record];
        // Failed rows still publish no geometry, including their first four sample slots.
        const correct =
          records[record + 2] === building &&
          (!certified || part >= 4
            ? vertexCount === 0
            : vertexCount >= 3 &&
              vertexCount <= 12 &&
              records[record + 1] === vertexCount &&
              records[record + 3] === 4 * building + part);
        if (!correct && firstIncorrectPiece === undefined)
          firstIncorrectPiece = {building, part, record: records.slice(record, record + 4)};
      }
    }
    expect(
      firstIncorrectRow,
      'every building has the independent expected certification status'
    ).toBeUndefined();
    expect(
      firstIncorrectPiece,
      'every certified piece is populated; every failed piece is empty'
    ).toBeUndefined();
  } finally {
    fixture.destroy();
  }
}, 120000);
