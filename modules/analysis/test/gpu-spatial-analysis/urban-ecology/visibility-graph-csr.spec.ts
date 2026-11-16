// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUVisibilityGraphCSRParameterValues,
  addVisibilityGraphRecipe as addSpatialVisibilityGraphRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {addVisibilityGraphRecipe} from '@luma-contributors/analysis/gpu-network';
import type {GPUVisibilityGraphCSRProps} from '@luma-contributors/analysis/gpu-network';
import {
  addUrbanPointGrid,
  getUrbanPointGridDeclarations
} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-point-grid';
import {getVisibilityCSRCountDeclarations} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr-query';
import {IS_FINITE_VALUE_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {publishVisibilityGraphOutput as publish} from './visibility-graph-test-utils';

function capture(fixture: RecipeTestFixture, name: string, result: ReturnType<typeof addCSR>) {
  const {status, ...fields} = result.outputs;
  return Object.fromEntries(
    Object.entries({...fields, ...status}).map(([field, view]) => [
      field,
      publish(fixture, `${name}-${field}`, view)
    ])
  );
}
function addCSR(fixture: RecipeTestFixture, props: GPUVisibilityGraphCSRProps) {
  return addVisibilityGraphRecipe(fixture.graph, props);
}
async function read(fixture: RecipeTestFixture, captured: ReturnType<typeof capture>) {
  const values: Record<string, number[]> = {};
  for (const [name, {output, source, components}] of Object.entries(captured))
    values[name] = source.format.startsWith('float')
      ? await fixture.readFloat32(output, source.length * components)
      : await fixture.readUint32(output, source.length * components);
  return values;
}
async function fixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}
function rows(values: Record<string, number[]>) {
  return values['offsets']
    .slice(0, -1)
    .map((offset, row) =>
      values['neighbors']
        .slice(offset, values['offsets'][row + 1])
        .sort((first, second) => first - second)
    );
}

it('CSR reproduces exact path degree, closeness and undirected betweenness', async () => {
  const test = await fixture('csr-path');
  try {
    const result = addCSR(test, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 8,
      pivotCount: 3,
      maximumIterations: 8,
      queryChunkSize: 64
    });
    const captured = capture(test, 'path', result);
    test.run();
    const values = await read(test, captured);
    expect(rows(values)).toEqual([[1], [0, 2], [1]]);
    expect(values['degree']).toEqual([1, 2, 1]);
    expect(values['reachedCounts']).toEqual([3, 3, 3]);
    [2 / 3, 1, 2 / 3].forEach((expected, row) =>
      expect(values['closeness'][row]).toBeCloseTo(expected, 5)
    );
    [0, 1, 0].forEach((expected, row) =>
      expect(values['betweenness'][row]).toBeCloseTo(expected, 5)
    );
    for (const [field, expected] of Object.entries({
      count: 4,
      requiredCount: 4,
      overflow: 0,
      candidateOverflow: 0,
      invalidCount: 0,
      approximation: 0,
      converged: 1,
      iterationLimitReached: 0,
      zeroWeightCycle: 0
    }))
      expect(values[field], field).toEqual([expected]);
    expect(values['weights'].slice(0, 4)).toEqual([1, 1, 1, 1]);
    expect(values['weights'].slice(4).every(Number.isNaN)).toBe(true);
  } finally {
    test.destroy();
  }
});

it.each([
  {edgeCapacity: 2, maximumGridVisits: 8192, overflow: 1, candidateOverflow: 0},
  {edgeCapacity: 8, maximumGridVisits: 1, overflow: 0, candidateOverflow: 1}
])('CSR failures suppress asymmetric graphs and misleading metrics: %j', async settings => {
  const test = await fixture('csr-failed');
  try {
    const result = addCSR(test, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      pivotCount: 3,
      maximumIterations: 8,
      ...settings
    });
    const captured = capture(test, 'failed', result);
    test.run();
    const values = await read(test, captured);
    expect(values['count']).toEqual([0]);
    expect(values['overflow']).toEqual([settings.overflow]);
    expect(values['candidateOverflow']).toEqual([settings.candidateOverflow]);
    expect(values['requiredCount']).toEqual([settings.candidateOverflow ? 0xffffffff : 4]);
    expect(values['offsets']).toEqual([0, 0, 0, 0]);
    expect(values['neighbors'].every(value => value === 0xffffffff)).toBe(true);
    expect(values['betweenness'].every(Number.isNaN)).toBe(true);
    expect(values['closeness'].every(Number.isNaN)).toBe(true);
    expect(values['converged']).toEqual([0]);
  } finally {
    test.destroy();
  }
});

it.each([
  1, 255, 256
])('CSR and dense outputs agree at exact and adjacent float32 radius boundaries for %i viewpoints', async count => {
  const test = await fixture(`csr-boundary-${count}`);
  try {
    const positions = new Float32Array(count * 2);
    for (let row = 0; row < count; row++) {
      positions[2 * row] = 100 + row * 4;
      positions[2 * row + 1] = 100;
    }
    positions[0] = 0;
    positions[1] = 0;
    if (count > 1) {
      positions[2] = 1;
      positions[3] = 0;
      positions[4] = 0;
      positions[5] = new Float32Array(new Uint32Array([0x3f800001]).buffer)[0];
      positions[6] = -new Float32Array(new Uint32Array([0x3f7fffff]).buffer)[0];
      positions[7] = 0;
    }
    const points = test.input('points', positions, 'float32x2', count),
      segments = test.input(
        'walls',
        new Float32Array([10000, 10000, 10001, 10000]),
        'float32x4',
        1
      );
    const csr = addCSR(test, {
      id: 'csr',
      mode: 'csr',
      positions: points,
      segments,
      maximumDistance: 1,
      edgeCapacity: 16,
      pivotCount: Math.min(count, 4),
      maximumIterations: 8,
      queryChunkSize: 128
    });
    const dense = addVisibilityGraphRecipe(test.graph, {
      id: 'dense',
      mode: 'reference',
      positions: points,
      segments,
      maximumDistance: 1,
      edgeCapacity: 8,
      searchStructure: 'brute-force',
      obstacleIndex: 'bvh'
    });
    const captured = capture(test, 'csr-read', csr);
    const denseFields = Object.fromEntries(
      Object.entries({...dense.outputs, status: dense.status}).map(([name, view]) => [
        name,
        publish(test, `dense-${name}`, view)
      ])
    );
    test.run();
    const values = await read(test, captured);
    const reference = await read(test, denseFields);
    const adjacency = reference['adjacency'];
    const neighborRows = rows(values);
    neighborRows.forEach((neighbors, row) => {
      const expected = Array.from({length: count}, (_, other) => other).filter(
        other => adjacency[row * count + other] === 1
      );
      expect(neighbors).toEqual(expected);
      expect(values['degree'][row]).toBe(reference['metrics'][4 * row]);
    });
    expect(values['viewpointClassifications']).toEqual(reference['viewpointClassifications']);
    expect(values['count']).toEqual([count === 1 ? 0 : 4]);
    expect(neighborRows[0]).toEqual(count === 1 ? [] : [1, 3]);
    // Every dense output is also checked by the grid/reference boundary fixture below; CSR's
    // centrality has sampled output semantics, so compare exact matching pivots separately.
    expect(values['closeness'][0]).toBeCloseTo(count === 1 ? 0 : 1, 5);
    expect(values['reachedCounts'][0]).toBe(count === 1 ? 1 : 3);
    expect(values['overflow']).toEqual([0]);
    expect(values['candidateOverflow']).toEqual([0]);
    expect(values['invalidCount']).toEqual([0]);
    expect(values['converged']).toEqual([1]);
  } finally {
    test.destroy();
  }
}, 120000);

it('10k viewpoints emit 39,600 directed city edges with sampled network metrics and a dense subset reference', async () => {
  const test = await fixture('csr-city');
  try {
    const positions = new Float32Array(20000);
    for (let row = 0; row < 10000; row++) {
      positions[2 * row] = row % 100;
      positions[2 * row + 1] = Math.floor(row / 100);
    }
    const segments = test.input(
      'walls',
      new Float32Array([-5, -5, -4, -5, 200, 200, 201, 200]),
      'float32x4',
      2
    );
    const city = addCSR(test, {
      id: 'city',
      mode: 'csr',
      positions: test.input('city-points', positions, 'float32x2', 10000),
      segments,
      maximumDistance: 1.01,
      edgeCapacity: 40000,
      pivotNodes: test.input('pivots', new Uint32Array([0, 99, 9900, 9999]), 'uint32', 4),
      laneCount: 4,
      maximumIterations: 256,
      queryChunkSize: 1024
    });
    const subsetPositions = test.input(
      'subset-points',
      new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
      'float32x2',
      4
    );
    const subset = addCSR(test, {
      id: 'subset',
      mode: 'csr',
      positions: subsetPositions,
      segments,
      maximumDistance: 1.01,
      edgeCapacity: 8,
      pivotCount: 4,
      maximumIterations: 8
    });
    const dense = addVisibilityGraphRecipe(test.graph, {
      id: 'reference',
      mode: 'reference',
      positions: subsetPositions,
      segments,
      maximumDistance: 1.01,
      edgeCapacity: 4,
      searchStructure: 'brute-force',
      obstacleIndex: 'bvh'
    });
    const captured = capture(test, 'city-read', city),
      small = capture(test, 'subset-read', subset);
    const denseAdjacency = publish(test, 'dense-adjacency', dense.outputs.adjacency),
      denseMetrics = publish(test, 'dense-metrics', dense.outputs.metrics),
      denseChoice = publish(test, 'dense-choice', dense.outputs.choice);
    test.run();
    const values = await read(test, captured),
      smallValues = await read(test, small);
    const adjacency = await test.readUint32(denseAdjacency.output, 16),
      metrics = await test.readFloat32(denseMetrics.output, 16),
      choice = await test.readFloat32(denseChoice.output, 4);
    rows(smallValues).forEach((neighbors, row) => {
      expect(neighbors).toEqual([0, 1, 2, 3].filter(other => adjacency[row * 4 + other] === 1));
      expect(smallValues['degree'][row]).toBe(metrics[4 * row]);
      expect(smallValues['closeness'][row]).toBeCloseTo(metrics[4 * row + 3], 5);
      expect(smallValues['betweenness'][row]).toBeCloseTo(choice[row], 5);
    });
    expect(values['count']).toEqual([39600]);
    expect(values['requiredCount']).toEqual([39600]);
    const neighborRows = rows(values);
    neighborRows.forEach((neighbors, row) => {
      const column = row % 100,
        vertical = Math.floor(row / 100);
      const expected = [
        ...(column > 0 ? [row - 1] : []),
        ...(column < 99 ? [row + 1] : []),
        ...(vertical > 0 ? [row - 100] : []),
        ...(vertical < 99 ? [row + 100] : [])
      ].sort((first, second) => first - second);
      expect(neighbors).toEqual(expected);
      expect(values['degree'][row]).toBe(expected.length);
    });
    values['closeness'].forEach(value => expect(value).toBeCloseTo(9999 / 990000, 6));
    expect(values['reachedCounts']).toEqual([10000, 10000, 10000, 10000]);
    expect(values['betweenness'].every(Number.isFinite)).toBe(true);
    for (const [field, expected] of Object.entries({
      overflow: 0,
      candidateOverflow: 0,
      invalidCount: 0,
      approximation: 1,
      converged: 1,
      iterationLimitReached: 0,
      zeroWeightCycle: 0
    }))
      expect(values[field], field).toEqual([expected]);
    expect(values['viewpointClassifications'].every(value => value === 0)).toBe(true);
  } finally {
    test.destroy();
  }
}, 120000);

it('one compiled CSR graph recovers after changed radius and input data overflow its edge capacity', async () => {
  const test = await fixture('csr-reuse');
  let compiled: ReturnType<typeof test.graph.compile> | undefined;
  try {
    const mutablePoints = test.parameterBuffer('mutable-points', 'float32', 6);
    const pointView = test.graph.createDataView(mutablePoints.importToGraph(test.graph).buffer, {
      format: 'float32x2',
      length: 3
    });
    const parameters = test.parameterBuffer('parameters', 'float32', 2);
    const result = addCSR(test, {
      mode: 'csr',
      positions: pointView,
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      parameters: parameters.importToGraph(test.graph),
      edgeCapacity: 4,
      pivotCount: 3,
      maximumIterations: 8
    });
    const captured = capture(test, 'reuse', result);
    compiled = test.graph.compile();
    for (const radius of [1.1, 3, 1.1]) {
      mutablePoints.write(
        new Float32Array(radius === 3 ? [0, 0, 1, 0.25, 2, 0] : [0, 0, 1, 0, 2, 0])
      );
      parameters.write(getGPUVisibilityGraphCSRParameterValues({maximumDistance: radius}));
      submitGraph(test.device, compiled, undefined);
      const values = await read(test, captured);
      expect(values['count']).toEqual([radius === 3 ? 0 : 4]);
      expect(values['requiredCount']).toEqual([radius === 3 ? 6 : 4]);
      expect(values['overflow']).toEqual([Number(radius === 3)]);
      expect(values['candidateOverflow']).toEqual([0]);
      expect(values['invalidCount']).toEqual([0]);
      expect(rows(values)).toEqual(radius === 3 ? [[], [], []] : [[1], [0, 2], [1]]);
      expect(values['betweenness'][1]).toBe(radius === 3 ? Number.NaN : 1);
    }
  } finally {
    compiled?.destroy();
    test.destroy();
  }
});

it('one pivot on a four-node path produces independently calculated sampled betweenness', async () => {
  const test = await fixture('csr-partial-pivots');
  try {
    const result = addCSR(test, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]), 'float32x2', 4),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.125,
      edgeCapacity: 8,
      pivotNodes: test.input('pivots', new Uint32Array([0]), 'uint32', 1),
      maximumIterations: 8
    });
    const captured = capture(test, 'sampled', result);
    test.run();
    const values = await read(test, captured);
    // From source 0, internal nodes 1 and 2 serve 2 and 1 downstream destinations.
    // Undirected sampling scale is (1/2) * (4 nodes / 1 pivot) = 2.
    [0, 4, 2, 0].forEach((expected, row) =>
      expect(values['betweenness'][row]).toBeCloseTo(expected, 5)
    );
    expect(values['degree']).toEqual([1, 2, 2, 1]);
    expect(rows(values)).toEqual([[1], [0, 2], [1, 3], [2]]);
    expect(values['closeness']).toEqual([0.5]);
    expect(values['reachedCounts']).toEqual([4]);
    for (const [field, expected] of Object.entries({
      count: 6,
      requiredCount: 6,
      approximation: 1,
      converged: 1,
      iterationLimitReached: 0,
      zeroWeightCycle: 0,
      overflow: 0,
      candidateOverflow: 0,
      invalidCount: 0
    }))
      expect(values[field], field).toEqual([expected]);
  } finally {
    test.destroy();
  }
});

it('a short iteration budget publishes partial long-path metrics and stopping flags', async () => {
  const test = await fixture('csr-short-budget');
  try {
    const count = 64;
    const points = Float32Array.from(Array.from({length: count}, (_, row) => [row, 0]).flat());
    const result = addCSR(test, {
      mode: 'csr',
      positions: test.input('points', points, 'float32x2', count),
      segments: test.input('walls', new Float32Array([200, 200, 201, 200]), 'float32x4', 1),
      maximumDistance: 1.125,
      edgeCapacity: 2 * (count - 1),
      pivotNodes: test.input('pivots', new Uint32Array([0]), 'uint32', 1),
      maximumIterations: 2,
      localIterations: 1
    });
    const captured = capture(test, 'partial', result);
    test.run();
    const values = await read(test, captured);
    expect(values['count']).toEqual([126]);
    expect(values['requiredCount']).toEqual([126]);
    expect(values['degree']).toEqual(
      Array.from({length: count}, (_, row) => (row === 0 || row === count - 1 ? 1 : 2))
    );
    expect(values['converged']).toEqual([0]);
    expect(values['iterationLimitReached']).toEqual([1]);
    expect(values['reachedCounts']).toEqual([3]);
    // Full endpoint closeness is (n-1)/(n*(n-1)/2) = 2/n. A stopped search reaches a proper prefix.
    expect(values['closeness'][0]).toBeCloseTo(2 / 3, 5);
    expect(values['closeness'][0]).toBeGreaterThan(2 / count);
    expect(values['closeness'][0]).toBeLessThanOrEqual(1);
    expect(values['betweenness'].every(Number.isFinite)).toBe(true);
    for (const field of ['overflow', 'candidateOverflow', 'invalidCount', 'zeroWeightCycle'])
      expect(values[field], field).toEqual([0]);
  } finally {
    test.destroy();
  }
});

it.each([
  1, 8192
])('the count query reads at most %i rows of a populated 256×256 grid', async maximumGridVisits => {
  const test = await fixture(`csr-grid-visits-${maximumGridVisits}`);
  try {
    const points = new Float32Array(65536 * 2);
    for (let row = 0; row < 65536; row++) {
      points[2 * row] = row % 256;
      points[2 * row + 1] = Math.floor(row / 256);
    }
    const positions = test.input('points', points, 'float32x2', 65536);
    const grid = addUrbanPointGrid(test.graph, {id: 'grid', positions});
    const output = test.output('query-record', 'uint32', 3);
    // One query exercises the production count function over the full index, keeping this
    // work-bound regression cheap. All 65,536 rows lie inside the domain-wide radius.
    test.graph.add(
      createWGSLKernelNode(test.graph, {
        id: 'count-visits',
        operation: 'visibility-grid-visit-bound',
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'pointGrid', view: grid.data, type: 'u32', access: 'read'},
          {name: 'output', view: output.view, type: 'u32', access: 'read_write'}
        ],
        declarations:
          IS_FINITE_VALUE_WGSL +
          getUrbanPointGridDeclarations(grid) +
          `
var<private> radiusValue:f32;
fn readPoint(row:u32)->vec2<f32>{return vec2<f32>(positions[positionsOffset+2u*row],positions[positionsOffset+2u*row+1u]);}
fn isVisible(first:u32,second:u32)->bool{return first!=second;}
` +
          getVisibilityCSRCountDeclarations(grid, maximumGridVisits),
        body: 'radiusValue=512.0;let record=countVisibleNeighbors(0u);output[outputOffset]=record.x;output[outputOffset+1u]=record.y;output[outputOffset+2u]=record.z;'
      })
    );
    test.run();
    expect(await test.readUint32(output, 3)).toEqual([maximumGridVisits - 1, 1, maximumGridVisits]);
  } finally {
    test.destroy();
  }
}, 120000);

it('769 identical viewpoints agree for packed and 128/256-row queries across success → overflow → success', async () => {
  const test = await fixture('csr-769-query-chunks');
  let compiled: ReturnType<typeof test.graph.compile> | undefined;
  try {
    const count = 769;
    const points = Float32Array.from(
      Array.from({length: count}, (_, row) => [4 * Math.floor(row / 2) + (row % 2), 0]).flat()
    );
    const positions = test.input('points', points, 'float32x2', count),
      segments = test.input(
        'walls',
        new Float32Array([10000, 10000, 10001, 10000]),
        'float32x4',
        1
      );
    const parameters = test.parameterBuffer('parameters', 'float32', 2);
    const parametersView = parameters.importToGraph(test.graph);
    const configurations = [undefined, 128, 256].map((queryChunkSize, route) => {
      const result = addCSR(test, {
        id: `route-${route}`,
        mode: 'csr',
        positions,
        segments,
        maximumDistance: 1.125,
        edgeCapacity: 768,
        parameters: parametersView,
        pivotCount: 4,
        maximumIterations: 8,
        queryChunkSize
      });
      return capture(test, `read-${route}`, result);
    });
    compiled = test.graph.compile();
    for (const maximumDistance of [1.125, 4096, 1.125]) {
      parameters.write(getGPUVisibilityGraphCSRParameterValues({maximumDistance}));
      submitGraph(test.device, compiled, undefined);
      const values = await Promise.all(configurations.map(captured => read(test, captured)));
      expect(values[1]).toEqual(values[0]);
      expect(values[2]).toEqual(values[0]);
      const actual = values[0],
        overflow = maximumDistance === 4096;
      expect(actual['count']).toEqual([overflow ? 0 : 768]);
      expect(actual['requiredCount']).toEqual([overflow ? count * (count - 1) : 768]);
      expect(actual['overflow']).toEqual([Number(overflow)]);
      expect(actual['candidateOverflow']).toEqual([0]);
      expect(actual['invalidCount']).toEqual([0]);
      expect(actual['approximation']).toEqual([1]);
      expect(actual['converged']).toEqual([Number(!overflow)]);
      expect(actual['iterationLimitReached']).toEqual([0]);
      expect(actual['zeroWeightCycle']).toEqual([0]);
      expect(actual['viewpointClassifications']).toEqual(Array(count).fill(0));
      expect(actual['degree']).toEqual(
        Array.from({length: count}, (_, row) => Number(!overflow && row < count - 1))
      );
      expect(rows(actual)).toEqual(
        Array.from({length: count}, (_, row) =>
          overflow || row === count - 1 ? [] : [row % 2 === 0 ? row + 1 : row - 1]
        )
      );
      expect(actual['reachedCounts']).toEqual(Array(4).fill(overflow ? 0 : 2));
      expect(actual['closeness']).toEqual(Array(4).fill(overflow ? Number.NaN : 1));
      expect(actual['betweenness']).toEqual(Array(count).fill(overflow ? Number.NaN : 0));
      expect(actual['weights']).toEqual(Array(768).fill(overflow ? Number.NaN : 1));
      expect(actual['gridVisits'].every(visited => visited <= 8192)).toBe(true);
      if (overflow) expect(actual['gridVisits']).toEqual(Array(count).fill(count));
    }
  } finally {
    compiled?.destroy();
    test.destroy();
  }
}, 120000);

it.each([
  0, 65536
])('canonical CSR sightlines match both dense directions near segment tolerances at translation %i', async translation => {
  const tolerance = 1 / 64;
  for (const [name, segment, visible] of [
    ['near-endpoint-block', [4 - 2 * tolerance, -1, 4 - 2 * tolerance, 1], false],
    ['near-endpoint-clear', [4 + 2 * tolerance, -1, 4 + 2 * tolerance, 1], true],
    ['near-collinear-block', [1, tolerance / 2, 3, tolerance / 2], false],
    ['near-collinear-clear', [1, 2 * tolerance, 3, 2 * tolerance], true]
  ] as const) {
    const test = await fixture(`csr-${name}-${translation}`);
    try {
      const positions = test.input(
        'points',
        new Float32Array([translation, translation, translation + 4, translation]),
        'float32x2',
        2
      );
      const segments = test.input(
        'wall',
        Float32Array.from(segment, value => value + translation),
        'float32x4',
        1
      );
      const csr = addCSR(test, {
        id: 'csr',
        mode: 'csr',
        positions,
        segments,
        tolerance,
        maximumDistance: 5,
        edgeCapacity: 2,
        pivotCount: 2,
        maximumIterations: 8
      });
      const captured = capture(test, 'csr-read', csr);
      const denseReadbacks = (['bvh', 'brute-force'] as const).map(obstacleIndex => {
        const dense = addVisibilityGraphRecipe(test.graph, {
          id: `dense-${obstacleIndex}`,
          mode: 'reference',
          positions,
          segments,
          tolerance,
          maximumDistance: 5,
          edgeCapacity: 1,
          searchStructure: 'brute-force',
          obstacleIndex
        });
        return publish(test, `dense-read-${obstacleIndex}`, dense.outputs.adjacency);
      });
      test.run();
      const actual = await read(test, captured);
      for (const dense of denseReadbacks)
        expect(await test.readUint32(dense.output, 4), name).toEqual([
          0,
          Number(visible),
          Number(visible),
          0
        ]);
      expect(rows(actual), name).toEqual(visible ? [[1], [0]] : [[], []]);
      expect(actual['degree']).toEqual([Number(visible), Number(visible)]);
      expect(actual['count']).toEqual([visible ? 2 : 0]);
      expect(actual['viewpointClassifications']).toEqual([0, 0]);
      expect(actual['invalidCount']).toEqual([0]);
      expect(actual['converged']).toEqual([1]);
      expect(actual['betweenness']).toEqual([0, 0]);
      expect(actual['closeness']).toEqual(visible ? [1, 1] : [0, 0]);
      expect(actual['reachedCounts']).toEqual(visible ? [2, 2] : [1, 1]);
    } finally {
      test.destroy();
    }
  }
}, 120000);

it('runtime non-finite radii suppress CSR and publish NaN sentinels, then recover in the same graph', async () => {
  const test = await fixture('csr-runtime-invalid-radii');
  let compiled: ReturnType<typeof test.graph.compile> | undefined;
  try {
    const parameters = test.parameterBuffer('runtime-radius', 'float32', 2);
    const result = addCSR(test, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      parameters: parameters.importToGraph(test.graph),
      edgeCapacity: 8,
      pivotCount: 3,
      maximumIterations: 8
    });
    const captured = capture(test, 'runtime-invalid', result);
    compiled = test.graph.compile();
    for (const radius of [1.1, Number.NaN, Number.POSITIVE_INFINITY, 1.1]) {
      parameters.write(getGPUVisibilityGraphCSRParameterValues({maximumDistance: radius}));
      submitGraph(test.device, compiled, undefined);
      const values = await read(test, captured);
      const valid = Number.isFinite(radius);
      expect(values['viewpointClassifications']).toEqual([0, 0, 0]);
      expect(values['invalidCount']).toEqual([valid ? 0 : 3]);
      expect(values['overflow']).toEqual([0]);
      expect(values['candidateOverflow']).toEqual([0]);
      expect(values['count']).toEqual([valid ? 4 : 0]);
      expect(values['requiredCount']).toEqual([valid ? 4 : 0]);
      expect(values['offsets']).toEqual(valid ? [0, 1, 3, 4] : [0, 0, 0, 0]);
      expect(rows(values)).toEqual(valid ? [[1], [0, 2], [1]] : [[], [], []]);
      expect(values['degree']).toEqual(valid ? [1, 2, 1] : [0, 0, 0]);
      expect(values['reachedCounts']).toEqual(valid ? [3, 3, 3] : [0, 0, 0]);
      expect(values['converged']).toEqual([Number(valid)]);
      if (valid) {
        expect(values['weights'].slice(0, 4)).toEqual([1, 1, 1, 1]);
        expect(values['weights'].slice(4).every(Number.isNaN)).toBe(true);
        expect(values['betweenness']).toEqual([0, 1, 0]);
        [2 / 3, 1, 2 / 3].forEach((expected, row) =>
          expect(values['closeness'][row]).toBeCloseTo(expected, 5)
        );
      } else {
        expect(values['neighbors']).toEqual(Array(8).fill(0xffffffff));
        expect(values['weights'].every(Number.isNaN)).toBe(true);
        expect(values['betweenness'].every(Number.isNaN)).toBe(true);
        expect(values['closeness'].every(Number.isNaN)).toBe(true);
      }
    }
  } finally {
    compiled?.destroy();
    test.destroy();
  }
});

it('the adjacency-only spatial entry publishes the independently known three-point path', async () => {
  const test = await fixture('spatial-csr-path');
  try {
    const result = addSpatialVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 8
    });
    const fields = {...result.outputs, ...result.status};
    const captured = Object.fromEntries(
      Object.entries(fields)
        .filter(([name]) => name !== 'status')
        .map(([name, view]) => [name, publish(test, `spatial-${name}`, view as GraphDataView)])
    );
    test.run();
    const values = await read(test, captured);
    expect(values['offsets']).toEqual([0, 1, 3, 4]);
    expect(rows(values)).toEqual([[1], [0, 2], [1]]);
    expect(values['degree']).toEqual([1, 2, 1]);
    expect(values['weights'].slice(0, 4)).toEqual([1, 1, 1, 1]);
    expect(values['weights'].slice(4).every(Number.isNaN)).toBe(true);
    expect(values['neighbors'].slice(4)).toEqual(Array(4).fill(0xffffffff));
    for (const [name, expected] of Object.entries({
      count: 4,
      requiredCount: 4,
      overflow: 0,
      candidateOverflow: 0,
      invalidCount: 0
    }))
      expect(values[name], name).toEqual([expected]);
  } finally {
    test.destroy();
  }
});

it('dense recipe output names coexist with independently checked path readbacks', async () => {
  const test = await fixture('dense-readback-path');
  try {
    const result = addVisibilityGraphRecipe(test.graph, {
      id: 'dense',
      mode: 'reference',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 8,
      searchStructure: 'brute-force',
      obstacleIndex: 'bvh'
    });
    const captured = Object.fromEntries(
      Object.entries({...result.outputs, status: result.status}).map(([name, view]) => [
        name,
        publish(test, `dense-${name}`, view)
      ])
    );
    test.run();
    const values = await read(test, captured);
    expect(values['adjacency']).toEqual([0, 1, 0, 1, 0, 1, 0, 1, 0]);
    expect(values['status']).toEqual([2, 2, 0, 0]);
    expect(values['edges']).toEqual([0, 1, 1, 2, ...Array(12).fill(0xffffffff)]);
    expect(values['choice']).toEqual([0, 1, 0]);
    expect(values['viewpointClassifications']).toEqual([0, 0, 0]);
    const expectedMetrics = [1, 2, 3, 2 / 3, 2, 2, 2, 1, 1, 2, 3, 2 / 3];
    values['metrics'].forEach((value, index) =>
      expect(value).toBeCloseTo(expectedMetrics[index], 5)
    );
  } finally {
    test.destroy();
  }
});
