// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialKMeans} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {runExample} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-k-means.example';
import {STATISTICS_CLUSTERING_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/statistics-clustering';
import {expectKMeansMajorityDistances} from './kmeans-centering-assertions';

const correctDistances = [
  ...new Array<number>(254).fill(0.000015378700499807765),
  0.9921722414455979
];

test.each([
  1, 127, 253
])('S1: K-means qualification rejects a finite wrong majority distance at row %s', row => {
  const distances = [...correctDistances];
  distances[row] = 1;
  expect(() => expectKMeansMajorityDistances(distances)).toThrow();
});

test('S1: K-means qualification checks the full vector length and accepts the independent fixture', () => {
  expect(() => expectKMeansMajorityDistances(correctDistances.slice(0, 254))).toThrow();
  expect(() => expectKMeansMajorityDistances(correctDistances)).not.toThrow();
});

test('FLT-04: the actual K-means example builds the centered graph', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<undefined>(device);
  const add = vi.spyOn(graph, 'add');
  const compute = vi.spyOn(graph, 'addComputePass');
  try {
    await runExample(graph, {
      observations: createTransientView(graph, 'points', 'float32x2', 255, Buffer.STORAGE),
      parameters: createTransientView(graph, 'parameters', 'uint32', 4, Buffer.STORAGE),
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'}
    });
    const contributor = add.mock.calls
      .map(([node]) => node)
      .find(node => node instanceof GPUSpatialKMeans);
    expect(contributor).toBeInstanceOf(GPUSpatialKMeans);
    if (!(contributor instanceof GPUSpatialKMeans)) throw new Error('Missing K-means contributor');
    expect(contributor.centerCoordinates).toBe(true);
    // The centered route publishes absolute centers by adding each cluster's origin back.
    expect(
      compute.mock.calls.find(([node]) => node.id === `${contributor.id}-publish-centers`)?.[0]
        .workload?.variant
    ).toBe('publish-centers-absolute');
  } finally {
    add.mockRestore();
    compute.mockRestore();
    device.destroy();
  }
});

test('FLT-04: formula and receipt state centered accumulation and the historical opt-out', () => {
  const entry = STATISTICS_CLUSTERING_OPERATORS['GPUSpatialKMeans'];
  expect(entry.choices.find(choice => choice.name === 'centerCoordinates')).toMatchObject({
    alternatives: [true, false]
  });
  if ('formula' in entry.formula) throw new Error('K-means requires a curated formula');
  expect(entry.formula.ascii).toContain('With centerCoordinates: true (explicit in this scenario)');
  expect(entry.formula.ascii).toContain('explicit false');
  expect(entry.formula.tex).toContain('x_i-o');
  const scenario = readFileSync(
    new URL(
      '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-k-means.lens.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(scenario).toContain('centerCoordinates: true');
});
