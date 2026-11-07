// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudVoxelMerge,
  GPUPointCloudVoxelDownsample,
  GPUPointCloudHeight,
  GPUPointCloudSurfaces,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudICP,
  GPUPointCloudChange
} from '@luma-contributors/graph/gpu-pointcloud';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import pdalOracle from './fixtures/pdal-pointcloud.json';
import changeOracle from './fixtures/change.json';
import {requireTestDevice} from '../utils/require-test-device';

it('owned rows with full halos reproduce monolithic neighborhoods and flag insufficient support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([-1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0])
    );
    const rows = fixture.view('uint32', new Uint32Array([1, 2, 0]));
    const indices = fixture.view('uint32', new Uint32Array(6));
    const distances = fixture.view('float32', new Float32Array(6));
    const counts = fixture.view('uint32', new Uint32Array(3));
    const status = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        queryRows: rows.view,
        supportBounds: {minimum: [-1, -2, -2], maximum: [2, 2, 2]},
        grid: {origin: [-1, -2, -2], dimensions: [4, 4, 4], cellSize: 1},
        radius: 1,
        capacity: 2,
        indices: indices.view,
        squaredDistances: distances.view,
        counts: counts.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(indices.buffer)).toEqual([0, 2, 1, 3, 1, 0xffffffff]);
    expect(await readIntegers(counts.buffer)).toEqual([2, 2, 1]);
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 4]);
    rows.buffer.write(new Uint32Array([3, 1, 99]));
    await fixture.run();
    expect((await readIntegers(status.buffer))[2] & 1).toBe(1);
  } finally {
    fixture.destroy();
  }
});

it('indexed voxel vector means exclude nonfinite whole vectors and preserve sampled source IDs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 1.2, 0, 0])
    );
    const values = fixture.view(
      'float32x3',
      new Float32Array([1, 2, 3, 3, 4, 5, NaN, 100, 200, 7, 8, 9])
    );
    const representatives = fixture.view('uint32', new Uint32Array(2));
    const counts = fixture.view('uint32', new Uint32Array(2));
    const output = fixture.view('float32x3', new Float32Array(6));
    const centroids = fixture.view('float32x3', new Float32Array(6));
    const means = fixture.view('float32x3', new Float32Array(6));
    const support = fixture.view('uint32', new Uint32Array(2));
    fixture.graph.add(
      new GPUPointCloudVoxelDownsample({
        positions: positions.view,
        origin: [0, 0, 0],
        gridSize: [2, 1, 1],
        voxelSize: 1,
        representatives: representatives.view,
        counts: counts.view,
        output: output.view,
        centroids: centroids.view,
        attribute: {values: values.view, means: means.view, counts: support.view}
      })
    );
    await fixture.run();
    expect(await readIntegers(representatives.buffer)).toEqual([0, 3]);
    expect(await readIntegers(support.buffer)).toEqual([2, 1]);
    expect(await readFloats(means.buffer)).toEqual([2, 3, 4, 7, 8, 9]);
    (await readFloats(centroids.buffer))
      .slice(0, 3)
      .forEach(value => expect(value).toBeCloseTo(0.3, 6));
  } finally {
    fixture.destroy();
  }
});

it('bilinear HAG reproduces a tilted plane while bounded IDW fills missing cells and preserves negative heights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([1, 1, 5, 0.5, 0.5, 0]));
    const terrain = fixture.view('float32', new Float32Array([1, 3, 3, 5]));
    const coverage = fixture.view('uint32', new Uint32Array([1, 1, 1, 1]));
    const heights = fixture.view('float32', new Float32Array(2));
    const status = fixture.view('uint32', new Uint32Array(2));
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: positions.view,
        terrain: terrain.view,
        coverage: coverage.view,
        grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2},
        interpolation: {method: 'bilinear'},
        heights: heights.view,
        status: status.view
      })
    );
    const filled = fixture.view('float32', new Float32Array(2));
    const filledStatus = fixture.view('uint32', new Uint32Array(2));
    fixture.graph.add(
      new GPUPointCloudHeight({
        id: 'idw-height',
        positions: positions.view,
        terrain: terrain.view,
        coverage: coverage.view,
        grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2},
        interpolation: {method: 'idw', radius: 2, power: 2, minimumSupport: 2},
        heights: filled.view,
        status: filledStatus.view
      })
    );
    await fixture.run();
    expect(await readFloats(heights.buffer)).toEqual([2, -1]);
    expect(await readIntegers(status.buffer)).toEqual([0, 0]);
    coverage.buffer.write(new Uint32Array([1, 1, 1, 0]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([2, 0]);
    expect((await readFloats(filled.buffer))[0]).toBeCloseTo(5 - 7 / 3, 5);
    expect((await readFloats(filled.buffer))[1]).toBe(-1);
  } finally {
    fixture.destroy();
  }
});

it('indexed robust ICP merges multiple moment blocks and rejects Tukey outliers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const values = Array.from({length: 600}, (_, index) => [
      index % 10,
      Math.floor(index / 10) % 10,
      Math.floor(index / 100)
    ]).flat();
    const shifted = values.map((value, index) => value + [0.05, -0.03, 0.02][index % 3]);
    values.push(30, 30, 30);
    const source = fixture.view('float32x3', new Float32Array(values));
    const target = fixture.view('float32x3', new Float32Array(shifted));
    const transform = fixture.view('float32', new Float32Array(16));
    const history = fixture.view('float32x3', new Float32Array(12));
    const mask = fixture.view('uint32', new Uint32Array(601));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        targetGrid: {origin: [-1, -1, -1], cellSize: 1, dimensions: [12, 12, 8]},
        robust: {method: 'tukey', scale: 0.5},
        maximumDistance: 60,
        iterations: 4,
        transform: transform.view,
        history: history.view,
        inlierMask: mask.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    const actual = await readFloats(transform.buffer);
    [0.05, -0.03, 0.02].forEach((value, index) => expect(actual[12 + index]).toBeCloseTo(value, 4));
    expect((await readIntegers(mask.buffer)).at(-1)).toBe(0);
    expect((await readFloats(history.buffer))[0]).toBe(600);
  } finally {
    fixture.destroy();
  }
});

it('normal-projected registration covariance changes confidence by orientation without changing signed means', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const reference = fixture.view('float32x3', new Float32Array([0, 0, -0.1, 0, 0, 0, 0, 0, 0.1]));
    const comparison = fixture.view('float32x3', new Float32Array([0, 0, 0.9, 0, 0, 1, 0, 0, 1.1]));
    const cores = fixture.view('float32x3', new Float32Array([0, 0, 0, 0, 0, 0]));
    const normals = fixture.view('float32x3', new Float32Array([0, 0, 1, 0, 0, -1]));
    const changes = fixture.view('float32x3', new Float32Array(6));
    const roughness = fixture.view('float32x3', new Float32Array(6));
    const counts = fixture.view('uint32', new Uint32Array(4));
    const status = fixture.view('uint32', new Uint32Array(2));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: reference.view,
        comparison: comparison.view,
        cores: cores.view,
        normals: normals.view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 2,
        registrationCovariance: [0.01, 0, 0, 0.04, 0, 0.09],
        uncertaintyCombination: 'quadrature',
        changes: changes.view,
        roughness: roughness.view,
        counts: counts.view,
        status: status.view
      })
    );
    await fixture.run();
    const actual = await readFloats(changes.buffer);
    const detection = 1.96 * Math.sqrt(0.01 / 3 + 0.01 / 3 + 0.09);
    expect(actual[0]).toBeCloseTo(1, 6);
    expect(actual[3]).toBeCloseTo(-1, 6);
    expect(actual[2]).toBeCloseTo(1 + detection, 5);
    expect(actual[4]).toBeCloseTo(-1 - detection, 5);
    expect(await readIntegers(status.buffer)).toEqual([0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('point-to-plane ICP recovers a rigid transform on three oriented faces and refuses planar rank loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const targetValues: number[] = [];
    const normalValues: number[] = [];
    for (let axis = 0; axis < 3; axis++)
      for (const first of [-1, 1])
        for (const second of [-1, 1]) {
          const point = [first, second, axis + 2];
          const normal = [0, 0, 1];
          if (axis === 0) {
            point.reverse();
            normal.reverse();
          } else if (axis === 1) {
            [point[1], point[2]] = [point[2], point[1]];
            [normal[1], normal[2]] = [normal[2], normal[1]];
          }
          targetValues.push(...point);
          normalValues.push(...normal);
        }
    const angle = 0.02;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const translation = [0.03, -0.02, 0.01];
    const sourceValues: number[] = [];
    for (let point = 0; point < 12; point++) {
      const horizontal = targetValues[3 * point] - translation[0];
      const vertical = targetValues[3 * point + 1] - translation[1];
      sourceValues.push(
        cosine * horizontal + sine * vertical,
        -sine * horizontal + cosine * vertical,
        targetValues[3 * point + 2] - translation[2]
      );
    }
    const source = fixture.view('float32x3', new Float32Array(sourceValues));
    const target = fixture.view('float32x3', new Float32Array(targetValues));
    const normals = fixture.view('float32x3', new Float32Array(normalValues));
    const transform = fixture.view('float32', new Float32Array(16));
    const history = fixture.view('float32x3', new Float32Array(18));
    const mask = fixture.view('uint32', new Uint32Array(12));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        targetNormals: normals.view,
        metric: 'point-to-plane',
        maximumDistance: 0.5,
        iterations: 6,
        transform: transform.view,
        history: history.view,
        inlierMask: mask.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    const actual = await readFloats(transform.buffer);
    translation.forEach((value, index) => expect(actual[12 + index]).toBeCloseTo(value, 4));
    expect(actual[1]).toBeCloseTo(sine, 4);
    expect(actual[4]).toBeCloseTo(-sine, 4);
    normals.buffer.write(Float32Array.from({length: 36}, (_, index) => Number(index % 3 === 2)));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([3]);
  } finally {
    fixture.destroy();
  }
});

it('explicit disjoint voxel merges preserve global representatives and independently counted vector means', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const parts = [
      {
        counts: [2, 0],
        means: [0.2, 0.2, 0.2, 0, 0, 0],
        representatives: [0, 0xffffffff],
        offset: 0,
        attributes: [2, 3, 4, 0, 0, 0],
        support: [2, 0]
      },
      {
        counts: [1, 1],
        means: [0.5, 0.5, 0.5, 1.2, 0, 0],
        representatives: [0, 1],
        offset: 2,
        attributes: [0, 0, 0, 7, 8, 9],
        support: [0, 1]
      }
    ].map(part => ({
      counts: fixture.view('uint32', new Uint32Array(part.counts)).view,
      centroids: fixture.view('float32x3', new Float32Array(part.means)).view,
      representatives: fixture.view('uint32', new Uint32Array(part.representatives)).view,
      sourceRowOffset: part.offset,
      attribute: {
        means: fixture.view('float32x3', new Float32Array(part.attributes)).view,
        counts: fixture.view('uint32', new Uint32Array(part.support)).view
      }
    }));
    const counts = fixture.view('uint32', new Uint32Array(2));
    const centroids = fixture.view('float32x3', new Float32Array(6));
    const representatives = fixture.view('uint32', new Uint32Array(2));
    const means = fixture.view('float32x3', new Float32Array(6));
    const support = fixture.view('uint32', new Uint32Array(2));
    const status = fixture.view('uint32', new Uint32Array(2));
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        parts,
        partitionPolicy: 'disjoint-owned-rows',
        counts: counts.view,
        centroids: centroids.view,
        representatives: representatives.view,
        attribute: {means: means.view, counts: support.view},
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(counts.buffer)).toEqual([3, 1]);
    expect(await readIntegers(representatives.buffer)).toEqual([0, 3]);
    expect(await readFloats(means.buffer)).toEqual([2, 3, 4, 7, 8, 9]);
    expect(await readIntegers(support.buffer)).toEqual([2, 1]);
    expect(await readIntegers(status.buffer)).toEqual([0, 0]);
    (await readFloats(centroids.buffer))
      .slice(0, 3)
      .forEach(value => expect(value).toBeCloseTo(0.3, 6));
  } finally {
    fixture.destroy();
  }
});

it('correlated surface-change confidence matches independent SciPy pairwise covariance integration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const reference = fixture.view('float32x3', new Float32Array(changeOracle.reference));
    const comparison = fixture.view('float32x3', new Float32Array(changeOracle.comparison));
    const cores = fixture.view('float32x3', new Float32Array([0, 0, 0]));
    const normals = fixture.view('float32x3', new Float32Array([0, 0, 1]));
    const changes = fixture.view('float32x3', new Float32Array(3));
    const roughness = fixture.view('float32x3', new Float32Array(3));
    const counts = fixture.view('uint32', new Uint32Array(2));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: reference.view,
        comparison: comparison.view,
        cores: cores.view,
        normals: normals.view,
        alignment: 'already-aligned',
        radius: changeOracle.radius,
        halfDepth: changeOracle.halfDepth,
        samplingCovariance: {
          referenceVariance: changeOracle.referenceVariance,
          comparisonVariance: changeOracle.comparisonVariance,
          correlationLength: changeOracle.correlationLength
        },
        registrationCovariance: [0, 0, 0, 0, 0, changeOracle.registrationVariance],
        uncertaintyCombination: 'quadrature',
        changes: changes.view,
        roughness: roughness.view,
        counts: counts.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(counts.buffer)).toEqual(changeOracle.counts);
    const actual = await readFloats(changes.buffer);
    const actualRoughness = await readFloats(roughness.buffer);
    changeOracle.changes.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 5));
    changeOracle.roughness.forEach((value, index) =>
      expect(actualRoughness[index]).toBeCloseTo(value, 5)
    );
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  2, 4
] as const)('packed vec%i attributes retain every component through voxel reduction and merge', async components => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0, 0, 0, 0.2, 0.2, 0.2]));
    const values = fixture.view(
      'float32',
      new Float32Array(Array.from({length: components * 2}, (_, index) => index + 1)),
      16
    );
    const means = fixture.view('float32', new Float32Array(components), 16);
    const mergedMeans = fixture.view('float32', new Float32Array(components), 16);
    const format = components === 2 ? 'float32x2' : 'float32x4';
    const valuesView = fixture.graph.createDataView(values.view.buffer, {
      format,
      length: 2,
      byteOffset: 16
    });
    const meansView = fixture.graph.createDataView(means.view.buffer, {
      format,
      length: 1,
      byteOffset: 16
    });
    const mergedMeansView = fixture.graph.createDataView(mergedMeans.view.buffer, {
      format,
      length: 1,
      byteOffset: 16
    });
    const counts = fixture.view('uint32', new Uint32Array(1));
    const attributeCounts = fixture.view('uint32', new Uint32Array(1));
    const representatives = fixture.view('uint32', new Uint32Array(1));
    const centroids = fixture.view('float32x3', new Float32Array(3));
    const output = fixture.view('float32x3', new Float32Array(3));
    const mergedCounts = fixture.view('uint32', new Uint32Array(1));
    const mergedAttributeCounts = fixture.view('uint32', new Uint32Array(1));
    const mergedRepresentatives = fixture.view('uint32', new Uint32Array(1));
    const mergedCentroids = fixture.view('float32x3', new Float32Array(3));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudVoxelDownsample({
        positions: positions.view,
        origin: [0, 0, 0],
        gridSize: [1, 1, 1],
        voxelSize: 1,
        counts: counts.view,
        representatives: representatives.view,
        centroids: centroids.view,
        output: output.view,
        attribute: {values: valuesView, means: meansView, counts: attributeCounts.view}
      })
    );
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts: [
          {
            counts: counts.view,
            centroids: centroids.view,
            representatives: representatives.view,
            sourceRowOffset: 7,
            attribute: {means: meansView, counts: attributeCounts.view}
          }
        ],
        counts: mergedCounts.view,
        centroids: mergedCentroids.view,
        representatives: mergedRepresentatives.view,
        status: status.view,
        attribute: {means: mergedMeansView, counts: mergedAttributeCounts.view}
      })
    );
    await fixture.run();
    expect(await readFloats(mergedMeans.buffer, 16)).toEqual(
      Array.from({length: components}, (_, component) => component + 1 + components / 2)
    );
    expect(await readIntegers(mergedCounts.buffer)).toEqual([2]);
    expect(await readIntegers(mergedRepresentatives.buffer)).toEqual([7]);
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('block-merged statistical filtering matches independent population moments across partial blocks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const distances = Array.from({length: 777}, (_, index) =>
      index % 29 === 0 ? 30 : 1 + (index % 11) / 10
    );
    const support = distances.map((_, index) => (index % 17 === 0 ? 0 : 1));
    const squaredDistances = fixture.view(
      'float32',
      new Float32Array(distances.map(value => value * value))
    );
    const counts = fixture.view('uint32', new Uint32Array(support));
    const means = fixture.view('float32', new Float32Array(distances.length));
    const mask = fixture.view('uint32', new Uint32Array(distances.length));
    fixture.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: squaredDistances.view,
        counts: counts.view,
        capacity: 1,
        standardDeviations: 1.5,
        meanDistances: means.view,
        mask: mask.view
      })
    );
    const supported = distances.filter((_, index) => support[index]);
    const mean = supported.reduce((sum, value) => sum + value, 0) / supported.length;
    const variance =
      supported.reduce((sum, value) => sum + (value - mean) ** 2, 0) / supported.length;
    const threshold = mean + 1.5 * Math.sqrt(variance);
    await fixture.run();
    expect(await readIntegers(mask.buffer)).toEqual(
      distances.map((value, index) => Number(Boolean(support[index]) && value <= threshold))
    );
    counts.buffer.write(new Uint32Array(support.length));
    await fixture.run();
    expect(await readIntegers(mask.buffer)).toEqual(new Array(support.length).fill(0));
  } finally {
    fixture.destroy();
  }
});

it('outlier classification and coincident-cell height sampling agree with actual PDAL 2.8.4', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array(pdalOracle.outlier.positions.flat())
    );
    const indices = fixture.view('uint32', new Uint32Array(positions.view.length * 3));
    const distances = fixture.view('float32', new Float32Array(positions.view.length * 3));
    const counts = fixture.view('uint32', new Uint32Array(positions.view.length));
    const means = fixture.view('float32', new Float32Array(positions.view.length));
    const mask = fixture.view('uint32', new Uint32Array(positions.view.length));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: 100,
        capacity: 3,
        indices: indices.view,
        squaredDistances: distances.view,
        counts: counts.view
      })
    );
    fixture.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: distances.view,
        counts: counts.view,
        capacity: 3,
        standardDeviations: 2,
        meanDistances: means.view,
        mask: mask.view
      })
    );
    const heightPositions = fixture.view(
      'float32x3',
      new Float32Array(pdalOracle.height.positions.flat())
    );
    const classifications = fixture.view('uint32', new Uint32Array([2, 2, 1, 1]));
    const terrain = fixture.view('float32', new Float32Array(2));
    const surface = fixture.view('float32', new Float32Array(2));
    const canopy = fixture.view('float32', new Float32Array(2));
    const coverage = fixture.view('uint32', new Uint32Array(2));
    const heights = fixture.view('float32', new Float32Array(4));
    const status = fixture.view('uint32', new Uint32Array(4));
    const grid = {origin: [0, 0] as const, cellSize: 1, width: 2, height: 1};
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        positions: heightPositions.view,
        classifications: classifications.view,
        grid,
        terrain: terrain.view,
        surface: surface.view,
        canopy: canopy.view,
        coverage: coverage.view
      })
    );
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: heightPositions.view,
        grid,
        terrain: terrain.view,
        coverage: coverage.view,
        coverageBit: 2,
        heights: heights.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(mask.buffer)).toEqual(pdalOracle.outlier.mask);
    expect(await readFloats(heights.buffer)).toEqual(pdalOracle.height.heights);
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('change uncertainty overflow invalidates finite observations and recovers on resident input updates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const reference = fixture.view('float32x3', new Float32Array([0, 0, -1, 0, 0, 1]));
    const comparison = fixture.view('float32x3', new Float32Array([0, 0, -1, 0, 0, 1]));
    const cores = fixture.view('float32x3', new Float32Array([0, 0, 0]));
    const normals = fixture.view('float32x3', new Float32Array([0, 0, 1]));
    const changes = fixture.view('float32x3', new Float32Array(3));
    const roughness = fixture.view('float32x3', new Float32Array(3));
    const counts = fixture.view('uint32', new Uint32Array(2));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: reference.view,
        comparison: comparison.view,
        cores: cores.view,
        normals: normals.view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 2,
        minimumSupport: 2,
        confidenceMultiplier: 3e38,
        changes: changes.view,
        roughness: roughness.view,
        counts: counts.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([3]);
    expect(await readIntegers(counts.buffer)).toEqual([2, 2]);
    expect(await readFloats(changes.buffer)).toEqual([0, 0, 0]);
    expect(await readFloats(roughness.buffer)).toEqual([0, 0, 0]);
    reference.buffer.write(new Float32Array(6));
    comparison.buffer.write(new Float32Array(6));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect(await readFloats(changes.buffer)).toEqual([0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  0.001, 1, 1000
])('IDW height weights remain finite near cell centres at cell size %s', async cellSize => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([
        (0.5 + 1e-6) * cellSize,
        0.5 * cellSize,
        0,
        (1.5 - 1e-6) * cellSize,
        1.5 * cellSize,
        0,
        cellSize,
        cellSize,
        0,
        0.5 * cellSize,
        0.5 * cellSize,
        0
      ])
    );
    const terrain = fixture.view('float32', new Float32Array([10, 20, 30, 40]));
    const coverage = fixture.view('uint32', new Uint32Array([1, 1, 1, 1]));
    const heights = fixture.view('float32', new Float32Array(4));
    const status = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: positions.view,
        terrain: terrain.view,
        coverage: coverage.view,
        grid: {origin: [0, 0], cellSize, width: 2, height: 2},
        interpolation: {method: 'idw', radius: 2 * cellSize, power: 8},
        heights: heights.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 0, 0]);
    const actual = await readFloats(heights.buffer);
    [-10, -40, -25, -10].forEach((expected, index) =>
      expect(actual[index]).toBeCloseTo(expected, 4)
    );
  } finally {
    fixture.destroy();
  }
});

it('voxel sufficient-statistic merges avoid overflow for opposite finite centroids and attributes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const parts = [3e38, -3e38].map((value, index) => ({
      counts: fixture.view('uint32', new Uint32Array([1])).view,
      centroids: fixture.view('float32x3', new Float32Array([value, value, value])).view,
      representatives: fixture.view('uint32', new Uint32Array([0])).view,
      sourceRowOffset: index,
      attribute: {
        means: fixture.view('float32', new Float32Array([value])).view,
        counts: fixture.view('uint32', new Uint32Array([1])).view
      }
    }));
    const counts = fixture.view('uint32', new Uint32Array(1));
    const centroids = fixture.view('float32x3', new Float32Array(3));
    const representatives = fixture.view('uint32', new Uint32Array(1));
    const means = fixture.view('float32', new Float32Array(1));
    const support = fixture.view('uint32', new Uint32Array(1));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        parts,
        partitionPolicy: 'disjoint-owned-rows',
        counts: counts.view,
        centroids: centroids.view,
        representatives: representatives.view,
        attribute: {means: means.view, counts: support.view},
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect(await readIntegers(counts.buffer)).toEqual([2]);
    expect(await readFloats(centroids.buffer)).toEqual([0, 0, 0]);
    expect(await readFloats(means.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('indexed ICP supports a search radius spanning more than signed-int32 grid cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = new Float32Array([0, 0, 0, 1e-10, 0, 0, 0, 1e-10, 0, 0, 0, 1e-10]);
    const source = fixture.view('float32x3', coordinates);
    const target = fixture.view('float32x3', coordinates);
    const inlierMask = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        maximumDistance: 1,
        iterations: 1,
        targetGrid: {origin: [0, 0, 0], cellSize: 1e-10, dimensions: [2, 2, 2]},
        transform: fixture.view('float32', new Float32Array(16)).view,
        history: fixture.view('float32x3', new Float32Array(3)).view,
        status: fixture.view('uint32', new Uint32Array(1)).view,
        inlierMask: inlierMask.view
      })
    );
    await fixture.run();
    expect(await readIntegers(inlierMask.buffer)).toEqual([1, 1, 1, 1]);
  } finally {
    fixture.destroy();
  }
});

it('statistical filtering keeps finite population moments when their unnormalized sums overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Two blocks also exercise the weighted merge of normalized block variances.
    const squared = new Float32Array(512);
    for (let index = 0; index < 128; index++) squared[index] = 1e38;
    squared[511] = 3.24e38;
    const meanDistances = fixture.view('float32', new Float32Array(512));
    const mask = fixture.view('uint32', new Uint32Array(512));
    fixture.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: fixture.view('float32', squared).view,
        counts: fixture.view('uint32', new Uint32Array(512).fill(1)).view,
        capacity: 1,
        standardDeviations: 2,
        meanDistances: meanDistances.view,
        mask: mask.view
      })
    );
    await fixture.run();
    expect(await readIntegers(mask.buffer)).toEqual([...new Array(511).fill(1), 0]);
    const distances = await readFloats(meanDistances.buffer);
    const mean = distances.reduce((sum, value) => sum + value, 0) / distances.length;
    const variance =
      distances.reduce((sum, value) => sum + (value - mean) ** 2, 0) / distances.length;
    const threshold = mean + 2 * Math.sqrt(variance);
    expect(distances.map(value => Number(value <= threshold))).toEqual([
      ...new Array(511).fill(1),
      0
    ]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  'cell',
  'bilinear',
  'idw'
] as const)('height %s sampling rejects overflowing differences and recovers on resident updates', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const magnitude = Math.fround(3e38);
    const positions = fixture.view(
      'float32x3',
      new Float32Array([1, 1, magnitude, 1, 1, -magnitude, 0.5, 0.5, 0])
    );
    const terrain = fixture.view('float32', new Float32Array(4).fill(magnitude));
    const coverage = fixture.view('uint32', new Uint32Array(4).fill(1));
    const heights = fixture.view('float32', new Float32Array(3));
    const status = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: positions.view,
        terrain: terrain.view,
        coverage: coverage.view,
        grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2},
        interpolation:
          method === 'idw'
            ? {method, radius: 2, power: 8}
            : method === 'bilinear'
              ? {method}
              : undefined,
        heights: heights.view,
        status: status.view
      })
    );
    for (const sign of [1, -1]) {
      terrain.buffer.write(new Float32Array(4).fill(sign * magnitude));
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual(sign === 1 ? [0, 3, 0] : [3, 0, 0]);
      const actual = await readFloats(heights.buffer);
      expect(actual.every(Number.isFinite)).toBe(true);
      expect(actual[sign === 1 ? 1 : 0]).toBe(0);
      expect(Math.abs(actual[sign === 1 ? 0 : 1]) / magnitude).toBeLessThan(1e-6);
      expect(actual[2] / magnitude).toBeCloseTo(-sign, 6);
    }
    terrain.buffer.write(new Float32Array(4));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 0]);
    expect(await readFloats(heights.buffer)).toEqual([magnitude, -magnitude, 0]);
    if (method !== 'cell') {
      terrain.buffer.write(new Float32Array([magnitude, -magnitude, -magnitude, magnitude]));
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([0, 0, 0]);
      const actual = await readFloats(heights.buffer);
      [1, -1, -1].forEach((expected, index) =>
        expect(actual[index] / magnitude).toBeCloseTo(expected, 6)
      );
    }
  } finally {
    fixture.destroy();
  }
});
