// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudFeatures,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudSurfaces,
  GPUPointCloudHeight,
  GPUPointCloudGround,
  GPUPointCloudICP,
  GPUPointCloudChange
} from '@luma-contributors/analysis/pointcloud-analysis';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import oracle from './fixtures/pointcloud.json';
import pdalOracle from './fixtures/pdal-pointcloud.json';
import {requireTestDevice} from '../utils/require-test-device';

it('3D neighborhoods and covariance match independent NumPy eigh on a translated tilted plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([...oracle.positions, 1024, -2048, 100])
    );
    const indices = fixture.view('uint32', new Uint32Array(80));
    const distances = fixture.view('float32', new Float32Array(80));
    const counts = fixture.view('uint32', new Uint32Array(10));
    const normals = fixture.view('float32x3', new Float32Array(30));
    const eigenvalues = fixture.view('float32x3', new Float32Array(30));
    const shapes = fixture.view('float32x3', new Float32Array(30));
    const status = fixture.view('uint32', new Uint32Array(10));
    const covariance = fixture.view('float32', new Float32Array(60));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: 5,
        capacity: 8,
        output: {indices: indices.view, squaredDistances: distances.view, counts: counts.view}
      })
    );
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: positions.view,
        indices: indices.view,
        counts: counts.view,
        capacity: 8,
        output: {
          normals: normals.view,
          eigenvalues: eigenvalues.view,
          shapes: shapes.view,
          status: status.view,
          covariance: covariance.view
        }
      })
    );
    await fixture.run(2);
    expect(await readIntegers(counts.buffer)).toEqual([...Array(9).fill(8), 0]);
    expect(await readIntegers(status.buffer)).toEqual([...Array(9).fill(0), 1]);
    const actualNormals = await readFloats(normals.buffer);
    const actualValues = await readFloats(eigenvalues.buffer);
    const actualCovariance = await readFloats(covariance.buffer);
    oracle.covariances
      .flat()
      .forEach((value, index) => expect(actualCovariance[index]).toBeCloseTo(value, 4));
    oracle.normals
      .flat()
      .forEach((value, index) => expect(actualNormals[index]).toBeCloseTo(value, 4));
    pdalOracle.normals
      .flat()
      .forEach((value, index) => expect(actualNormals[index]).toBeCloseTo(value, 4));
    oracle.eigenvalues
      .flat()
      .forEach((value, index) => expect(actualValues[index]).toBeCloseTo(value, 4));
    positions.buffer.write(new Float32Array(30));
    await fixture.run();
    expect(await readIntegers(counts.buffer)).toEqual(Array(10).fill(9));
    expect(await readIntegers(status.buffer)).toEqual(Array(10).fill(3));
    expect(await readFloats(normals.buffer)).toEqual(Array(30).fill(0));
  } finally {
    fixture.destroy();
  }
});

it('3D neighborhoods expose truncation, stable ties and collinear degeneracy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 2, 0, 0, -2])
    );
    const indices = fixture.view('uint32', new Uint32Array(15));
    const distances = fixture.view('float32', new Float32Array(15));
    const counts = fixture.view('uint32', new Uint32Array(5));
    const normals = fixture.view('float32x3', new Float32Array(15));
    const eigenvalues = fixture.view('float32x3', new Float32Array(15));
    const shapes = fixture.view('float32x3', new Float32Array(15));
    const status = fixture.view('uint32', new Uint32Array(5));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: 5,
        capacity: 3,
        output: {indices: indices.view, squaredDistances: distances.view, counts: counts.view}
      })
    );
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: positions.view,
        indices: indices.view,
        counts: counts.view,
        capacity: 3,
        output: {
          normals: normals.view,
          eigenvalues: eigenvalues.view,
          shapes: shapes.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect((await readIntegers(indices.buffer)).slice(0, 3)).toEqual([1, 2, 3]);
    expect((await readFloats(distances.buffer)).slice(0, 3)).toEqual([1, 1, 4]);
    expect(await readIntegers(status.buffer)).toEqual([3, 3, 3, 3, 3]);
    expect(await readFloats(normals.buffer)).toEqual(Array(15).fill(0));
  } finally {
    fixture.destroy();
  }
});

it('statistical filtering retains original rows and voxel downsampling keeps minimum-ID representatives', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0, 100, 0, 0])
    );
    const indices = fixture.view('uint32', new Uint32Array(8));
    const distances = fixture.view('float32', new Float32Array(8));
    const counts = fixture.view('uint32', new Uint32Array(4));
    const means = fixture.view('float32', new Float32Array(4));
    const mask = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: 200,
        capacity: 2,
        output: {indices: indices.view, squaredDistances: distances.view, counts: counts.view}
      })
    );
    fixture.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: distances.view,
        counts: counts.view,
        capacity: 2,
        standardDeviations: 1,
        output: {meanDistances: means.view, validity: mask.view}
      })
    );
    const representatives = fixture.view('uint32', new Uint32Array(2));
    const voxelCounts = fixture.view('uint32', new Uint32Array(2));
    const samples = fixture.view('float32x3', new Float32Array(6));
    fixture.graph.add(
      new GPUPointCloudVoxelDownsample({
        positions: positions.view,
        origin: [0, 0, 0],
        voxelSize: 3,
        gridSize: [2, 1, 1],
        representatives: representatives.view,
        counts: voxelCounts.view,
        output: samples.view
      })
    );
    await fixture.run();
    expect(await readFloats(means.buffer)).toEqual([1.5, 1, 1.5, 98.5]);
    expect(await readIntegers(mask.buffer)).toEqual([1, 1, 1, 0]);
    expect(await readIntegers(representatives.buffer)).toEqual([0, 0xffffffff]);
    expect(await readFloats(samples.buffer)).toEqual([0, 0, 0, 0, 0, 0]);
    positions.buffer.write(new Float32Array(12));
    await fixture.run();
    expect(await readFloats(means.buffer)).toEqual([0, 0, 0, 0]);
    expect(await readIntegers(mask.buffer)).toEqual([1, 1, 1, 1]);
    expect(await readIntegers(voxelCounts.buffer)).toEqual([4, 0]);
  } finally {
    fixture.destroy();
  }
});

it('cell terrain/surface/canopy and HAG preserve negative heights and missing coverage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const grid = {origin: [0, 0] as const, cellSize: 1, width: 3, height: 1};
    const positions = fixture.view(
      'float32x3',
      new Float32Array([0.2, 0.2, 5, 0.4, 0.2, 9, 0.6, 0.2, 3, 1.2, 0.2, 6, 3, 0.2, 4])
    );
    const classes = fixture.view('uint32', new Uint32Array([2, 5, 1, 1, 2]));
    const terrain = fixture.view('float32', new Float32Array(3));
    const surface = fixture.view('float32', new Float32Array(3));
    const canopy = fixture.view('float32', new Float32Array(3));
    const coverage = fixture.view('uint32', new Uint32Array(3));
    const heights = fixture.view('float32', new Float32Array(5));
    const status = fixture.view('uint32', new Uint32Array(5));
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: classes.view,
        grid,
        output: {
          terrain: terrain.view,
          surface: surface.view,
          canopy: canopy.view,
          coverage: coverage.view
        }
      })
    );
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: positions.view,
        grid,
        terrain: terrain.view,
        coverage: coverage.view,
        coverageBit: 2,
        output: {heights: heights.view, status: status.view}
      })
    );
    await fixture.run();
    expect(await readFloats(terrain.buffer)).toEqual([5, 0, 0]);
    expect(await readFloats(surface.buffer)).toEqual([9, 6, 0]);
    expect(await readFloats(canopy.buffer)).toEqual([4, 0, 0]);
    expect(await readIntegers(coverage.buffer)).toEqual([3, 1, 0]);
    expect(await readFloats(heights.buffer)).toEqual([0, 4, -2, 0, 0]);
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 0, 2, 1]);
  } finally {
    fixture.destroy();
  }
});

it('progressive morphology removes an elevated obstacle and reports edge support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const values = Array.from({length: 49}, (_, index) => (index === 24 ? 10 : 0));
    const positions = fixture.view(
      'float32x3',
      new Float32Array(
        values.flatMap((value, index) => [(index % 7) + 0.5, Math.floor(index / 7) + 0.5, value])
      )
    );
    const classes = fixture.view('uint32', new Uint32Array(values.map(value => (value ? 6 : 1))));
    const minimum = fixture.view('float32', new Float32Array(values));
    const coverage = fixture.view('uint32', new Uint32Array(49).fill(1));
    const ground = fixture.view('uint32', new Uint32Array(49));
    const output = fixture.view('uint32', new Uint32Array(49));
    const status = fixture.view('uint32', new Uint32Array(49));
    fixture.graph.add(
      new GPUPointCloudGround({
        positions: positions.view,
        classifications: classes.view,
        grid: {origin: [0, 0], cellSize: 1, width: 7, height: 7},
        radii: [1, 2],
        initialThreshold: 0.2,
        slope: 0.1,
        maximumThreshold: 1,
        minimumSurface: minimum.view,
        coverage: coverage.view,
        output: {groundMask: ground.view, outputClassifications: output.view, status: status.view}
      })
    );
    await fixture.run();
    expect(await readIntegers(ground.buffer)).toEqual(values.map(value => (value ? 0 : 1)));
    expect(await readIntegers(output.buffer)).toEqual(values.map(value => (value ? 6 : 2)));
    expect((await readIntegers(status.buffer))[0]).toBe(2);
  } finally {
    fixture.destroy();
  }
});

it('bounded rigid ICP matches independent NumPy SVD Kabsch and reports convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const source = fixture.view('float32x3', new Float32Array(oracle.source));
    const target = fixture.view('float32x3', new Float32Array(oracle.target));
    const transform = fixture.view('float32', new Float32Array(16));
    const history = fixture.view('float32x3', new Float32Array(12));
    const mask = fixture.view('uint32', new Uint32Array(5));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        maximumDistance: 1,
        maximumIterations: 4,
        output: {
          transform: transform.view,
          history: history.view,
          inlierMask: mask.view,
          status: status.view
        }
      })
    );
    await fixture.run(2);
    const actual = await readFloats(transform.buffer);
    oracle.transform.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 4));
    pdalOracle.icp.transform.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 4));
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect(await readIntegers(mask.buffer)).toEqual([1, 1, 1, 1, 1]);
    expect((await readFloats(history.buffer))[10]).toBeLessThan(1e-4);
    target.buffer.write(new Float32Array(oracle.source));
    await fixture.run();
    const identity = await readFloats(transform.buffer);
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].forEach((value, index) =>
      expect(identity[index]).toBeCloseTo(value, 4)
    );
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('ICP reports insufficient correspondences rather than a zero-error success', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const source = fixture.view('float32x3', new Float32Array([0, 0, 0]));
    const target = fixture.view('float32x3', new Float32Array([100, 0, 0]));
    const transform = fixture.view('float32', new Float32Array(16));
    const history = fixture.view('float32x3', new Float32Array(3));
    const mask = fixture.view('uint32', new Uint32Array(1));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        maximumDistance: 1,
        maximumIterations: 1,
        output: {
          transform: transform.view,
          history: history.view,
          inlierMask: mask.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([2]);
    expect(await readIntegers(mask.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('cylindrical surface change matches analytic roughness/CI and reverses signed normals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const reference = fixture.view(
      'float32x3',
      new Float32Array([-0.1, 0, -0.1, 0, 0, 0, 0.1, 0, 0.1])
    );
    const comparison = fixture.view(
      'float32x3',
      new Float32Array([-0.1, 0, 0.9, 0, 0, 1, 0.1, 0, 1.1])
    );
    const cores = fixture.view('float32x3', new Float32Array([0, 0, 0, 0, 0, 0, 20, 0, 0]));
    const normals = fixture.view('float32x3', new Float32Array([0, 0, 1, 0, 0, -1, 0, 0, 1]));
    const changes = fixture.view('float32x3', new Float32Array(9));
    const roughness = fixture.view('float32x3', new Float32Array(9));
    const counts = fixture.view('uint32', new Uint32Array(6));
    const status = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: reference.view,
        comparison: comparison.view,
        cores: cores.view,
        normals: normals.view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 2,
        registrationError: 0.02,
        output: {
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    const detection = 1.96 * (Math.sqrt(0.01 / 3 + 0.01 / 3) + 0.02);
    const actual = await readFloats(changes.buffer);
    [1, 1 - detection, 1 + detection, -1, -1 - detection, -1 + detection].forEach((value, index) =>
      expect(actual[index]).toBeCloseTo(value, 5)
    );
    expect(await readIntegers(counts.buffer)).toEqual([3, 3, 3, 3, 0, 0]);
    expect(await readIntegers(status.buffer)).toEqual([0, 0, 2]);
    const noise = await readFloats(roughness.buffer);
    expect(noise[0]).toBeCloseTo(0.1, 5);
    expect(noise[1]).toBeCloseTo(0.1, 5);
    comparison.buffer.write(new Float32Array([100, 0, 0, 100, 0, 0, 100, 0, 0]));
    normals.buffer.write(new Float32Array([0, 0, 1, 0, 0, -1, 0, 0, 0]));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([2, 2, 1]);
    expect(await readIntegers(counts.buffer)).toEqual([3, 0, 3, 0, 0, 0]);
    expect(await readFloats(changes.buffer)).toEqual(Array(9).fill(0));
  } finally {
    fixture.destroy();
  }
});

it('ICP distinguishes iteration exhaustion, supplied alignment and underdetermined collinear rotation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = [
    {source: oracle.source, target: oracle.target, initialTransform: undefined, expected: 1},
    {source: oracle.source, target: oracle.target, initialTransform: oracle.transform, expected: 0},
    {
      source: [0, 0, 0, 1, 0, 0, 2, 0, 0],
      target: [0, 0, 0, 1, 0, 0, 2, 0, 0],
      initialTransform: undefined,
      expected: 3
    }
  ];
  for (const scenario of cases) {
    const fixture = makeSpatialFixture(device);
    try {
      const source = fixture.view('float32x3', new Float32Array(scenario.source));
      const target = fixture.view('float32x3', new Float32Array(scenario.target));
      const transform = fixture.view('float32', new Float32Array(16));
      const history = fixture.view('float32x3', new Float32Array(3));
      const mask = fixture.view('uint32', new Uint32Array(scenario.source.length / 3));
      const status = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudICP({
          source: source.view,
          target: target.view,
          initialTransform: scenario.initialTransform,
          maximumDistance: 1,
          maximumIterations: 1,
          output: {
            transform: transform.view,
            history: history.view,
            inlierMask: mask.view,
            status: status.view
          }
        })
      );
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([scenario.expected]);
      expect(await readIntegers(mask.buffer)).toEqual(Array(scenario.source.length / 3).fill(1));
    } finally {
      fixture.destroy();
    }
  }
});

it('surface change applies a supplied resident comparison-to-reference alignment', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const reference = fixture.view('float32x3', new Float32Array([-0.1, 0, 0, 0, 0, 0, 0.1, 0, 0]));
    const comparison = fixture.view(
      'float32x3',
      new Float32Array([-0.1, 0, 1, 0, 0, 1, 0.1, 0, 1])
    );
    const transform = fixture.view(
      'float32',
      new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 1])
    );
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
        alignment: {transform: transform.view},
        radius: 1,
        halfDepth: 2,
        output: {
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readFloats(changes.buffer)).toEqual([0, 0, 0]);
    expect(await readIntegers(counts.buffer)).toEqual([3, 3]);
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
