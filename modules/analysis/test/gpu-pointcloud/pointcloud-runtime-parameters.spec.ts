// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUPointCloudChange,
  GPUPointCloudFeatures,
  GPUPointCloudGround,
  GPUPointCloudICP,
  GPUPointCloudSMRF,
  GPUPointCloudStatisticalFilter,
  getGPUPointCloudChangeParameterValues,
  getGPUPointCloudFeaturesParameterValues,
  getGPUPointCloudGroundParameterValues,
  getGPUPointCloudICPParameterValues,
  getGPUPointCloudSMRFParameterValues,
  getGPUPointCloudStatisticalFilterParameterValues
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import smrfOracle from './fixtures/smrf.json';

it('updates viewpoint on one compiled graph, preserving empty, truncated and padding rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'feature-orientation',
    format: 'float32',
    length: 4,
    values: getGPUPointCloudFeaturesParameterValues({viewpoint: [0, 0, 10]})
  });
  try {
    const normals = fixture.view('float32x3', new Float32Array(12), 12);
    const eigenvalues = fixture.view('float32x3', new Float32Array(12));
    const status = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: fixture.view(
          'float32x3',
          new Float32Array([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0])
        ).view,
        indices: fixture.view(
          'uint32',
          new Uint32Array([
            0, 1, 2, 3, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0, 1, 2, 3, 0, 1,
            0xffffffff, 0xffffffff
          ])
        ).view,
        counts: fixture.view('uint32', new Uint32Array([4, 0, 5, 2])).view,
        capacity: 4,
        parameters: parameters.importToGraph(fixture.graph),
        viewpoint: [0, 0, -10],
        output: {
          normals: normals.view,
          eigenvalues: eigenvalues.view,
          covariance: fixture.view('float32', new Float32Array(24)).view,
          shapes: fixture.view('float32x3', new Float32Array(12)).view,
          status: status.view
        }
      })
    );
    const expectedValues = [0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0];
    for (const viewpoint of [[0, 0, 10], [0, 0, -10], undefined] as const) {
      parameters.write(getGPUPointCloudFeaturesParameterValues({viewpoint}));
      await fixture.run(); // The fixture compiles once and reuses that graph.
      expect(await readIntegers(status.buffer)).toEqual([0, 1, 2, 1]);
      const direction = viewpoint?.[2] === -10 ? -1 : 1;
      const actualNormals = await readFloats(normals.buffer, 12);
      [0, 0, direction, 0, 0, 0, 0, 0, direction, 0, 0, 0].forEach((value, index) =>
        expect(actualNormals[index]).toBeCloseTo(value, 6)
      );
      const actualValues = await readFloats(eigenvalues.buffer);
      expectedValues.forEach((value, index) => expect(actualValues[index]).toBeCloseTo(value, 6));
    }
  } finally {
    fixture.destroy();
    parameters.destroy();
  }
});

it('retains an inherited zero deviation multiplier and getter-backed output bindings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const meanDistances = fixture.view('float32', new Float32Array(3));
    const validity = fixture.view('uint32', new Uint32Array(3));
    const output = Object.create({
      get meanDistances() {
        return meanDistances.view;
      },
      get validity() {
        return validity.view;
      }
    });
    const props = Object.assign(
      Object.create({
        get standardDeviations() {
          return 0;
        }
      }),
      {
        counts: fixture.view('uint32', Uint32Array.of(1, 1, 1)).view,
        squaredDistances: fixture.view('float32', Float32Array.of(1, 4, 9)).view,
        capacity: 1,
        output
      }
    );
    fixture.graph.add(new GPUPointCloudStatisticalFilter(props));
    await fixture.run();
    expect(await readFloats(meanDistances.buffer)).toEqual([1, 2, 3]);
    expect(await readIntegers(validity.buffer)).toEqual([1, 1, 0]);
  } finally {
    fixture.destroy();
  }
});

it('updates the population deviation multiplier with empty and padded neighbor lists', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const parameters = fixture.view(
      'float32',
      getGPUPointCloudStatisticalFilterParameterValues({standardDeviations: 0}),
      4
    );
    const validity = fixture.view('uint32', new Uint32Array(4));
    const meanDistances = fixture.view('float32', new Float32Array(4));
    fixture.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: fixture.view(
          'float32',
          new Float32Array([1, 1e30, 4, 1e30, 81, 81, 1e30, 1e30])
        ).view,
        counts: fixture.view('uint32', new Uint32Array([1, 1, 4, 0])).view,
        capacity: 2,
        parameters: parameters.view,
        standardDeviations: 0,
        output: {meanDistances: meanDistances.view, validity: validity.view}
      })
    );
    for (const multiplier of [0, 2, 0]) {
      parameters.buffer.write(
        getGPUPointCloudStatisticalFilterParameterValues({standardDeviations: multiplier}),
        4
      );
      await fixture.run();
      expect(await readFloats(meanDistances.buffer)).toEqual([1, 2, 9, -1]);
      // Valid means are 1, 2, 9: mean 4, population variance 38/3.
      expect(await readIntegers(validity.buffer)).toEqual([1, 1, Number(multiplier === 2), 0]);
    }
  } finally {
    fixture.destroy();
  }
});

it('updates progressive ground thresholds at inclusive float32 boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  const grid = {origin: [0, 0] as const, cellSize: 1, width: 9, height: 9};
  const radii = [1, 2];
  try {
    const pack = (initialThreshold: number, slope: number, maximumThreshold: number) =>
      getGPUPointCloudGroundParameterValues(
        {initialThreshold, slope, maximumThreshold},
        grid,
        radii
      );
    const parameters = fixture.view('float32', pack(0.2, 0.1, 1), 4);
    const groundMask = fixture.view('uint32', new Uint32Array(4));
    const status = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudGround({
        positions: fixture.view(
          'float32x3',
          new Float32Array([4.5, 4.5, 0.3, 9, 0, 1, 4.5, 4.5, 0.5, 4.5, 4.5, 0])
        ).view,
        classifications: fixture.view('uint32', new Uint32Array(4).fill(1)).view,
        grid,
        radii,
        initialThreshold: 0,
        slope: 0,
        maximumThreshold: 0,
        parameters: parameters.view,
        minimumSurface: fixture.view('float32', new Float32Array(81)).view,
        coverage: fixture.view('uint32', new Uint32Array(81).fill(1)).view,
        output: {
          groundMask: groundMask.view,
          status: status.view,
          outputClassifications: fixture.view('uint32', new Uint32Array(4)).view
        }
      })
    );
    for (const {values, expected} of [
      {values: pack(0.2, 0.1, 1), expected: [1, 0, 0, 1]},
      {values: pack(0.2, 0, 1), expected: [0, 0, 0, 1]},
      {values: pack(0.5, 1, 0.5), expected: [1, 0, 1, 1]}
    ]) {
      parameters.buffer.write(values, 4);
      await fixture.run();
      expect(await readIntegers(groundMask.buffer)).toEqual(expected);
      expect(await readIntegers(status.buffer)).toEqual([0, 1, 0, 0]);
    }
  } finally {
    fixture.destroy();
  }
});

it('updates SMRF scalar on one compiled graph against the independent PDAL/SciPy fixtures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  const scenario = smrfOracle.fixtures.find(candidate => candidate.name === 'slope-scalar')!;
  const zeroScalar = smrfOracle.fixtures.find(candidate => candidate.name === 'slope-zero-scalar')!;
  expect(scenario.positions).toEqual(zeroScalar.positions);
  const grid = {
    origin: [0, 0] as const,
    cellSize: 1,
    width: scenario.width,
    height: scenario.height,
    rowDirection: 'north' as const
  };
  try {
    const parameters = fixture.view(
      'float32',
      getGPUPointCloudSMRFParameterValues(scenario, grid, scenario.window),
      4
    );
    const outputClassifications = fixture.view(
      'uint32',
      new Uint32Array(scenario.classifications.length)
    );
    fixture.graph.add(
      new GPUPointCloudSMRF({
        positions: fixture.view('float32x3', new Float32Array(scenario.positions)).view,
        classifications: fixture.view('uint32', new Uint32Array(scenario.classifications)).view,
        mask: fixture.view('uint32', new Uint32Array(scenario.selection)).view,
        grid,
        window: scenario.window,
        cut: scenario.cut,
        slope: 0,
        threshold: 0,
        scalar: 0,
        parameters: parameters.view,
        output: {
          groundSurface: fixture.view('float32', new Float32Array(grid.width * grid.height)).view,
          groundMask: fixture.view('uint32', new Uint32Array(scenario.classifications.length)).view,
          outputClassifications: outputClassifications.view,
          status: fixture.view('uint32', new Uint32Array(scenario.classifications.length)).view
        }
      })
    );
    for (const settings of [scenario, zeroScalar, scenario]) {
      parameters.buffer.write(
        getGPUPointCloudSMRFParameterValues(settings, grid, scenario.window),
        4
      );
      await fixture.run();
      expect(await readIntegers(outputClassifications.buffer)).toEqual(settings.expectedClasses);
    }
  } finally {
    fixture.destroy();
  }
});

it('updates SMRF slope and threshold over a flat grid with a single raised cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  const grid = {origin: [0, 0] as const, cellSize: 1, width: 9, height: 9};
  const center = 40;
  try {
    const positions = Float32Array.from(
      Array.from({length: 81}, (_, cell) => [
        (cell % 9) + 0.5,
        Math.floor(cell / 9) + 0.5,
        cell === center ? 0.4 : 0
      ]).flat()
    );
    const pack = (slope: number, threshold: number) =>
      getGPUPointCloudSMRFParameterValues({slope, threshold, scalar: 0}, grid, 1);
    const parameters = fixture.view('float32', pack(0, 0.2));
    const groundMask = fixture.view('uint32', new Uint32Array(81));
    fixture.graph.add(
      new GPUPointCloudSMRF({
        positions: fixture.view('float32x3', positions).view,
        classifications: fixture.view('uint32', new Uint32Array(81).fill(1)).view,
        grid,
        window: 1,
        parameters: parameters.view,
        output: {
          groundSurface: fixture.view('float32', new Float32Array(81)).view,
          groundMask: groundMask.view,
          outputClassifications: fixture.view('uint32', new Uint32Array(81)).view,
          status: fixture.view('uint32', new Uint32Array(81)).view
        }
      })
    );
    // The opening removes the bump. Slope 0 rejects it as an anchor (inpainted terrain 0);
    // slope 1 retains it (terrain 0.4). With slope 0, threshold 0.5 accepts its 0.4 residual.
    for (const [slope, threshold, expectedCenter] of [
      [0, 0.2, 0],
      [1, 0.2, 1],
      [0, 0.5, 1],
      [0, 0.2, 0]
    ]) {
      parameters.buffer.write(pack(slope, threshold));
      await fixture.run();
      const expected = new Array(81).fill(1);
      expected[center] = expectedCenter;
      expect(await readIntegers(groundMask.buffer)).toEqual(expected);
    }
  } finally {
    fixture.destroy();
  }
});

for (const metric of ['point-to-point', 'point-to-plane'] as const) {
  it(`updates ${metric} ICP distance, robust scale, convergence and initial transform on one graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    const source = [
      [0, 1, 1],
      [0, 2, 1],
      [0, 1, 2],
      [1, 0, 1],
      [2, 0, 1],
      [1, 0, 2],
      [1, 1, 0],
      [2, 1, 0],
      [1, 2, 0]
    ];
    const target = source.map(point => [point[0] + 0.1, point[1], point[2]]);
    const initialTransform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 1];
    try {
      const pack = (
        maximumDistance: number,
        robustScale: number,
        tolerance: number,
        initial?: number[]
      ) =>
        getGPUPointCloudICPParameterValues({
          maximumDistance,
          robustScale,
          robustMethod: 'tukey',
          translationTolerance: tolerance,
          rotationTolerance: tolerance,
          initialTransform: initial
        });
      const parameters = fixture.view('float32', pack(0.05, 1, 1), 4);
      const status = fixture.view('uint32', new Uint32Array(1));
      const transform = fixture.view('float32', new Float32Array(16));
      fixture.graph.add(
        new GPUPointCloudICP({
          metric,
          source: fixture.view('float32x3', new Float32Array(source.flat())).view,
          target: fixture.view('float32x3', new Float32Array(target.flat())).view,
          targetNormals: fixture.view(
            'float32x3',
            Float32Array.from(
              source.flatMap((_, row) => [
                Number(row < 3),
                Number(row >= 3 && row < 6),
                Number(row >= 6)
              ])
            )
          ).view,
          targetGrid: {origin: [-1, -1, -1], cellSize: 1, dimensions: [5, 5, 5]},
          maximumDistance: 0.01,
          robust: {method: 'tukey', scale: 0.01},
          translationTolerance: 0,
          rotationTolerance: 0,
          maximumIterations: 1,
          parameters: parameters.view,
          output: {
            status: status.view,
            transform: transform.view,
            inlierMask: fixture.view('uint32', new Uint32Array(9)).view,
            history: fixture.view('float32x3', new Float32Array(3)).view
          }
        })
      );
      for (const {values, expectedStatus, expectedTranslation} of [
        {values: pack(0.05, 1, 1), expectedStatus: 2, expectedTranslation: 0},
        {values: pack(0.5, 1, 1), expectedStatus: 0, expectedTranslation: 0.1},
        {values: pack(0.5, 0.01, 1), expectedStatus: 2, expectedTranslation: 0},
        {values: pack(0.5, 1, 0), expectedStatus: 1, expectedTranslation: 0.1},
        {values: pack(0.01, 1, 1, initialTransform), expectedStatus: 0, expectedTranslation: 0.1}
      ]) {
        parameters.buffer.write(values, 4);
        await fixture.run();
        expect(await readIntegers(status.buffer)).toEqual([expectedStatus]);
        expect((await readFloats(transform.buffer))[12]).toBeCloseTo(expectedTranslation, 5);
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('updates M3C2 support and correlated uncertainty with a closed-form duplicate-point oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const pack = (
      minimumSupport: number,
      registrationError: number,
      covariance: number,
      confidenceMultiplier: number,
      referenceVariance: number,
      comparisonVariance: number
    ) =>
      getGPUPointCloudChangeParameterValues({
        minimumSupport,
        registrationError,
        confidenceMultiplier,
        registrationCovariance: [0, 0, 0, 0, 0, covariance],
        samplingCovariance: {referenceVariance, comparisonVariance, correlationLength: 1}
      });
    const parameters = fixture.view('uint32', pack(3, 2, 9, 1, 1, 4), 4);
    const changes = fixture.view('float32x3', new Float32Array(9));
    const roughness = fixture.view('float32x3', new Float32Array(9));
    const counts = fixture.view('uint32', new Uint32Array(6));
    const status = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: fixture.view('float32x3', new Float32Array(9)).view,
        comparison: fixture.view(
          'float32x3',
          new Float32Array([0, 0, 0.25, 0, 0, 0.25, 0, 0, 0.25])
        ).view,
        cores: fixture.view('float32x3', new Float32Array([0, 0, 0, 100, 100, 100, 0, 0, 0])).view,
        normals: fixture.view('float32x3', new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 0])).view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 1,
        parameters: parameters.view,
        samplingCovariance: {referenceVariance: 0, comparisonVariance: 0, correlationLength: 1},
        output: {
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    for (const {values, detection, expectedStatus} of [
      {values: pack(3, 2, 9, 1, 1, 4), detection: Math.sqrt(5) + Math.sqrt(13), expectedStatus: 0},
      {
        values: pack(3, 2, 9, 2, 1, 4),
        detection: 2 * (Math.sqrt(5) + Math.sqrt(13)),
        expectedStatus: 0
      },
      {values: pack(0xffffffff, 0, 0, 1, 0, 0), detection: 0, expectedStatus: 2},
      {values: pack(3, 0, 0, 1, 0, 0), detection: 0, expectedStatus: 0}
    ]) {
      parameters.buffer.write(values, 4);
      await fixture.run();
      expect(await readIntegers(counts.buffer)).toEqual([3, 3, 0, 0, 0, 0]);
      expect(await readIntegers(status.buffer)).toEqual([expectedStatus, 2, 1]);
      // Coincident samples are perfectly correlated: the per-cloud mean variances equal
      // the supplied point variances, not their values divided by three.
      const expected =
        expectedStatus === 0 ? [0.25, 0.25 - detection, 0.25 + detection] : [0, 0, 0];
      const actual = await readFloats(changes.buffer);
      [...expected, 0, 0, 0, 0, 0, 0].forEach((value, index) =>
        expect(actual[index]).toBeCloseTo(value, 5)
      );
      expect((await readFloats(roughness.buffer))[2]).toBeCloseTo(detection, 5);
    }
  } finally {
    fixture.destroy();
  }
});

for (const indexed of [false, true]) {
  it(`updates M3C2 correlation length for separated samples on one compiled graph, indexed=${indexed}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const pack = (correlationLength: number) =>
        getGPUPointCloudChangeParameterValues({
          minimumSupport: 3,
          confidenceMultiplier: 1,
          samplingCovariance: {referenceVariance: 1, comparisonVariance: 1, correlationLength}
        });
      const parameters = fixture.view('uint32', pack(0.5), 4);
      const changes = fixture.view('float32x3', new Float32Array(3));
      const roughness = fixture.view('float32x3', new Float32Array(3));
      const counts = fixture.view('uint32', new Uint32Array(2));
      const status = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudChange({
          reference: fixture.view('float32x3', new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0])).view,
          comparison: fixture.view(
            'float32x3',
            new Float32Array([0, 0, 0.25, 1, 0, 0.25, 2, 0, 0.25])
          ).view,
          cores: fixture.view('float32x3', new Float32Array([1, 0, 0])).view,
          normals: fixture.view('float32x3', new Float32Array([0, 0, 1])).view,
          alignment: 'already-aligned',
          radius: 2,
          halfDepth: 1,
          parameters: parameters.view,
          samplingCovariance: {referenceVariance: 1, comparisonVariance: 1, correlationLength: 1},
          confidenceMultiplier: 1,
          ...(indexed
            ? {grid: {origin: [-2, -2, -2] as const, cellSize: 1, dimensions: [6, 6, 6] as const}}
            : {}),
          output: {
            changes: changes.view,
            roughness: roughness.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      // Each epoch has three diagonal pairs, four ordered pairs at distance 1, and two at
      // distance 2. Unit point variance gives mean variance
      // (3 + 4 exp(-1/(2 length²)) + 2 exp(-4/(2 length²))) / 9.
      // These independent binary64 values do not use the contributor's covariance sum.
      const cases = [
        {correlationLength: 0.5, meanVariance: 0.3935568953558062},
        {correlationLength: 2, meanVariance: 0.8603387699737388},
        {correlationLength: 0.5, meanVariance: 0.3935568953558062}
      ];
      for (const {correlationLength, meanVariance} of cases) {
        parameters.buffer.write(pack(correlationLength), 4);
        await fixture.run(); // Compiles once, then only the resident parameter bytes change.
        expect(await readIntegers(counts.buffer)).toEqual([3, 3]);
        expect(await readIntegers(status.buffer)).toEqual([0]);
        // Epoch errors are independent, so their two mean variances add. No registration error.
        const detection = Math.sqrt(2 * meanVariance);
        const actualChanges = await readFloats(changes.buffer);
        [0.25, 0.25 - detection, 0.25 + detection].forEach((value, component) =>
          expect(actualChanges[component]).toBeCloseTo(value, 5)
        );
        const actualRoughness = await readFloats(roughness.buffer);
        [0, 0, detection].forEach((value, component) =>
          expect(actualRoughness[component]).toBeCloseTo(value, 5)
        );
        expect(actualRoughness[2] ** 2 / 2).toBeCloseTo(meanVariance, 5);
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('updates Huber ICP scales outside the float32 square range on one compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const sourceValues = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];
    const source = fixture.view('float32x3', new Float32Array(sourceValues));
    const target = fixture.view(
      'float32x3',
      Float32Array.from(
        sourceValues,
        (value, component) => value + (component % 3 === 0 ? 0.125 : 0)
      )
    );
    const pack = (robustScale: number) =>
      getGPUPointCloudICPParameterValues({
        maximumDistance: 1,
        robustMethod: 'huber',
        robustScale,
        translationTolerance: 1,
        rotationTolerance: 1
      });
    const parameters = fixture.view('float32', pack(1e20), 4);
    const transform = fixture.view('float32', new Float32Array(16));
    const status = fixture.view('uint32', new Uint32Array(1));
    const inlierMask = fixture.view('uint32', new Uint32Array(4));
    const history = fixture.view('float32x3', new Float32Array(3));
    const referenceTransform = fixture.view('float32', new Float32Array(16));
    const referenceStatus = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        id: 'runtime-huber-icp',
        source: source.view,
        target: target.view,
        maximumDistance: 1,
        robust: {method: 'huber', scale: 1},
        parameters: parameters.view,
        maximumIterations: 1,
        translationTolerance: 1,
        rotationTolerance: 1,
        output: {
          transform: transform.view,
          history: history.view,
          inlierMask: inlierMask.view,
          status: status.view
        }
      })
    );
    fixture.graph.add(
      new GPUPointCloudICP({
        id: 'unweighted-icp',
        source: source.view,
        target: target.view,
        maximumDistance: 1,
        maximumIterations: 1,
        translationTolerance: 1,
        rotationTolerance: 1,
        output: {
          transform: referenceTransform.view,
          status: referenceStatus.view,
          history: fixture.view('float32x3', new Float32Array(3)).view,
          inlierMask: fixture.view('uint32', new Uint32Array(4)).view
        }
      })
    );
    // All four correspondences have distance 1/8, so Huber gives equal weights at every scale.
    // The unique rigid solution of this tetrahedron is identity rotation and translation (1/8,0,0).
    const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.125, 0, 0, 1];
    for (const robustScale of [1e20, 1e-30, 1e20]) {
      parameters.buffer.write(pack(robustScale), 4);
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([0]);
      expect(await readIntegers(referenceStatus.buffer)).toEqual([0]);
      expect(await readIntegers(inlierMask.buffer)).toEqual([1, 1, 1, 1]);
      const actual = await readFloats(transform.buffer);
      const reference = await readFloats(referenceTransform.buffer);
      expected.forEach((value, entry) => {
        expect(actual[entry]).toBeCloseTo(value, 5);
        expect(actual[entry]).toBeCloseTo(reference[entry], 5);
      });
      const actualHistory = await readFloats(history.buffer);
      [4, 0.125, 0.125].forEach((value, component) =>
        expect(actualHistory[component]).toBeCloseTo(value, 5)
      );
    }
  } finally {
    fixture.destroy();
  }
});
