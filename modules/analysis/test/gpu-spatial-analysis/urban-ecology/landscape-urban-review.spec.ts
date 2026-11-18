// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addSegregationCounterfactualRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import {
  addIsovistsRecipe,
  getGPUIsovistRayCount
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology-advanced.json';

it('R4 separate isovist outputs retain first angle zero and unit-range diamond area two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'isovist-independent-angle-measure');
  try {
    const angles = fixture.output('angles', 'float32', 4);
    const measures = fixture.output('measures', 'float32x4', 1);
    addIsovistsRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0]), 'float32x2', 1),
      segments: fixture.input('wall', new Float32Array([2, -2, 2, 2]), 'float32x4', 1),
      rayCount: 4,
      maximumDistance: 1,
      output: {angles: angles.view, measures: measures.view}
    });
    fixture.run();
    const actualAngles = await fixture.readFloat32(angles, 4);
    const actualMeasures = await fixture.readFloat32(measures, 4);
    expect(actualAngles[0]).toBe(0);
    // All four rays miss the wall beyond unit range: (1,0),(0,1),(-1,0),(0,-1).
    const expectedMeasures = [2, 4 * Math.sqrt(2), 1, 1];
    expectedMeasures.forEach((expected, index) => {
      expect(Number.isFinite(actualMeasures[index])).toBe(true);
      expect(actualMeasures[index]).toBeCloseTo(expected, 5);
    });
  } finally {
    fixture.destroy();
  }
});

it('disjoint scenario subviews preserve the unequal PySAL Shapley components', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'counterfactual-disjoint-scenarios');
  try {
    const scenarios = fixture.output('scenarios', 'float32x2', 8);
    const compositionOnly = fixture.graph.createDataView(scenarios.view.buffer, {
      format: 'float32x2',
      length: 4
    });
    const spatialOnly = fixture.graph.createDataView(scenarios.view.buffer, {
      format: 'float32x2',
      byteOffset: 32,
      length: 4
    });
    const decomposition = fixture.output('decomposition', 'float32x4', 1);
    addSegregationCounterfactualRecipe(fixture.graph, {
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
      method: 'composition',
      output: {compositionOnly, spatialOnly, decomposition: decomposition.view}
    });
    fixture.run();
    const expected = reference.counterfactual.methods.composition;
    const actualScenarios = await fixture.readFloat32(scenarios, 16);
    expected.populations.flat().forEach((value, index) => {
      expect(Number.isFinite(actualScenarios[index])).toBe(true);
      expect(actualScenarios[index]).toBeCloseTo(value, 4);
    });
    const actual = await fixture.readFloat32(decomposition, 4);
    expected.decomposition.forEach((value, index) => {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(actual[index]).toBeCloseTo(value, 5);
    });
    // Conservation alone would also pass if both scenarios had overwritten the same output.
    expect(actual[1]).toBeGreaterThan(0.25);
    expect(actual[2]).toBeLessThan(0.05);
  } finally {
    fixture.destroy();
  }
});

it('vertex isovists fill exactly helper-sized caller outputs with all nine angular events', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'isovist-expanded-output');
  try {
    const rayCount = getGPUIsovistRayCount(3, 1, 'vertices');
    expect(rayCount).toBe(9);
    const endpoints = fixture.output('endpoints', 'float32x2', rayCount);
    const distances = fixture.output('distances', 'float32', rayCount);
    const segments = fixture.output('hits', 'uint32', rayCount);
    const validity = fixture.output('validity', 'uint32', rayCount);
    const angles = fixture.output('angles', 'float32', rayCount);
    addIsovistsRecipe(fixture.graph, {
      positions: fixture.input('positions', new Float32Array([0, 0]), 'float32x2', 1),
      segments: fixture.input('obstacles', new Float32Array([2, -2, 2, 2]), 'float32x4', 1),
      rayCount: 3,
      sampling: 'vertices',
      angularEpsilon: 0.001,
      maximumDistance: 1,
      output: {
        endpoints: endpoints.view,
        distances: distances.view,
        segments: segments.view,
        validity: validity.view,
        angles: angles.view
      }
    });
    fixture.run();
    const expectedAngles = [
      0,
      Math.PI / 4 - 0.001,
      Math.PI / 4,
      Math.PI / 4 + 0.001,
      (2 * Math.PI) / 3,
      (4 * Math.PI) / 3,
      (7 * Math.PI) / 4 - 0.001,
      (7 * Math.PI) / 4,
      (7 * Math.PI) / 4 + 0.001
    ];
    const actualAngles = await fixture.readFloat32(angles, rayCount);
    const actualEndpoints = await fixture.readFloat32(endpoints, 2 * rayCount);
    const actualDistances = await fixture.readFloat32(distances, rayCount);
    expectedAngles.forEach((angle, index) => {
      expect(Number.isFinite(actualAngles[index])).toBe(true);
      expect(actualAngles[index]).toBeCloseTo(angle, 5);
      expect(Number.isFinite(actualEndpoints[2 * index])).toBe(true);
      expect(Number.isFinite(actualEndpoints[2 * index + 1])).toBe(true);
      expect(actualEndpoints[2 * index]).toBeCloseTo(Math.cos(angle), 5);
      expect(actualEndpoints[2 * index + 1]).toBeCloseTo(Math.sin(angle), 5);
      expect(actualDistances[index]).toBe(1);
    });
    expect(await fixture.readUint32(segments, rayCount)).toEqual(Array(9).fill(0xffffffff));
    expect(await fixture.readUint32(validity, rayCount)).toEqual(Array(9).fill(1));
  } finally {
    fixture.destroy();
  }
});
