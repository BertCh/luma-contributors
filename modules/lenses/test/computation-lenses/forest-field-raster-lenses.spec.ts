// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice, submitGraph} from '@luma-contributors/analysis/testing';
import {
  validateGPUComputationRecord,
  validateGPUComputationSidecars
} from '@luma-contributors/analysis/computation-record';
import {getLensScenario} from '../../src/computation-lenses';
import {buildForestInferenceScenario} from '../../src/computation-lenses/scenarios/gpu-dataframe/forest-inference.lens';
import {buildGaussianRandomFieldScenario} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/gaussian-random-field.lens';
import {buildRasterContrastEqualizationScenario} from '../../src/computation-lenses/scenarios/gpu-raster/raster-contrast-equalization.lens';
import {buildRasterEmbeddingQuantizeScenario} from '../../src/computation-lenses/scenarios/gpu-raster/raster-embedding-quantize.lens';
import {createCoverageTrancheOneHarness} from './raster-terrain/coverage-tranche-one-harness';
const SCENARIOS = [
  {
    id: 'gpu-dataframe.forest-inference',
    element: 'GPUForestInference',
    cells: 5,
    build: buildForestInferenceScenario,
    outputs: [
      ['predictions', 'float32', 5],
      ['scores', 'float32', 10],
      ['validity', 'uint32', 5],
      ['leafIndices', 'uint32', 5]
    ]
  },
  {
    id: 'gpu-spatial-analysis.gaussian-random-field',
    element: 'GPUGaussianRandomField',
    cells: 16,
    build: buildGaussianRandomFieldScenario,
    outputs: [
      ['values', 'float32', 16],
      ['validity', 'uint32', 16],
      ['approximation', 'uint32', 1],
      ['clippedEigenvalueMass', 'float32', 1]
    ]
  },
  {
    id: 'gpu-raster.raster-contrast-equalization',
    element: 'GPURasterContrastEqualization',
    cells: 16,
    build: buildRasterContrastEqualizationScenario,
    outputs: [
      ['equalizedValues', 'float32', 16],
      ['equalizedValidity', 'uint32', 16],
      ['approximation', 'uint32', 1]
    ]
  },
  {
    id: 'gpu-raster.raster-embedding-quantize',
    element: 'GPURasterEmbeddingQuantize',
    cells: 3,
    build: buildRasterEmbeddingQuantizeScenario,
    outputs: [
      ['quantized', 'uint32', 3],
      ['validity', 'uint32', 3]
    ]
  }
] as const;

it.each(SCENARIOS)('captures $element and preserves its edge cases', async scenario => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCoverageTrancheOneHarness(device, scenario.id, scenario.cells);
  try {
    await scenario.build(harness.session);
    harness.upload();
    const compiled = harness.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const read = (name: string) => harness.read(compiled, name);
      if (scenario.element === 'GPUForestInference') {
        // Hand-derived: <=0 and missing route to a tied leaf (first class); >0 to class 1.
        const predictions = await read('predictions');
        expect(predictions.slice(0, 4)).toEqual([0, 0, 1, 0]);
        expect(predictions[4]).toBeNaN();
        const scores = await read('scores');
        expect(scores.slice(0, 8)).toEqual([0.5, 0.5, 0.5, 0.5, 0, 1, 0.5, 0.5]);
        expect(scores.slice(8).every(Number.isNaN)).toBe(true);
        expect(await read('validity')).toEqual([1, 1, 1, 1, 0]);
        expect(await read('leafIndices')).toEqual([1, 1, 2, 1, 0xffffffff]);
      } else if (scenario.element === 'GPUGaussianRandomField') {
        expect(await read('validity')).toEqual(new Array(16).fill(1));
        expect((await read('values')).every(Number.isFinite)).toBe(true);
        expect(await read('approximation')).toEqual([1]);
        expect(await read('clippedEigenvalueMass')).toEqual([0]);
      } else if (scenario.element === 'GPURasterContrastEqualization') {
        const values = await read('equalizedValues');
        expect(values[0]).toBeNaN();
        expect(values.slice(1).every(value => value >= 0 && value <= 1)).toBe(true);
        expect(await read('equalizedValidity')).toEqual([0, ...new Array(15).fill(1)]);
        expect(await read('approximation')).toEqual([1]);
      } else {
        // [64,0,-127,-128], [-128,-128,-128,-128], [-128,0,0,0], low byte first.
        expect(await read('quantized')).toEqual([0x80810040, 0x80808080, 0x80]);
        expect(await read('validity')).toEqual([1, 0, 0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    harness.destroy();
  }
  const descriptor = getLensScenario(scenario.id);
  if (!descriptor) throw new Error(`Missing scenario ${scenario.id}`);
  const executable = await descriptor.load();
  const record = await executable.run(device);
  expect(validateGPUComputationRecord(record).valid).toBe(true);
  expect(
    validateGPUComputationSidecars(record, {captures: record.captures, metrics: record.metrics})
      .valid
  ).toBe(true);
  expect(record.captures?.snapshots.length).toBeGreaterThan(0);
});
