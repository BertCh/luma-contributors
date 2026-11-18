// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice, getParsedKernelSources} from '@luma-contributors/analysis/testing';
import {auditWGSLModule} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/kernel-audit-wgsl';
import {listLensScenarios} from '../../src/computation-lenses';
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

it.each(SCENARIOS)('builds $element with typed captures and valid kernels', async scenario => {
  expect(listLensScenarios().find(candidate => candidate.id === scenario.id)?.element).toBe(
    scenario.element
  );
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  const harness = createCoverageTrancheOneHarness(device, scenario.id, scenario.cells);
  try {
    await scenario.build(harness.session);
    for (const [name, format, length] of scenario.outputs) {
      const capture = harness.captures.get(name);
      expect(capture?.view).toMatchObject({format, length});
      expect(capture?.spec.meaning?.length).toBeGreaterThan(10);
    }
    const kernels = getParsedKernelSources((harness.graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.size).toBeGreaterThan(0);
    for (const {source, constants} of kernels.configurations)
      expect(auditWGSLModule(source, constants)).toEqual([]);
    const compiled = harness.graph.compile();
    compiled.destroy();
  } finally {
    harness.destroy();
    device.destroy();
  }
});
