// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUAreaOfApplicability} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-area-of-applicability';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {watchConstructionFields} from './construction-capture-audit.test-utils';
import {executeSpatialAuditKernel} from './spatial-design-audit.test-utils';

it('R3 identical decimal predictors have exactly zero sample deviation', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `constant-area-${nextView++}`, format, length);
  try {
    const contributor = new GPUAreaOfApplicability({
      training: view('float32', 16),
      query: view('float32', 2),
      predictorCount: 2,
      output: {
        trainDissimilarity: view('float32', 8),
        queryDissimilarity: view('float32', 1),
        applicable: view('uint32', 1),
        summary: view('float32', 6),
        status: view('uint32', 1)
      }
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id.endsWith('-moments'))
    );
    const moments = new Float32Array(4);
    executeSpatialAuditKernel(kernel, {
      training: Float32Array.from({length: 16}, (_, index) =>
        index % 2 === 0 ? 0.1 : (index - 1) / 2
      ),
      rowValid: new Uint32Array(8).fill(1),
      moments
    });
    // Identical packed values imply exact SD=0; [0,...,7] has sample variance 6.
    expect(moments[0]).toBe(Math.fround(0.1));
    expect(moments[1]).toBe(0);
    expect(moments[2]).toBe(3.5);
    expect(Math.abs(moments[3] - Math.sqrt(6))).toBeLessThan(1e-6);
  } finally {
    device.destroy();
  }
});

it('captures AOA getters and cross-validation/output maps once before graph construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `area-capture-${nextView++}`, format, length);
  try {
    const crossValues = {folds: view('uint32', 8), training: view('uint32', 16), foldCount: 2};
    const cross = watchConstructionFields(crossValues, 'crossValidation');
    const outputValues = {
      trainDissimilarity: view('float32', 8),
      queryDissimilarity: view('float32', 2),
      applicable: view('uint32', 2),
      summary: view('float32', 6),
      status: view('uint32', 1)
    };
    const output = watchConstructionFields(outputValues, 'output');
    const values = {
      id: 'area-capture',
      training: view('float32', 16),
      query: view('float32', 4),
      predictorCount: 2,
      weights: view('float32', 2),
      mask: view('uint32', 8),
      crossValidation: cross.props,
      output: output.props
    };
    const originalWeights = values.weights,
      originalSummary = outputValues.summary;
    const watched = watchConstructionFields(values, 'props');
    const contributor = new GPUAreaOfApplicability(watched.props);
    for (const watcher of [watched, cross, output]) watcher.expectReadOnce();
    values.predictorCount = 16;
    values.weights = view('float32', 1);
    crossValues.foldCount = 64;
    crossValues.training = view('uint32', 1);
    outputValues.summary = view('float32', 1);
    expect(contributor.props.predictorCount).toBe(2);
    expect(contributor.props.weights).toBe(originalWeights);
    expect(contributor.props.output.summary).toBe(originalSummary);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    for (const watcher of [watched, cross, output]) watcher.expectReadOnce();
  } finally {
    device.destroy();
  }
});

it('audit-validation AOA suppresses overflowing query distances instead of publishing its finite search sentinel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `area-audit-${nextView++}`, format, length);
  const contributor = new GPUAreaOfApplicability({
    training: view('float32', 8),
    query: view('float32', 1),
    predictorCount: 1,
    output: {
      trainDissimilarity: view('float32', 8),
      queryDissimilarity: view('float32', 1),
      applicable: view('uint32', 1),
      summary: view('float32', 6),
      status: view('uint32', 1)
    }
  });
  const [kernel] = getGPUCommandNodeKernelSources(
    contributor.getCommandNodes(graph).filter(node => node.id === 'area-of-applicability-query')
  );
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    const values: Record<string, Float32Array | Uint32Array> = {
      query: Float32Array.of(1e20),
      scaled: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]),
      scaledQuery: Float32Array.of(1e20),
      rowValid: new Uint32Array(8).fill(1),
      summary: Float32Array.of(1, 1, 8, 8, 1, 1),
      queryDissimilarity: new Float32Array(1),
      applicable: new Uint32Array(1)
    };
    const bindings = Object.fromEntries(
      new WgslReflect(kernel.source).storage.map(binding => [binding.binding, values[binding.name]])
    );
    new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
      kernel.entryPoint,
      [1, 1, 1],
      {0: bindings},
      {constants: kernel.constants}
    );
    // |1e20 - row|^2 overflows f32 for every candidate; no computed distance is usable.
    expect(values['queryDissimilarity'][0]).toBeNaN();
    expect(Array.from(values['applicable'])).toEqual([0]);
  } finally {
    errors.mockRestore();
    device.destroy();
  }
});
