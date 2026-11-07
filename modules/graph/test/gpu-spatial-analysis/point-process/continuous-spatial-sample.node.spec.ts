// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('continuous designs expose intensity provenance and audit all generated line/area kernels', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32' | 'uint32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `continuous-audit-${serial++}`, format, length);
  for (const type of ['line', 'area'] as const) {
    const positions = view('float32x2', 6);
    const output = {
      positions: view('float32x2', 32),
      designWeights: view('float32', 32),
      provenance: view('uint32x4', 32),
      status: view('uint32x4', 2)
    };
    const props = {
      frame:
        type === 'line'
          ? {type, positions, legacySites: view('float32', 2)}
          : {type, positions, bounds: [0, 0, 1, 1] as const, legacySites: view('float32x2', 2)},
      spacing: 0.5,
      seed: 7,
      waves: 2,
      capacity: 16,
      parameters: view('uint32', 2),
      output
    };
    const sample = new GPUContinuousSpatialSample(props);
    expect(sample.samplingIntensity).toBe(type === 'line' ? 2 : 4);
    for (const [identifier, source] of getParsedKernelSources(sample.getCommandNodes(graph)))
      expect(auditWGSLModule(source), identifier).toEqual([]);
    expect(() => new GPUContinuousSpatialSample({...props, spacing: 0})).toThrow();
    expect(
      () => new GPUContinuousSpatialSample({...props, output: {...output, positions}})
    ).toThrow();
    expect(() => new GPUContinuousSpatialSample({...props, replicate: 0xffffffff})).toThrow();
  }
});
