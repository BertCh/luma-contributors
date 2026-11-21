// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validatePointProcessView} from '../../../src/gpu-spatial-analysis/point-process/point-process-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
it('VAL-11 identifies the instance, exact port, actual format and both row counts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createTransientView(graph, 'wrong-input', 'uint32', 3);
  expect(() =>
    validatePointProcessView('my-layer', 'output.intensity', view, ['float32'], 4)
  ).toThrow(/my-layer.*output.intensity.*float32.*uint32/);
  expect(() =>
    validatePointProcessView('my-layer', 'output.intensity', view, ['uint32'], 4)
  ).toThrow(/my-layer.*output.intensity.*4.*3/);
  expect(() =>
    validatePointProcessView('my-layer', 'output.intensity', view, ['uint32'], 3)
  ).not.toThrow();
});

it('reports how to migrate the old point-process view validation call', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const view = createTransientView(graph, 'legacy-call', 'uint32', 3);
    expect(() =>
      // @ts-expect-error Exercise an untyped call with the retired argument list.
      validatePointProcessView(view, 'uint32', 3, 'density')
    ).toThrow('validatePointProcessView(id, path, view, formats, expectedRows)');
  } finally {
    device.destroy();
  }
});

it('DEF-12 generated envelope kernels use only statistic-specific bindings and pass host WGSL audit', async () => {
  const {GPUPointPatternEnvelope} = await import(
    '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope'
  );
  const {getParsedKernelSources} = await import('../../utils/gpu-contributor-test-utils');
  const {auditWGSLModule} = await import('../../analysis-core/kernel-audit/kernel-audit-wgsl');
  for (const statistic of [
    'maximum-deviation',
    'integrated-squared-deviation',
    'extreme-rank'
  ] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const envelope = new GPUPointPatternEnvelope({
      id: 'audit-envelope',
      statistic,
      observed: createTransientView(graph, 'observed', 'float32', 2),
      simulations: createTransientView(graph, 'simulations', 'float32', 38),
      ...(statistic === 'extreme-rank'
        ? {}
        : {reference: createTransientView(graph, 'reference', 'float32', 2)}),
      ...(statistic === 'integrated-squared-deviation'
        ? {integrationWeights: createTransientView(graph, 'weights', 'float32', 2)}
        : {}),
      nullModel: 'specified null',
      fittedParameters: 'none',
      output: {
        lower: createTransientView(graph, 'lower', 'float32', 2),
        upper: createTransientView(graph, 'upper', 'float32', 2),
        statistics: createTransientView(graph, 'statistics', 'float32', 4),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    });
    for (const {id, source, constants} of getParsedKernelSources(envelope.getCommandNodes(graph))
      .configurations)
      expect(auditWGSLModule(source, constants), id).toEqual([]);
  }
});

it('decoders preserve readback byte slices and reject missing or partial packed words', async () => {
  const decoders = await import(
    '../../../src/gpu-spatial-analysis/point-process/point-process-decoders'
  );
  const backing = Uint32Array.of(99, 2, 3, 1, 4, 99);
  expect(
    decoders.decodeGPUContinuousSpatialSampleStatus(new Uint8Array(backing.buffer, 4, 16))
  ).toEqual({
    count: 2,
    requiredCount: 3,
    randomAcceptedCount: 1,
    flags: 4,
    complete: false,
    invalidInput: false,
    overflow: false,
    candidateOverflow: true
  });
  expect(
    decoders.decodeGPUPointPatternEnvelopeStatistics(Float32Array.of(1, 0.05, 0.05, 19))
      .simulationCount
  ).toBe(19);
  expect(() => decoders.decodeGPUPointPatternEnvelopeStatus(new Uint32Array())).toThrow(
    /at least 1/
  );
  expect(() => decoders.decodeGPUContinuousSpatialSampleStatus(new Uint8Array(15))).toThrow(
    /whole 4-byte/
  );
  expect(() => decoders.decodeGPUSpatialBalancedSampleStatus(Uint32Array.of(1))).toThrow(
    /at least 2/
  );
  expect(() =>
    decoders.decodeGPUContinuousSpatialSampleProvenance(Uint32Array.of(1, 2, 3))
  ).toThrow(/at least 4/);
});
