// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {getSpaceTimeHawkesEventWGSL} from '../../../src/gpu-spatial-analysis/space-time-hawkes/space-time-hawkes-wgsl';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {computeSpaceTimeHawkesOnCPU} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createOutputBuffer,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

function createFixture(maximumIterations = 1, oversizedParameters = false) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameterBuffer = createTransientView(graph, 'parameter-buffer', 'float32', 128);
  const contributor = new GPUSpaceTimeHawkes({
    id: 'hawkes',
    maximumIterations,
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    times: createTransientView(graph, 'times', 'float32', 4),
    parameters: oversizedParameters
      ? graph.createDataView(parameterBuffer.buffer, {
          format: 'float32',
          byteOffset: 252,
          length: 9
        })
      : createTransientView(graph, 'parameters', 'float32', 9),
    output: {estimates: createTransientView(graph, 'estimates', 'float32', 4)}
  });
  return {device, graph, contributor};
}

it('caps only strictly earlier parents, independently of the size of a timestamp tie', () => {
  const source = getSpaceTimeHawkesEventWGSL(1);
  expect(source).toContain('let parentEnd = low;');
  expect(source).toContain('if (parentEnd - low > MAXIMUM_PARENTS)');
  expect(source).toContain('parent < parentEnd;');
  expect(source).not.toContain('row - low > MAXIMUM_PARENTS');
  expect(source).toContain('getScaledQuotient(distanceSquared, variance, 1.0, -1)');
  expect(source).not.toContain('0.5 / variance');
  expect(source).not.toContain('6.2831853 * variance');
});

it('checks every Hawkes binding before adding any resources or nodes', () => {
  for (const [oversizedParameters, limit, message] of [
    [true, 256, /hawkes parameters needs 288 bytes/],
    [false, 64, /hawkes state needs 68 bytes/]
  ] as const) {
    const {device, graph, contributor} = createFixture(1, oversizedParameters);
    try {
      Object.defineProperty(device, 'limits', {
        value: {...device.limits, maxStorageBufferBindingSize: limit}
      });
      const declarations = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
      const before = [declarations.buffers.size, declarations.nodes.length];
      expect(() => contributor.getCommandNodes(graph)).toThrow(message);
      expect([declarations.buffers.size, declarations.nodes.length]).toEqual(before);
    } finally {
      device.destroy();
    }
  }
});

it('keeps the number of shader sources and pipelines constant as EM rounds grow', () => {
  const counts: number[][] = [];
  for (const maximumIterations of [2, 7]) {
    const {device, graph, contributor} = createFixture(maximumIterations);
    try {
      const nodes = contributor.getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      counts.push([
        new Set(kernels.map(kernel => kernel.source)).size,
        getGPUCommandNodePipelineKeys(nodes).size
      ]);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      for (const kernel of kernels.filter(kernel => kernel.operation === 'GPUSpaceTimeHawkes')) {
        const source = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
        expect(
          (source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
            WGSL_RESERVED_WORDS.has(identifier)
          )
        ).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
  expect(counts[0]).toEqual(counts[1]);
});

it.each([[4, Infinity]])('rejects runtime parameter %i = %s in the CPU oracle', (slot, value) => {
  const parameters = new Float32Array([2, 10, 10, 10, 1e-6, 0.1, 0.5, 1, 1]);
  parameters[slot] = value;
  const result = computeSpaceTimeHawkesOnCPU(
    {
      positions: new Float32Array([0, 0, 1, 0]),
      times: new Float32Array([1, 2]),
      parameters
    },
    {maximumIterations: 1}
  );
  expect(result.iterationCount[0]).toBe(0);
  expect(result.estimates.every(Number.isNaN)).toBe(true);
});

it.each([
  1e30, 1e-30
])('preserves the float64 CPU reference variance range for deviation %s', deviation => {
  const parameters = new Float32Array([2, 10, 10, 10, 1e-6, 0.1, 0.5, 1, deviation]);
  const result = computeSpaceTimeHawkesOnCPU(
    {
      positions: new Float32Array([0, 0, 0, 0]),
      times: new Float32Array([2, 2]),
      parameters
    },
    {maximumIterations: 1}
  );
  expect(result.iterationCount[0]).toBe(1);
  expect(result.estimates[3]).toBe(parameters[8]);
  expect(Array.from(result.backgroundProbability)).toEqual([1, 1]);
});

it('rejects overlapping output views of one imported physical buffer', () => {
  const {device, graph, contributor} = createFixture();
  const buffer = createOutputBuffer(device, 64);
  try {
    const estimates = importGraphBuffer(graph, 'alias-estimates', buffer, 'float32', 4);
    const logLikelihood = graph.createDataView(estimates.buffer, {format: 'float32', length: 1});
    expect(
      () => new GPUSpaceTimeHawkes({...contributor.props, output: {estimates, logLikelihood}})
    ).toThrow(/hawkes .*overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('rejects a non-storage Hawkes parameter buffer without declaring scratch', () => {
  const {device, graph, contributor} = createFixture();
  const buffer = device.createBuffer({byteLength: 36, usage: Buffer.COPY_DST});
  try {
    const parameters = importGraphBuffer(graph, 'non-storage', buffer, 'float32', 9);
    const invalid = new GPUSpaceTimeHawkes({...contributor.props, parameters});
    const declarations = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = [declarations.buffers.size, declarations.nodes.length];
    expect(() => invalid.getCommandNodes(graph)).toThrow(/parameters.*Buffer.STORAGE/);
    expect([declarations.buffers.size, declarations.nodes.length]).toEqual(before);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it.each([
  [4, Infinity],
  [8, 1e30],
  [8, 1e-30]
])('clears invalid fit state and probabilities for parameter %i = %s', (slot, value) => {
  const {device, graph, contributor} = createFixture();
  try {
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const status = {
      converged: scalar('converged'),
      iterationLimitReached: scalar('limit'),
      iterationCount: scalar('iterations'),
      maximumIterations: scalar('budget'),
      candidateOverflow: scalar('cut'),
      invalidCount: scalar('invalid'),
      approximation: scalar('approximation')
    };
    const backgroundProbability = createTransientView(graph, 'probabilities', 'float32', 4);
    recordWGSLKernelNodes(graph);
    const nodes = new GPUSpaceTimeHawkes({
      ...contributor.props,
      output: {...contributor.props.output, backgroundProbability, status}
    }).getCommandNodes(graph);
    const initialize = nodes.find(node => node.id === 'hawkes-initialize')!;
    const record = getWGSLKernelNodeRecord(graph, initialize.id)!;
    const state = record.bindings.find(binding => binding.name === 'state')!.view;
    const memory = createCPUGraphMemory();
    memory.write(state, new Float32Array(17).fill(7));
    const parameters = new Float32Array([2, 10, 10, 10, 1e-6, 0.1, 0.5, 1, 1]);
    parameters[slot] = value;
    memory.write(contributor.props.parameters, parameters);
    const responsibilities = nodes.find(node => node.id === 'hawkes-responsibilities')!;
    const probabilityRecord = getWGSLKernelNodeRecord(graph, responsibilities.id)!;
    memory.write(
      probabilityRecord.bindings.find(binding => binding.name === 'events')!.view,
      new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 3, 0])
    );
    memory.write(
      probabilityRecord.bindings.find(binding => binding.name === 'timeOrder')!.view,
      new Uint32Array([0, 1, 2, 3])
    );
    memory.write(backgroundProbability, new Float32Array(4).fill(0.5));
    executeCommandNodesOnCPU(
      [initialize, responsibilities, nodes.find(node => node.id === 'hawkes-publish-status')!],
      memory
    );
    expect(Array.from(memory.readFloat32(state).slice(4, 12))).toEqual(new Array(8).fill(0));
    expect(memory.readFloat32(state)[12]).toBe(1);
    for (const name of [
      'converged',
      'iterationLimitReached',
      'iterationCount',
      'candidateOverflow'
    ] as const)
      expect(memory.readUint32(status[name])[0]).toBe(0);
    expect(memory.readUint32(status.invalidCount)[0]).toBe(4);
    expect(memory.readUint32(status.approximation)[0]).toBe(1);
    // Budget is construction metadata, even for an unavailable fit.
    executeCommandNodesOnCPU([nodes.find(node => node.id === 'hawkes-publish-budget')!], memory);
    expect(memory.readUint32(status.maximumIterations)[0]).toBe(1);
    expect(Array.from(memory.readFloat32(backgroundProbability)).every(Number.isNaN)).toBe(true);
  } finally {
    device.destroy();
  }
});
