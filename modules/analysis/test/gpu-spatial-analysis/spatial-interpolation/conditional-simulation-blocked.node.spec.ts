// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';

function makeSimulation(observationCount: number, queryCount: number, replicateCount = 1) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let viewIndex = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `simulation-${viewIndex++}`, format, length);
  return {
    device,
    graph,
    props: {
      positions: view('float32x2', observationCount),
      observations: view('float32x4', observationCount),
      queries: view('float32x2', queryCount),
      covariance: {model: 'exponential' as const, range: 1.3, sill: 1},
      seed: 19,
      replicateCount,
      output: {
        values: view('float32', queryCount * replicateCount),
        moments: view('float32x2', queryCount),
        status: view('uint32', replicateCount)
      }
    }
  };
}

for (const reusePreparation of [false, true]) {
  for (const [observationCount, queryCount] of [
    [2, 3],
    [129, 129],
    [0, 129],
    [1, 129]
  ]) {
    it(`keeps every declared binding statically reachable in ${observationCount}+${queryCount} simulation, retained=${reusePreparation}`, () => {
      const fixture = makeSimulation(observationCount, queryCount);
      const contributor = new GPUConditionalGaussianSimulation({
        ...fixture.props,
        reusePreparation
      });
      try {
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(fixture.graph));
        expect(kernels.length).toBeGreaterThan(0);
        for (const {id, source} of kernels) {
          expectValidQueryShaderLiterals(source);
          const words =
            source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ??
            [];
          expect(
            words.filter(word => WGSL_RESERVED_WORDS.has(word)),
            id
          ).toEqual([]);
          const reflection = new WgslReflect(source);
          // Entry resources follow called functions, excluding references in unused helpers.
          // Checking declarations or identifier occurrences alone misses inferred-layout drops.
          const declared = [...reflection.storage, ...reflection.uniforms].map(
            binding => `${binding.group}:${binding.binding}:${binding.name}`
          );
          expect(reflection.entry.compute.length, id).toBe(1);
          const reachable = reflection.entry.compute[0].resources.map(
            binding => `${binding.group}:${binding.binding}:${binding.name}`
          );
          expect(reachable.sort(), id).toEqual(declared.sort());
        }
      } finally {
        contributor.destroy();
        fixture.device.destroy();
      }
    });
  }
}

it('simulation preserves the cooperative path and declares blocked domains above its old limits', () => {
  for (const [observations, queries, blocked] of [
    [128, 128, false],
    [129, 127, true],
    [127, 129, true],
    [2048, 2048, true]
  ] as const) {
    const fixture = makeSimulation(observations, queries);
    try {
      const nodes = new GPUConditionalGaussianSimulation(fixture.props).getCommandNodes(
        fixture.graph
      );
      expect(nodes.some(node => node.id.endsWith('blocked-realizations'))).toBe(blocked);
      expect(nodes.some(node => node.id.endsWith('cooperative-factor'))).toBe(!blocked);
    } finally {
      fixture.device.destroy();
    }
  }
});

it('simulation rejects unsupported joint domains before allocating retained GPU buffers', () => {
  const fixture = makeSimulation(2048, 2049);
  try {
    expect(() => new GPUConditionalGaussianSimulation(fixture.props)).toThrow(
      'observations plus queries must be at most 4096'
    );
  } finally {
    fixture.device.destroy();
  }
});

it('validates every compiled covariance and mean literal at construction', () => {
  const fixture = makeSimulation(129, 129);
  try {
    for (const name of ['range', 'sill', 'nugget'] as const) {
      expect(
        () =>
          new GPUConditionalGaussianSimulation({
            ...fixture.props,
            covariance: {...fixture.props.covariance, [name]: 1e100}
          })
      ).toThrow(new RegExp(`covariance.${name}.*finite`));
    }
    expect(() => new GPUConditionalGaussianSimulation({...fixture.props, mean: 1e100})).toThrow(
      /mean.*finite/
    );
    expect(
      () =>
        new GPUConditionalGaussianSimulation({
          ...fixture.props,
          covariance: {...fixture.props.covariance, range: Number.MIN_VALUE}
        })
    ).toThrow(/covariance.range.*finite/);
  } finally {
    fixture.device.destroy();
  }
});

it('simulation checks the dense factor and realization batch against the actual binding limit', () => {
  for (const [observations, queries, replicates, bindingLimit, reason] of [
    [129, 129, 1, 256 * 1024, 'joint covariance factor'],
    [1, 129, 1024, 256 * 1024, 'values chunk 0']
  ] as const) {
    const fixture = makeSimulation(observations, queries, replicates);
    fixture.device.limits.maxStorageBufferBindingSize = bindingLimit;
    try {
      const contributor = new GPUConditionalGaussianSimulation(fixture.props);
      expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(reason);
    } finally {
      fixture.device.destroy();
    }
  }
});

it('reuses blocked factorization pipelines as block counts grow, with exact per-block dispatches', () => {
  const pipelineCounts: number[] = [];
  const shaderCounts: number[] = [];
  for (const observationCount of [129, 257, 513]) {
    const fixture = makeSimulation(observationCount, 129);
    try {
      const allNodes = new GPUConditionalGaussianSimulation(fixture.props).getCommandNodes(
        fixture.graph
      );
      const nodes = allNodes.filter(node =>
        [
          'blocked-diagonal',
          'blocked-panel',
          'blocked-update',
          'blocked-whiten-block',
          'blocked-whiten-update'
        ].includes(node.workload?.variant ?? '')
      );
      const sources = getGPUCommandNodeKernelSources(nodes);
      pipelineCounts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
      shaderCounts.push(new Set(sources.map(({source}) => source)).size);
      const capacity = observationCount + 129;
      expect(
        nodes
          .filter(node => node.workload?.variant === 'blocked-panel')
          .map(node => node.workload?.maximumWorkgroupCount)
      ).toEqual(
        Array.from({length: Math.ceil(capacity / 16) - 1}, (_, index) =>
          Math.ceil((capacity - index * 16 - 16) / 64)
        )
      );
      for (const {constants} of sources) {
        expect(constants).toEqual({});
      }
      for (const {source} of getGPUCommandNodeKernelSources(allNodes)) {
        expectValidQueryShaderLiterals(source);
        const words =
          source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
    } finally {
      fixture.device.destroy();
    }
  }
  expect(new Set(pipelineCounts).size).toBe(1);
  expect(new Set(shaderCounts).size).toBe(1);
});

for (const reusePreparation of [false, true]) {
  it(`keeps every lane alive until cooperative blocked-update barriers, retained=${reusePreparation}`, () => {
    const fixture = makeSimulation(129, 129);
    const contributor = new GPUConditionalGaussianSimulation({...fixture.props, reusePreparation});
    try {
      const nodes = contributor.getCommandNodes(fixture.graph);
      const updates = nodes.filter(node => node.workload?.variant === 'blocked-update');
      expect(updates.length).toBeGreaterThan(1);
      for (const {source, constants} of getGPUCommandNodeKernelSources(updates)) {
        const main = source.slice(source.indexOf('fn main('));
        const beforeSynchronization = main.slice(0, main.indexOf('workgroupUniformLoad('));
        expect(beforeSynchronization).not.toMatch(/if\s*\(index\s*>=/);
        expect(beforeSynchronization).toContain('if (workgroupIndex >= getChunkParameter(1u))');
        expect(source).toContain('workgroupBarrier()');
        expect(constants).toEqual({});
        expectValidQueryShaderLiterals(source);
        const words =
          source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
      expect(new Set(getGPUCommandNodeKernelSources(updates).map(({source}) => source)).size).toBe(
        1
      );
    } finally {
      contributor.destroy();
      fixture.device.destroy();
    }
  });
}

for (const [limit, value] of [
  ['maxUniformBufferBindingSize', 128],
  ['maxComputeWorkgroupSizeX', 128],
  ['maxStorageBuffersPerShaderStage', 5]
] as const) {
  it(`prechecks blocked ${limit} before graph or persistent allocations`, () => {
    const fixture = makeSimulation(129, 129);
    fixture.device.limits[limit] = value;
    const contributor = new GPUConditionalGaussianSimulation(fixture.props);
    const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
    const createView = vi.spyOn(fixture.graph, 'createDataView');
    const createBuffer = vi.spyOn(fixture.device, 'createBuffer');
    try {
      expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(
        /chunk uniform|workgroupSize|storage bindings/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(createBuffer).not.toHaveBeenCalled();
    } finally {
      vi.restoreAllMocks();
      fixture.device.destroy();
    }
  });
}
