// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type ComputePass} from '@luma.gl/core';
import {type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createReviewScenario, REVIEW_CASES} from './independent-row-review.test-utils';
import {expectValidShaderIdentifiers} from './chunk-shader-validation.test-utils';

it('checks identifiers against the canonical full WGSL reserved-word list', () => {
  for (const identifier of ['target', 'active', 'filter', 'module', 'meta', 'reinterpret_cast']) {
    expect(() => expectValidShaderIdentifiers([`fn ${identifier}() {}`])).toThrow();
  }
  expectValidShaderIdentifiers(['fn sample() {}']);
});

for (const kind of REVIEW_CASES) {
  it(`${kind} namespaces two contributors' nodes and scratch in one graph`, () => {
    const device = createNullWebGPUDevice();
    const scenario = createReviewScenario(device, kind, [0, 513, 0, 1, 257, 0]);
    const allocations = vi.spyOn(scenario.graph, 'createTransientBuffer');
    try {
      const identifiers = new Set<string>();
      for (const id of ['first-contributor', 'second-contributor']) {
        const start = allocations.mock.calls.length;
        const contributor = scenario.createContributor(id);
        const nodes = contributor.getCommandNodes(scenario.graph);
        expect(nodes.length).toBeGreaterThan(0);
        for (const node of nodes) {
          expect(node.id.startsWith(`${id}-`)).toBe(true);
          expect(identifiers.has(node.id)).toBe(false);
          identifiers.add(node.id);
          scenario.graph.add(node);
        }
        for (const [descriptor] of allocations.mock.calls.slice(start)) {
          expect(descriptor.id.startsWith(`${id}-`)).toBe(true);
          expect(identifiers.has(descriptor.id)).toBe(false);
          identifiers.add(descriptor.id);
        }
      }
    } finally {
      scenario.destroy();
      device.destroy();
    }
  });

  it(`${kind} accepts adjacent shared-buffer outputs and dispatches each chunk at its own workgroup count`, () => {
    const keys = new Set<string>();
    const sources = new Set<string>();
    for (const lengths of [
      [513],
      [513, 1, 257],
      Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
      [0, 255, 0, 256, 257, 0],
      [1, 3, 5]
    ]) {
      const device = createNullWebGPUDevice();
      Object.defineProperty(device, 'props', {
        value: {...device.props, _cacheShaders: false, _cachePipelines: false}
      });
      vi.spyOn(device, 'createComputePipeline').mockImplementation(
        props =>
          ({id: props.id ?? '', hash: '', destroy: () => {}}) as ReturnType<
            typeof device.createComputePipeline
          >
      );
      const scenario = createReviewScenario(device, kind, lengths);
      const nodes = scenario.contributor.getCommandNodes(scenario.graph);
      try {
        getParsedKernelSources(nodes);
        for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
          keys.add(key);
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          sources.add(kernel.source);
          if (kernel.source.includes('graphChunkGroupParameters'))
            expect(kernel.constants).toEqual({});
        }
        const allocations: Buffer[] = [];
        const buffers = new Map<GraphDataView['buffer'], Buffer>();
        const getBuffer = (view: GraphDataView) => {
          if (view.buffer.defaultBuffer) return view.buffer.defaultBuffer as Buffer;
          let buffer = buffers.get(view.buffer);
          if (!buffer) {
            buffer = device.createBuffer({
              byteLength: view.buffer.byteLength,
              usage: Buffer.STORAGE
            });
            buffers.set(view.buffer, buffer);
            allocations.push(buffer);
          }
          return buffer;
        };
        const rowNodes = nodes.filter(node => /-chunk-\d+$/.test(node.id));
        const copiesPerChunk = kind.startsWith('forest') ? (kind === 'forest-raster' ? 4 : 2) : 1;
        const expectedRows = lengths
          .filter(length => length > 0)
          .flatMap(length => Array<number>(copiesPerChunk).fill(length));
        expect(rowNodes.map(node => node.workload?.maximumInvocationCount)).toEqual(expectedRows);
        expect(rowNodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
          expectedRows.map(rows => Math.ceil(rows / 256))
        );
        const dispatches: number[][] = [];
        const computePass = {
          setPipeline: () => {},
          setBindings: () => {},
          dispatch: (x: number, y: number, z: number) => dispatches.push([x, y, z])
        } as unknown as ComputePass;
        for (const node of rowNodes) {
          if (node.type !== 'compute') throw new Error('expected compute node');
          const compiled = node.compile({device});
          compiled.encode({computePass, getBuffer} as never);
          compiled.destroy?.();
        }
        expect(dispatches).toEqual(expectedRows.map(rows => [Math.ceil(rows / 256), 1, 1]));
        for (const buffer of allocations) buffer.destroy();
      } finally {
        scenario.destroy();
        device.destroy();
      }
    }
    expect(keys.size).toBe(
      kind.startsWith('forest')
        ? kind === 'forest-raster'
          ? 3
          : 2
        : ['calendar', 'period', 'keys'].includes(kind)
          ? 2
          : 1
    );
    expect(sources.size).toBe(keys.size);
    expectValidShaderIdentifiers(sources);
  });

  it(`${kind} checks uniform/device support before graph resources or nodes are declared`, () => {
    const device = createNullWebGPUDevice();
    const scenario = createReviewScenario(device, kind, [255, 256, 257]);
    Object.defineProperty(device.limits, 'maxUniformBufferBindingSize', {value: 128});
    const createBuffer = vi.spyOn(scenario.graph, 'createTransientBuffer');
    const createView = vi.spyOn(scenario.graph, 'createDataView');
    const add = vi.spyOn(scenario.graph, 'add');
    expect(() => scenario.contributor.getCommandNodes(scenario.graph)).toThrow(
      /chunk uniform byteLength.*128/
    );
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    scenario.destroy();
    device.destroy();
  });
}
