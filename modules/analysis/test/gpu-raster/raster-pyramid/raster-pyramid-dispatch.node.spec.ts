// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('uses one fused pyramid pipeline as levels grow and dispatches each batch at its own workgroup count', async () => {
  const {GPURasterExtremaPyramid, getGPURasterExtremaPyramidLayout} = await import(
    '../../../src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid'
  );
  const counts = [3, 7, 8].map(levelCount => {
    const {graph, grid, view} = createRasterContributorFixture(128, 128);
    const layout = getGPURasterExtremaPyramidLayout(grid, {firstBlockSize: 1, levelCount});
    const contributor = new GPURasterExtremaPyramid({
      grid,
      firstBlockSize: 1,
      levelCount,
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32')},
        validity: view('uint32')
      },
      output: {combined: view('float32', layout.length * 2)}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const reductions = nodes.filter(node => node.id.includes('-levels-'));
    expect(getGPUCommandNodePipelineKeys(reductions).size).toBe(1);
    expect(reductions[0].workload?.maximumWorkgroupCount).toBe(64);
    if (reductions.length > 1) expect(reductions[1].workload?.maximumWorkgroupCount).toBe(1);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const {id, source} of sources)
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
    return {
      sources: new Set(sources.map(source => source.source)).size,
      pipelines: getGPUCommandNodePipelineKeys(nodes).size
    };
  });
  expect(counts[1]).toEqual(counts[0]);
  expect(counts[2]).toEqual(counts[0]);
});
it('checks the complete pyramid parameter allocation before declaring any node or resource', async () => {
  const {GPURasterExtremaPyramid, getGPURasterExtremaPyramidLayout} = await import(
    '../../../src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid'
  );
  const {graph, grid, view} = createRasterContributorFixture(4, 4);
  const layout = getGPURasterExtremaPyramidLayout(grid, {firstBlockSize: 1});
  const contributor = new GPURasterExtremaPyramid({
    grid,
    firstBlockSize: 1,
    input: {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32')},
      validity: view('uint32')
    },
    output: {maximum: view('float32', layout.length)}
  });
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxBufferSize: 128}
  });
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  const views = vi.spyOn(graph, 'createDataView');
  const nodes = vi.spyOn(graph, 'addComputePass');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /fused parameter slot.*maxBufferSize.*128/
  );
  expect(resources).not.toHaveBeenCalled();
  expect(views).not.toHaveBeenCalled();
  expect(nodes).not.toHaveBeenCalled();
});
