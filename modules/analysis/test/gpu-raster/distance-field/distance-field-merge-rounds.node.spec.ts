// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('shrinks exact-distance merge dispatches by round while retaining one shader and pipeline', async () => {
  const {GPUDistanceField} = await import(
    '../../../src/raster-analysis/distance-field/gpu-distance-field'
  );
  for (const width of [1025, 4097]) {
    const {graph, grid, view} = createRasterContributorFixture(width, 33);
    const contributor = new GPUDistanceField({
      grid,
      parameters: view('float32', 8),
      seedMask: view('uint32'),
      output: {distances: view('float32')}
    });
    const nodes = contributor
      .getCommandNodes(graph)
      .filter(node => /-row-merge-\d+$/.test(node.id));
    expect(nodes.length).toBeGreaterThan(1);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(new Set(sources.map(source => source.source)).size).toBe(1);
    expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(1);
    expect(nodes.at(-1)?.workload?.maximumInvocationCount).toBe(33);
    expect(nodes.at(-1)?.workload?.maximumWorkgroupCount).toBe(1);
    expect(nodes[0].workload!.maximumWorkgroupCount).toBeGreaterThan(1);
    for (const {id, source} of sources)
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
  }
});
