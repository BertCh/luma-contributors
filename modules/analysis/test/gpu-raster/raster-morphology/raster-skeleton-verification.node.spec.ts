// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPURasterSkeleton} from '../../../src/raster-analysis/gpu-raster-skeleton';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

// RT-134: with maximumIterations 1 the graph declares iteration 0 (thinning) and iteration 1
// (the fixed-point test). The fixed-point test must not write either thinning plane, so the
// published skeleton is the state after the budgeted iteration.
it('RT-134 the skeleton verification iteration reads the budgeted plane without writing it', () => {
  const {graph, grid, view} = createRasterContributorFixture(5, 5);
  const nodes = new GPURasterSkeleton({
    grid,
    input: view('uint32'),
    maximumIterations: 1,
    output: {skeleton: view('uint32'), status: {converged: view('uint32', 1)}}
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const phase of [0, 1]) {
    const thinning = sources.find(kernel => kernel.id.endsWith(`raster-skeleton-0-${phase}`));
    expect(thinning, `thinning phase ${phase}`).toBeDefined();
    expect(thinning!.source).toContain('var<storage, read_write> next');
    const verification = sources.find(kernel => kernel.id.endsWith(`raster-skeleton-1-${phase}`));
    expect(verification, `verification phase ${phase}`).toBeDefined();
    expect(verification!.source).not.toContain('var<storage, read_write> next');
  }
  // The publish kernel reads the plane iteration 0 phase 1 wrote.
  const lastThinning = nodes.find(node => node.id === 'raster-skeleton-0-1')!;
  const written = lastThinning.resources!.find(
    resource => resource.usage === 'storage-read-write' && 'buffer' in resource
  );
  const publish = nodes.find(node => node.id === 'raster-skeleton-publish')!;
  expect(
    publish.resources!.some(
      resource =>
        resource.usage === 'storage-read' &&
        'buffer' in resource &&
        written !== undefined &&
        'buffer' in written &&
        resource.buffer === written.buffer
    )
  ).toBe(true);
  // No verification kernel writes a resource other than the shared iteration status.
  for (const node of nodes.filter(node => /raster-skeleton-1-[01]$/.test(node.id))) {
    const writes = node.resources!.filter(resource => resource.usage === 'storage-read-write');
    expect(writes.length, node.id).toBeLessThanOrEqual(1);
  }
});
