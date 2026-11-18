// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {GPURasterReconstruction} from '../../../src/raster-analysis/gpu-raster-reconstruction';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('R2 keeps its validated eight-node budget and output ports after caller mutation', () => {
  const {graph, grid, view} = createRasterContributorFixture();
  const output = {
    values: view('float32'),
    validity: view('uint32'),
    status: {converged: view('uint32', 1)}
  };
  const originalValues = output.values;
  const props = {
    grid,
    marker: view('float32'),
    mask: view('float32'),
    maximumIterations: 1,
    output
  };
  const contributor = new GPURasterReconstruction(props);
  props.maximumIterations = 5;
  output.values = props.marker;
  expect(contributor.getCommandNodes(graph)).toHaveLength(8);
  expect(contributor.estimatedCommandNodeCount).toBe(8);
  expect(contributor.props.output.values).toBe(originalValues);
});

it('the verification round reads the budgeted plane without writing or swapping it', () => {
  const {graph, grid, view} = createRasterContributorFixture();
  const nodes = new GPURasterReconstruction({
    grid,
    marker: view('float32'),
    mask: view('float32'),
    maximumIterations: 1,
    output: {
      values: view('float32'),
      validity: view('uint32'),
      status: {converged: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  const verification = sources.find(kernel => kernel.id.endsWith('-round-1'))!;
  expect(verification).toBeDefined();
  expect(verification.source).not.toContain('var<storage, read_write> next');
  const publish = nodes.find(node => node.id.endsWith('-publish'))!;
  const round = nodes.find(node => node.id.endsWith('-round-0'))!;
  const written = round.resources!.find(resource => resource.usage === 'storage-read-write');
  expect(
    publish.resources!.some(
      resource =>
        resource.usage === 'storage-read' &&
        'buffer' in resource &&
        written &&
        'buffer' in written &&
        resource.buffer === written.buffer
    )
  ).toBe(true);
});

it('canonicalizes both reconstruction bands and preserves a constant pipeline count across rounds', () => {
  const counts = [8, 20].map(maximumIterations => {
    const {graph, grid, view, band} = createRasterContributorFixture();
    const contributor = new GPURasterReconstruction({
      grid,
      marker: band,
      mask: {...band, id: 'constraint'},
      maximumIterations,
      output: {
        values: view('float32'),
        validity: view('uint32'),
        status: {converged: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const {id, source} of sources) {
      const stripped = stripWGSLComments(source);
      expect(findReservedIdentifiers(stripped), id).toEqual([]);
      const identifiers = new Set(stripped.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []);
      expect(
        [...WGSL_RESERVED_WORDS].filter(word => identifiers.has(word)),
        id
      ).toEqual([]);
    }
    return {
      shaders: new Set(sources.map(source => source.source)).size,
      pipelines: getGPUCommandNodePipelineKeys(nodes).size
    };
  });
  expect(counts[1]).toEqual(counts[0]);
});
it('rejects bad band bindings before reconstruction declares canonicalization or loop scratch', () => {
  const {graph, grid, view, band} = createRasterContributorFixture();
  const buffer = graph.createTransientBuffer({
    id: 'copy-only',
    byteLength: 20,
    usage: Buffer.COPY_DST
  });
  const marker = {
    ...band,
    storage: {
      kind: 'buffer' as const,
      values: graph.createDataView(buffer, {format: 'uint32', length: 5})
    }
  };
  const contributor = new GPURasterReconstruction({
    grid,
    marker,
    mask: band,
    output: {
      values: view('float32'),
      validity: view('uint32'),
      status: {converged: view('uint32', 1)}
    }
  });
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  const nodes = vi.spyOn(graph, 'addComputePass');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(resources).not.toHaveBeenCalled();
  expect(nodes).not.toHaveBeenCalled();
});
