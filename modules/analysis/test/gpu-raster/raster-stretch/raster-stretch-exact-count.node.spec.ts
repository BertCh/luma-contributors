// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPURasterStretch} from '../../../src/raster-analysis/raster-stretch/gpu-raster-stretch';
import {getGPURasterStretchParameterValues} from '../../../src/raster-analysis/raster-stretch/raster-stretch-parameters';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('emits an exact uint32 stretch count without changing the legacy float statistics layout', () => {
  const {graph, grid, view} = createRasterContributorFixture(4097, 4097);
  const contributor = new GPURasterStretch({
    grid,
    values: view('float32'),
    parameters: view('float32', 16),
    binCount: 16,
    output: {validCount: view('uint32', 1), statistics: view('float32', 8)}
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  expect(
    sources.some(({source}) => source.includes('validCountOut[validCountOutOffset] = count'))
  ).toBe(true);
  expect(
    sources.some(({source}) => source.includes('let validCount = keys[keysOffset + 2u]'))
  ).toBe(true);
  for (const {id, source} of sources)
    expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
});
it('rejects legacy nodata sentinels that would emit non-finite float32 WGSL literals', () => {
  const {grid, view} = createRasterContributorFixture();
  expect(
    () =>
      new GPURasterStretch({
        grid,
        values: view('float32'),
        noDataValue: 1e100,
        parameters: view('float32', 16),
        output: {validCount: view('uint32', 1)}
      })
  ).toThrow(/noDataValue.*finite/);
});

it('uses uniform tile loops before barriers and exponent-scaled quotients for both histogram routes', () => {
  for (const binCount of [16, 65536]) {
    const {graph, grid, view} = createRasterContributorFixture(257);
    const nodes = new GPURasterStretch({
      grid,
      values: view('float32'),
      parameters: view('float32', 16),
      binCount,
      output: {stretched: view('float32')}
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const {id, source} of sources) {
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      if (
        id === 'raster-stretch-reduce' ||
        (binCount === 16 && id === 'raster-stretch-histogram')
      ) {
        expect(source, id).toContain('workgroupBarrier()');
        expect(source, id).not.toMatch(/\bbreak\s*;/);
        expect(source, id).toContain('if (cell < CELL_COUNT)');
      }
      if (id === 'raster-stretch-histogram' || id === 'raster-stretch-apply') {
        expect(source, id).toContain('fn getScaledQuotient');
        expect(source, id).toContain('frexp(');
        expect(source, id).toContain('ldexp(');
      }
    }
  }
});

it('maps extreme finite domains to the closed-form zero, half and one outputs', () => {
  for (const [minimum, maximum] of [
    [0, 2 ** 127],
    [0, 2 ** -125],
    [-(2 ** 127), 2 ** 127]
  ]) {
    const {graph, grid, view} = createRasterContributorFixture(3);
    const values = view('float32');
    const parameters = view('float32', 16);
    const stretched = view('float32');
    const contributor = new GPURasterStretch({grid, values, parameters, output: {stretched}});
    const apply = contributor
      .getCommandNodes(graph)
      .find(node => node.id === 'raster-stretch-apply')!;
    const state = apply
      .resources!.map(resource => ('buffer' in resource ? resource.buffer : undefined))
      .find(
        buffer => buffer instanceof GraphDataView && buffer.buffer.id === 'raster-stretch-state'
      ) as GraphDataView<'float32'>;
    const memory = createCPUGraphMemory();
    memory.write(values, Float32Array.of(minimum, (minimum + maximum) / 2, maximum));
    memory.write(parameters, getGPURasterStretchParameterValues());
    memory.write(state, Float32Array.of(minimum, maximum, 0, 0, minimum, maximum, 0, 0, 0, 0));
    executeCommandNodesOnCPU([apply], memory);
    expect(Array.from(memory.readFloat32(stretched, 3))).toEqual([0, 0.5, 1]);
  }
});
