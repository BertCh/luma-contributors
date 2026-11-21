// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUInequality} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {createInequalityLorenzPrefixNodes} from '../../../src/dataframe-analysis/composite-indicators/inequality-lorenz-prefix';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('weighted Lorenz hierarchy keeps four pipelines as prefix levels grow', () => {
  const build = (rows: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format) =>
      createTransientView(graph, `prefix-${serial++}`, format, rows);
    try {
      const {nodes} = createInequalityLorenzPrefixNodes(graph, {
        id: 'lorenz',
        values: view('float32'),
        weights: view('float32'),
        sortedRows: view('uint32'),
        sortedZones: view('uint32'),
        zoneCount: 5
      });
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      expect(nodes[0].workload?.maximumInvocationCount).toBe(Math.ceil(rows / 256));
      const keys = getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
      expect(keys.size).toBe(4);
      return keys;
    } finally {
      device.destroy();
    }
  };
  expect(build(1)).toEqual(build(257));
  expect(build(257)).toEqual(build(65537));
});

it.each([
  {rows: 1, mode: 'parallel'},
  {rows: 257, mode: 'parallel'},
  {rows: 65537, mode: 'parallel'},
  {rows: 257, mode: 'reference'}
] as const)('inequality audits all generated identifiers: %j', ({rows, mode}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `inequality-${serial++}`, format, length);
  try {
    const output = {
      gini: view('float32', 3),
      atkinson: view('float32', 3),
      mean: view('float32', 3),
      lorenzKnots: view('float32', 33),
      counts: view('uint32', 3),
      globalSummary: view('float32', 8)
    };
    const contributor = new GPUInequality({
      mode,
      values: view('float32', rows),
      weights: view('float32', rows),
      zoneIds: view('uint32', rows),
      zoneCount: 3,
      parameters: view('float32', 4),
      output
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    expect(contributor.mode).toBe(mode);
    expect(nodes.some(node => node.id === 'inequality-zone-gini-publish')).toBe(true);
    expect(nodes.some(node => node.id === 'inequality-global-gini-publish')).toBe(true);
  } finally {
    device.destroy();
  }
});
