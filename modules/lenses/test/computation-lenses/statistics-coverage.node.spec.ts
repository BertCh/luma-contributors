// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '@luma-contributors/graph/testing';
import {auditWGSLModule} from '@luma-contributors/graph/test/analysis-core/kernel-audit/kernel-audit-wgsl';
import type {S2Uploads} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';

import {buildStatisticsCoverageScenario} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/statistics-coverage-examples';

const STATISTICS_LENS_BUILDERS = [
  {element: 'GPUSpatialPanelLagMaximumLikelihood'},
  {element: 'GPUFuzzyContiguity'},
  {element: 'GPURankMarkov'},
  {element: 'GPUSpatialPanelErrorMaximumLikelihood'},
  {element: 'GPURestrictedRegimeRegression'},
  {element: 'GPUInstrumentalVariablesCovariance'},
  {element: 'GPUIslandAttachment'},
  {element: 'GPUSpatialInteraction'},
  {element: 'GPUMultivariateLocalGearyPermutationTest'},
  {element: 'GPUSeeminglyUnrelatedRegression'},
  {element: 'GPUSpatialCrossValidation'},
  {element: 'GPUSpatialCrossValidatedOLS'},
  {element: 'GPUProbitRegression'},
  {element: 'GPULocalCollinearity'}
];

it.each(STATISTICS_LENS_BUILDERS)('$element builds real capturable kernels', async ({element}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const owned: {destroy(): void}[] = [];
  const uploads: S2Uploads = {
    own(resource) {
      owned.push(resource);
      return resource;
    },
    view(id, data, format, length) {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
      return graph.createDataView(handle, {format, length: length ?? data.length / components});
    }
  };
  try {
    const captures = await buildStatisticsCoverageScenario(element, graph, uploads);
    expect(Object.values(captures).some(capture => capture.role === 'output')).toBe(true);
    expect(new Set(Object.values(captures).map(capture => capture.view)).size).toBe(
      Object.keys(captures).length
    );
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels)
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
  } finally {
    for (const resource of owned) resource.destroy();
    device.destroy();
  }
});
