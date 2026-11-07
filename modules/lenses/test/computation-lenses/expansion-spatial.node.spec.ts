// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '@luma-contributors/analysis/testing';
import {buildExpansionScenario} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/expansion-examples';
import {EXPANSION_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/expansion';
import type {S2Uploads} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';

it.each(
  Object.keys(EXPANSION_OPERATORS)
)('%s builds its real capturable graph within core binding limits', async element => {
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
    const captures = await buildExpansionScenario(element, graph, uploads);
    expect(Object.keys(captures).length).toBeGreaterThan(0);
    expect(Object.keys(captures).some(name => /^output-\d+$/.test(name))).toBe(false);
    expect(new Set(Object.values(captures).map(capture => capture.view)).size).toBe(
      Object.keys(captures).length
    );
    for (const capture of Object.values(captures)) expect(capture.view.length).toBeGreaterThan(0);
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    for (const resource of owned) resource.destroy();
    device.destroy();
  }
});
