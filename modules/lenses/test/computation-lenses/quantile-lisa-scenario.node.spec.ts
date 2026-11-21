// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {LensCommandGraph} from '../../src/computation-lenses/lens-view-snapshot';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import * as runs from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/quantile-lisa.lens';

test('real Quantile LISA scenario declares the same legacy alternative it records', async () => {
  const device = createNullWebGPUDevice();
  const graph = new LensCommandGraph<undefined>(device);
  const resources: {destroy(): void}[] = [];
  const boundary = new Error('declaration complete; no GPU submission');
  const add = vi.spyOn(graph, 'add');
  // Stop at the runner boundary while exercising the real scenario closure and builder.
  const run = vi.spyOn(runs, 'runS2Lens').mockImplementation(async (_device, props) => {
    expect(props.choices?.['alternative']).toBe('geoda-legacy');
    expect(props.recordChoices).toContainEqual(
      expect.objectContaining({
        name: 'alternative',
        value: 'geoda-legacy'
      })
    );
    // Exercise the runner's allowed asynchronous builder and resource-ownership contract.
    const build: typeof props.build = async (buildGraph, uploads) => {
      uploads.own({destroy() {}});
      return props.build(buildGraph, uploads);
    };
    const uploads: runs.S2Uploads = {
      view<Format extends GPUVectorFormat>(
        name: string,
        values: Float32Array | Uint32Array | Int32Array,
        format: Format,
        length?: number
      ) {
        const buffer = graph.createTransientBuffer({
          id: name,
          byteLength: values.byteLength,
          usage: Buffer.STORAGE | Buffer.COPY_SRC
        });
        const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
        return graph.createDataView(buffer, {format, length: length ?? values.length / components});
      },
      own(resource) {
        resources.push(resource);
        return resource;
      }
    };
    const captures = await build(graph, uploads);
    expect(captures['pseudo-p-values'].role).toBe('output');
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        props: expect.objectContaining({alternative: 'geoda-legacy'})
      })
    );
    throw boundary;
  });
  try {
    const scenario = getLensScenario('gpu-spatial-analysis.quantile-lisa')!;
    await expect((await scenario.load()).run(device)).rejects.toBe(boundary);
    expect(run).toHaveBeenCalledOnce();
    expect(resources).toHaveLength(1);
  } finally {
    run.mockRestore();
    add.mockRestore();
    for (const resource of resources) resource.destroy();
    device.destroy();
  }
});
