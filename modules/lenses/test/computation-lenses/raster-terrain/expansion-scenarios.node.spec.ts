// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '@luma-contributors/analysis/testing';
import {buildRasterExpansionScenario} from '../../../src/computation-lenses/scenarios/gpu-raster/expansion-runner';
import type {
  RasterLensSession,
  RasterLensFormat
} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-harness';
import {RASTER_EXPANSION_OPERATORS} from '../../../src/gpu-raster/computation-pack/operators/expansion';
import {TERRAIN_EXPANSION_OPERATORS} from '../../../src/gpu-terrain/computation-pack/operators/expansion';

it.each(
  Object.keys({...RASTER_EXPANSION_OPERATORS, ...TERRAIN_EXPANSION_OPERATORS})
)('%s constructs the actual executable scenario within core storage limits', async element => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const owned: {destroy(): void}[] = [];
  const captures: GraphDataView[] = [];
  function makeView<Format extends RasterLensFormat>(
    name: string,
    format: Format,
    values: ArrayLike<number>
  ): GraphDataView<Format> {
    const data =
      format === 'float32'
        ? Float32Array.from(values)
        : format === 'uint32'
          ? Uint32Array.from(values)
          : Int32Array.from(values);
    const buffer = device.createBuffer({
      id: name,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    owned.push(buffer);
    return graph.createDataView(
      graph.importBuffer({id: name, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format, length: data.length}
    );
  }
  const session: RasterLensSession = {
    device,
    graph,
    input(name, format, values, spec) {
      const view = makeView(name, format, values);
      if (spec !== false) captures.push(view);
      return view;
    },
    output(name, format, spec, length = 9) {
      const view = makeView(name, format, new Array(length).fill(0));
      if (spec !== false) captures.push(view);
      return view;
    },
    capture(_name, view) {
      captures.push(view);
    },
    track(resource) {
      if (resource) owned.push(resource);
    }
  };
  try {
    await buildRasterExpansionScenario(session, element, {
      width: 3,
      height: 3,
      affine: [1, 0, 0, 0, 1, 0],
      pixelInterpretation: 'area'
    });
    expect(captures.length).toBeGreaterThan(1);
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    for (const resource of owned) resource.destroy();
    device.destroy();
  }
});
