// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import typescript from 'typescript';
import {resolve} from 'node:path';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  createVirtualProgram,
  getRepositoryCompilerOptions,
  formatDiagnostic
} from '../analysis-core/doc-program';
import {entries as rasterBandsIndicesEntries} from '../../../../website/src/catalog/entries/raster-bands-indices';
import {entries as rasterEmbeddingsFlowEntries} from '../../../../website/src/catalog/entries/raster-embeddings-flow';

vi.mock('../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));
vi.mock('../../../../website/src/gpu/runs/inputs', () => ({
  createImageryBands() {},
  createSmoothField() {}
}));
vi.mock('../../../../website/src/gpu/runs/entries/raster-bands-indices/support/bands', () => ({
  drawRaster() {}
}));
const SCENE_MODULE =
  '../../../../website/src/gpu/runs/entries/raster-bands-indices/make-raster-band';

it('builds the live raster-band calibration example through the producer API', async () => {
  type SceneBuffer = {view: GraphDataView};
  type Builder = {
    graph: GPUCommandGraph;
    upload(id: string, format: 'float32', values: Float32Array): SceneBuffer;
    allocate(id: string, format: 'float32' | 'uint32', length: number): SceneBuffer;
  };
  const specification = (await import(SCENE_MODULE)).default as {
    build(
      builder: Builder,
      inputs: {
        values: Float32Array;
        columns: number;
        rows: number;
        greenDigital: Float32Array;
        nearInfraredDigital: Float32Array;
      },
      settings: {scale: number; offset: number; noDataValue: number}
    ): {outputs: {raw: SceneBuffer; output: SceneBuffer}};
  };
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const addPass = vi.spyOn(graph, 'addComputePass');
  try {
    const built = specification.build(
      {
        graph,
        upload: (id, format, values) => ({
          view: createTransientView(graph, id, format, values.length)
        }),
        allocate: (id, format, length) => ({view: createTransientView(graph, id, format, length)})
      },
      {
        values: Float32Array.of(0),
        columns: 1,
        rows: 1,
        greenDigital: Float32Array.of(10000),
        nearInfraredDigital: Float32Array.of(20000)
      },
      {scale: 2.75, offset: -0.2, noDataValue: 1}
    );
    expect(built.outputs.raw.view.length).toBe(1);
    expect(built.outputs.output.view.length).toBe(1);
    expect(addPass).toHaveBeenCalledTimes(2);
  } finally {
    addPass.mockRestore();
    device.destroy();
  }
});

it('typechecks all six generated catalog examples against public raster exports', () => {
  const symbols = new Set([
    'GPURasterNDWI',
    'GPURasterNBR',
    'GPURasterEVI',
    'GPURasterSAVI',
    'GPURasterEmbeddingKMeans',
    'GPURasterEmbeddingNearestPixels'
  ]);
  const entries = [...rasterBandsIndicesEntries, ...rasterEmbeddingsFlowEntries].filter(entry =>
    symbols.has(entry.symbol)
  );
  expect(entries).toHaveLength(6);
  const sources = new Map(
    entries.map(entry => [
      resolve(import.meta.dirname, `raster-api-${entry.symbol}.virtual.ts`),
      `import type {Device} from '@luma.gl/core';\ndeclare const device: Device;\n${entry.example}`
    ])
  );
  const program = createVirtualProgram(sources, getRepositoryCompilerOptions());
  const diagnostics = typescript
    .getPreEmitDiagnostics(program)
    .filter(diagnostic => diagnostic.file && sources.has(diagnostic.file.fileName));
  expect(diagnostics.map(formatDiagnostic)).toEqual([]);
  for (const entry of entries) expect(entry.example).toContain(`graph.add(new ${entry.symbol}(`);
});
