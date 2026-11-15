// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  getGPUSpatialWeightsTransposeParameterValues,
  GPUSpatialWeightsTranspose,
  GPUChunkedSpatialWeightsTranspose
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

type TransposeLayout = 'packed' | 'chunked-input' | 'chunked-output';

function createTranspose(
  context: ReturnType<typeof setup>,
  layout: TransposeLayout,
  symmetryTolerance: number | GraphDataView<'float32'>,
  symmetry = true
) {
  const output = {
    ...context.output,
    asymmetricSlots: symmetry ? context.output.asymmetricSlots : undefined
  };
  const chunkedOutput = {
    ...context.chunkedOutput,
    asymmetricSlots: symmetry ? context.chunkedOutput.asymmetricSlots : undefined
  };
  return layout === 'packed'
    ? new GPUSpatialWeightsTranspose({
        id: 'transpose-parameters',
        weights: context.weights,
        output,
        symmetryTolerance
      })
    : layout === 'chunked-input'
      ? new GPUChunkedSpatialWeightsTranspose({
          id: 'transpose-parameters',
          weights: context.chunkedWeights,
          output,
          symmetryTolerance
        })
      : new GPUChunkedSpatialWeightsTranspose({
          id: 'transpose-parameters',
          weights: context.chunkedWeights,
          output: chunkedOutput,
          symmetryTolerance
        });
}

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `transpose-runtime-${serial++}`, format, length);
  const createWeights = (rows: number, capacity: number) => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity)
  });
  const weights = createWeights(4, 8);
  const chunkedWeights = {
    rowCount: 4,
    chunks: [
      {rowStart: 0, rowCount: 2, weights: createWeights(2, 4)},
      {rowStart: 2, rowCount: 2, weights: createWeights(2, 4)}
    ]
  };
  const output = {...createWeights(4, 8), asymmetricSlots: view('uint32', 1)};
  const chunkedOutput = {
    chunks: [0, 2].map(rowStart => ({
      rowStart,
      rowCount: 2,
      weights: {
        ...createWeights(2, 2),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    })),
    asymmetricSlots: view('uint32', 1)
  };
  return {device, graph, view, weights, chunkedWeights, output, chunkedOutput};
}

for (const layout of ['packed', 'chunked-input', 'chunked-output'] as const) {
  it(`${layout} transpose tolerance changes do not rebuild its compute pipelines`, () => {
    const sources = (setting: number | 'runtime') => {
      const context = setup();
      try {
        const producer = createTranspose(
          context,
          layout,
          setting === 'runtime' ? context.view('float32', 1) : setting
        );
        const nodes = producer.getCommandNodes(context.graph);
        expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        }
        if ('estimatedCommandNodeCount' in producer) {
          expect(nodes.length).toBeLessThanOrEqual(producer.estimatedCommandNodeCount);
        }
        return getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source);
      } finally {
        context.device.destroy();
      }
    };
    expect(sources(0)).toEqual(sources(0.125));
    expect(sources(0)).toEqual(sources('runtime'));
  });
}

it('transpose packer validates float32 tolerances without changing target padding on success or failure', () => {
  expect([...getGPUSpatialWeightsTransposeParameterValues()]).toEqual([0]);
  const target = new Float32Array([7, 99]);
  expect(getGPUSpatialWeightsTransposeParameterValues({symmetryTolerance: 0.125}, target)).toBe(
    target
  );
  expect([...target]).toEqual([0.125, 99]);
  for (const symmetryTolerance of [-1, NaN, Infinity, 1e40]) {
    expect(() => getGPUSpatialWeightsTransposeParameterValues({symmetryTolerance}, target)).toThrow(
      /^getGPUSpatialWeightsTransposeParameterValues .*symmetryTolerance/
    );
    expect([...target]).toEqual([0.125, 99]);
  }
  expect(() => getGPUSpatialWeightsTransposeParameterValues({}, new Float32Array(0))).toThrow(
    /target.length/
  );
});

for (const layout of ['packed', 'chunked-input', 'chunked-output'] as const) {
  it(`${layout} transpose rejects malformed and aliased parameter views at construction`, () => {
    const context = setup();
    try {
      for (const symmetryTolerance of [
        -1,
        NaN,
        Infinity,
        1e40,
        context.view('float32', 0),
        context.view('uint32', 1) as unknown as GraphDataView<'float32'>
      ]) {
        expect(() => createTranspose(context, layout, symmetryTolerance)).toThrow(
          /^transpose-parameters .*symmetryTolerance/
        );
      }
      const outputWeights =
        layout === 'chunked-output' ? context.chunkedOutput.chunks[0].weights : context.output;
      const aliased = context.graph.createDataView(outputWeights.weights.buffer, {
        format: 'float32',
        length: 1
      });
      expect(() => createTranspose(context, layout, aliased)).toThrow(/share buffers/);
      const diagnostic =
        layout === 'chunked-output'
          ? context.chunkedOutput.asymmetricSlots
          : context.output.asymmetricSlots;
      const aliasedDiagnostic = context.graph.createDataView(diagnostic.buffer, {
        format: 'float32',
        length: 1
      });
      expect(() => createTranspose(context, layout, aliasedDiagnostic)).toThrow(/share buffers/);
    } finally {
      context.device.destroy();
    }
  });

  it(`${layout} transpose preflights parameter ownership and binding size before allocating scratch`, () => {
    for (const failure of ['ownership', 'binding-size'] as const) {
      const context = setup();
      const otherGraph = new GPUCommandGraph(context.device);
      try {
        const tolerance =
          failure === 'ownership'
            ? createTransientView(otherGraph, 'foreign-tolerance', 'float32', 1)
            : context.view('float32', 17);
        const producer = createTranspose(context, layout, tolerance);
        if (failure === 'binding-size')
          Object.assign(context.device.limits, {maxStorageBufferBindingSize: 64});
        const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
        expect(() => producer.getCommandNodes(context.graph)).toThrow(
          failure === 'ownership' ? /target graph/ : /limit 64/
        );
        expect(allocate).not.toHaveBeenCalled();
      } finally {
        context.device.destroy();
      }
    }
  });

  it(`${layout} transpose allocates one numeric-default upload only when symmetry is requested`, () => {
    for (const symmetry of [false, true]) {
      for (const runtime of [false, true]) {
        const context = setup();
        try {
          const producer = createTranspose(
            context,
            layout,
            runtime ? context.view('float32', 1) : 0.125,
            symmetry
          );
          const nodes = producer.getCommandNodes(context.graph);
          expect(nodes.filter(node => node.type === 'copy')).toHaveLength(
            symmetry && !runtime ? 1 : 0
          );
          if ('estimatedCommandNodeCount' in producer)
            expect(nodes.length).toBeLessThanOrEqual(producer.estimatedCommandNodeCount);
        } finally {
          context.device.destroy();
        }
      }
    }
  });
}
