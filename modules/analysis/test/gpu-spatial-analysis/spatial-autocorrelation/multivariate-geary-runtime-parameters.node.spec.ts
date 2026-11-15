// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUMultivariateLocalGearyPermutationTest,
  getGPUMultivariateLocalGearyPermutationTestParameterValues,
  type GPUMultivariateLocalGearyPermutationTestSettings
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

import {createSpatialStatisticsParameterNodes} from '../../../src/analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  computeMultivariateGearyRuntimeOracle,
  MULTIVARIATE_GEARY_RUNTIME_SEED
} from './multivariate-geary-runtime-oracle';

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `geary-runtime-${serial++}`, format, length);
  return {
    device,
    graph,
    view,
    props: {
      values: view('float32', 16),
      variableCount: 2,
      maximumPermutations: 19,
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 16),
        weights: view('float32', 16)
      },
      output: {
        statistic: view('float32', 8),
        pValues: view('float32', 8),
        classes: view('uint32', 8)
      }
    }
  };
}

for (const strategy of ['exact', 'tiled'] as const) {
  for (const multipleTesting of ['none', 'benjamini-hochberg'] as const) {
    it(`${strategy} ${multipleTesting} Geary seed and level changes do not change its pipelines`, () => {
      const sources = (
        seed: readonly [number, number],
        significance: number,
        runtime = false,
        chunked = false
      ) => {
        const context = setup();
        try {
          const nodes = new GPUMultivariateLocalGearyPermutationTest({
            ...context.props,
            strategy,
            multipleTesting,
            seed,
            significance,
            rowTileSize: 3,
            permutationTileSize: 7,
            weights: chunked
              ? {rowCount: 8, chunks: [{rowStart: 0, rowCount: 8, weights: context.props.weights}]}
              : context.props.weights,
            parameters: runtime ? context.view('uint32', 3) : undefined
          }).getCommandNodes(context.graph);
          const kernels = getGPUCommandNodeKernelSources(nodes);
          for (const kernel of kernels)
            expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
          expect(
            getParsedKernelSources(nodes.filter(node => !node.id.includes('-multiple-testing-')))
              .size
          ).toBeGreaterThan(0);
          expect(nodes.filter(node => node.type === 'copy')).toHaveLength(runtime ? 0 : 1);
          return kernels.map(kernel => kernel.source);
        } finally {
          context.device.destroy();
        }
      };
      for (const chunked of [false, true]) {
        expect(sources([0, 0], 0.05, false, chunked)).toEqual(
          sources([0xffffffff, 0x80000000], 0.25, false, chunked)
        );
        expect(sources([0, 0], 0.05, false, chunked)).toEqual(
          sources([9, 11], 0.75, true, chunked)
        );
      }
    });
  }
}

it('packs every Philox key bit and level bits while preserving target padding and failed writes', () => {
  expect([...getGPUMultivariateLocalGearyPermutationTestParameterValues()]).toEqual([
    0, 0, 0x3d4ccccd
  ]);
  const target = new Uint32Array([0, 0, 0, 0xdeadbeef]);
  expect(
    getGPUMultivariateLocalGearyPermutationTestParameterValues(
      {seed: [0xffffffff, 0x80000001], significance: 0.5},
      target
    )
  ).toBe(target);
  expect([...target]).toEqual([0xffffffff, 0x80000001, 0x3f000000, 0xdeadbeef]);
  const invalid: GPUMultivariateLocalGearyPermutationTestSettings[] = [
    ...[-1, 1.5, NaN, Infinity, 2 ** 32].map(value => ({seed: [value, 0] as const})),
    ...[0, 1, -1, NaN, Infinity].map(significance => ({significance}))
  ];
  for (const settings of invalid) {
    expect(() =>
      getGPUMultivariateLocalGearyPermutationTestParameterValues(settings, target)
    ).toThrow(
      /^getGPUMultivariateLocalGearyPermutationTestParameterValues .*seed|^getGPUMultivariateLocalGearyPermutationTestParameterValues .*significance/
    );
    expect([...target]).toEqual([0xffffffff, 0x80000001, 0x3f000000, 0xdeadbeef]);
  }
  expect(() =>
    getGPUMultivariateLocalGearyPermutationTestParameterValues({}, new Uint32Array(2))
  ).toThrow(/target.length/);
});

for (const strategy of ['exact', 'tiled'] as const) {
  for (const multipleTesting of ['none', 'benjamini-hochberg'] as const) {
    it(`${strategy} ${multipleTesting} validates parameter views before generating any nodes`, () => {
      const context = setup();
      try {
        const construct = (parameters: GraphDataView<'uint32'>) =>
          new GPUMultivariateLocalGearyPermutationTest({
            ...context.props,
            strategy,
            multipleTesting,
            parameters
          });
        for (const parameters of [
          context.view('uint32', 2),
          context.view('float32', 3) as unknown as GraphDataView<'uint32'>
        ]) {
          expect(() => construct(parameters)).toThrow(/parameters/);
        }
        for (const output of [context.props.output.classes, context.props.output.statistic]) {
          const alias = context.graph.createDataView(output.buffer, {format: 'uint32', length: 3});
          expect(() => construct(alias)).toThrow(/share buffers/);
        }
        const foreignGraph = new GPUCommandGraph(context.device);
        const foreign = createTransientView(foreignGraph, 'foreign-geary-parameters', 'uint32', 3);
        const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
        expect(() => construct(foreign).getCommandNodes(context.graph)).toThrow(/target graph/);
        expect(allocate).not.toHaveBeenCalled();
        allocate.mockRestore();
        const oversized = context.view('uint32', 17);
        Object.assign(context.device.limits, {maxStorageBufferBindingSize: 64});
        const preflight = vi.spyOn(context.graph, 'createTransientBuffer');
        expect(() => construct(oversized).getCommandNodes(context.graph)).toThrow(
          /parameters binding.*limit 64/
        );
        expect(preflight).not.toHaveBeenCalled();
      } finally {
        context.device.destroy();
      }
    });
  }
}

it('word defaults copy on every encoding and release the owned GPUParameterBuffer', () => {
  const {device, graph} = setup();
  try {
    const values = getGPUMultivariateLocalGearyPermutationTestParameterValues({
      seed: [0xffffffff, 0x80000001]
    });
    const runtime = createSpatialStatisticsParameterNodes(graph, 'word-defaults', values);
    expect(runtime.parameters.format).toBe('uint32');
    const executable = runtime.nodes[0].compile({device});
    const destination = device.createBuffer({byteLength: values.byteLength});
    const copyBufferToBuffer = vi.fn();
    const context = {commandEncoder: {copyBufferToBuffer}, getBuffer: () => destination};
    executable.encode(context as never);
    executable.encode(context as never);
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(2);
    const source = copyBufferToBuffer.mock.calls[0][0].sourceBuffer;
    const destroy = vi.spyOn(source, 'destroy');
    executable.destroy?.();
    expect(destroy).toHaveBeenCalledOnce();
    expect(
      createSpatialStatisticsParameterNodes(graph, 'borrowed-words', runtime.parameters).nodes
    ).toEqual([]);
    destination.destroy();
  } finally {
    device.destroy();
  }
});

it('degree-one oracle has a hand-derived variance and a discriminating corrected significance case', () => {
  const low = computeMultivariateGearyRuntimeOracle(
    MULTIVARIATE_GEARY_RUNTIME_SEED,
    0.01,
    'benjamini-hochberg'
  );
  const high = computeMultivariateGearyRuntimeOracle(
    MULTIVARIATE_GEARY_RUNTIME_SEED,
    0.99,
    'benjamini-hochberg'
  );
  expect(low.statistic.slice(0, 5)).toEqual([1, 4, 16, 64, 225].map(value => value / 117.25));
  expect(low.classes.slice(0, 5)).toEqual([0, 0, 0, 0, 0]);
  expect(high.classes.some(value => value === 1 || value === 2)).toBe(true);
  expect(high.classes.slice(5)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
});
