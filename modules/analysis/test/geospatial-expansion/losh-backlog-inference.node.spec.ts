// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULOSHResidualPermutationTest,
  GPULocalSpatialHeteroscedasticity
} from '../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let next = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${next++}`, format, length);
  const props = {
    values: view('float32', 5),
    mask: view('uint32', 5),
    weights: {
      offsets: view('uint32', 6),
      neighbors: view('uint32', 10),
      weights: view('float32', 10),
      status: {overflow: view('uint32', 1)}
    },
    parameters: view('uint32', 4),
    maximumPermutations: 99,
    maximumNeighbors: 4,
    alternative: 'greater' as const,
    multipleTesting: 'benjamini-hochberg' as const,
    output: {
      statistic: view('float32', 5),
      status: view('uint32', 5),
      normalizedResiduals: view('float32', 5),
      inference: {
        exceedances: view('uint32', 5),
        pseudoPValues: view('float32', 5),
        adjustedPValues: view('float32', 5),
        significant: view('uint32', 5),
        status: {overflow: view('uint32', 1)}
      }
    }
  };
  return {graph, props};
}

it('conditional LOSH uses shared permutation kernels with maximal bindings and no readback', () => {
  const {graph, props} = setup();
  const nodes = new GPULOSHResidualPermutationTest(props).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(5);
  for (const {source} of sources) expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  // Parse newly authored LOSH stages; the shared scan uses syntax unsupported by wgsl_reflect.
  const descriptiveNodes = new GPULOSHResidualPermutationTest({...props, id: 'parsed'})
    .getCommandNodes(graph)
    .filter(node => node.id.startsWith('parsed-descriptive'));
  expect(getParsedKernelSources(descriptiveNodes).size).toBe(4);
});

it('conditional LOSH rejects missing tails and cross-stage aliases', () => {
  const {props} = setup();
  expect(() => new GPULOSHResidualPermutationTest({...props, alternative: undefined!})).toThrow(
    'explicit'
  );
  expect(
    () =>
      new GPULOSHResidualPermutationTest({
        ...props,
        output: {
          ...props.output,
          inference: {...props.output.inference, pseudoPValues: props.output.statistic}
        }
      })
  ).toThrow();
});

it('standalone LOSH rejects normalized residual aliases with either descriptive output', () => {
  const {graph, props} = setup();
  for (const buffer of [props.output.statistic.buffer, props.output.status.buffer]) {
    const normalizedResiduals = graph.createDataView(buffer, {format: 'float32', length: 5});
    expect(
      () =>
        new GPULocalSpatialHeteroscedasticity({
          ...props,
          output: {...props.output, normalizedResiduals}
        })
    ).toThrow('outputs must not share buffers');
  }
  const status = graph.createDataView(props.output.statistic.buffer, {format: 'uint32', length: 5});
  expect(
    () =>
      new GPULocalSpatialHeteroscedasticity({
        ...props,
        output: {...props.output, status, normalizedResiduals: undefined}
      })
  ).toThrow('outputs must not share buffers');
});
