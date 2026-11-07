// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPUSpatialPanelRegression,
  type GPUSpatialPanelRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('panel effects construct GPU nodes and preserve pooled default', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `panel-view-${serial++}`, format, length);
  const props: GPUSpatialPanelRegressionProps = {
    weights: {
      offsets: view('uint32', 6),
      neighbors: view('uint32', 10),
      weights: view('float32', 10)
    },
    periodCount: 4,
    predictorCount: 1,
    predictors: view('float32', 20),
    response: view('float32', 20),
    output: {
      table: view('float32', 12),
      summary: view('float32', 16),
      status: {code: view('uint32', 1)}
    }
  };
  const pooled = new GPUSpatialPanelRegression(props).getCommandNodes(graph);
  const within = new GPUSpatialPanelRegression({
    ...props,
    id: 'unit',
    effects: 'unit'
  }).getCommandNodes(graph);
  expect(within.length).toBe(pooled.length + 2);
  expect(within.some(node => node.id === 'unit-within')).toBe(true);
  expect(() => new GPUSpatialPanelRegression({...props, response: view('float32', 19)})).toThrow(
    /balanced/
  );
  expect(
    () =>
      new GPUSpatialPanelRegression({
        ...props,
        // @ts-expect-error Unsupported semantics must also fail at runtime.
        effects: 'time'
      })
  ).toThrow(/unsupported panel effects/);
  device.destroy();
});

it('maximal unit panel kernels parse within eight storage bindings', () => {
  for (const instrumentOrder of [1, 2] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `max-panel-${serial++}`, format, length);
    const predictorCount = instrumentOrder === 1 ? 8 : 5;
    const rows = 5120;
    const nodes = new GPUSpatialPanelRegression({
      effects: 'unit',
      instrumentOrder,
      weights: {
        offsets: view('uint32', 6),
        neighbors: view('uint32', 10),
        weights: view('float32', 10),
        status: {overflow: view('uint32', 1)}
      },
      periodCount: 1024,
      predictorCount,
      predictors: view('float32', rows * predictorCount),
      response: view('float32', rows),
      output: {
        table: view('float32', (predictorCount + 2) * 4),
        summary: view('float32', 16),
        residuals: view('float32', rows),
        status: {code: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.length).toBe(nodes.length);
    for (const {id, source} of sources) {
      const reflection = new WgslReflect(source);
      expect(reflection.storage.length, id).toBeLessThanOrEqual(8);
    }
    device.destroy();
  }
});
