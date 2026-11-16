// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {getPanelWithinNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-within-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `f-reg-${serial++}`, format, length);
  const sur = (rows: number) => ({
    id: 'test-sur',
    rowCount: rows,
    equationCount: 2,
    predictorCount: 1,
    predictors: view('float32', rows * 2),
    responses: view('float32', rows * 2),
    output: {
      coefficients: view('float32', 4),
      covariance: view('float32', 16),
      residualCovariance: view('float32', 4),
      residuals: view('float32', rows * 2),
      status: view('uint32', 1)
    }
  });
  const panel = (periods: number) => ({
    id: 'test-panel',
    periodCount: periods,
    weights: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    },
    predictorCount: 1,
    predictors: view('float32', periods * 2),
    response: view('float32', periods * 2),
    output: {
      table: view('float32', 12),
      summary: view('float32', 22),
      residuals: view('float32', periods * 2),
      status: {code: view('uint32', 1), converged: view('uint32', 1)}
    }
  });
  return {device, graph, view, sur, panel};
}

it('tiles long-panel means with a constant shader count and audits every generated identifier', () => {
  for (const periodCount of [8, 1025, 32768]) {
    const context = createContext();
    try {
      const props = context.panel(periodCount);
      expect(() => new GPUSpatialPanelErrorMaximumLikelihood(props)).not.toThrow();
      const nodes = getPanelWithinNodes(context.graph, {
        ...props,
        entityCount: 2,
        output: {
          predictors: context.view('float32', periodCount * 2),
          response: context.view('float32', periodCount * 2)
        }
      });
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(new Set(sources.map(source => source.source)).size).toBe(4);
      expect(new Set(getGPUCommandNodePipelineKeys(nodes)).size).toBe(4);
      for (const source of sources) {
        // Run the same reflection parser used by the integration coverage audit. Its legacy
        // scanner still treats some valid WGSL identifiers (for example block) as tokens.
        const reflection = new WgslReflect(source.source);
        expect(reflection.storage.length, source.id).toBeLessThanOrEqual(8);
        expect(auditWGSLModule(source.source, source.constants), source.id).toEqual([]);
      }
    } finally {
      context.device.destroy();
    }
  }
});

it('preflights expanded panel CSR before allocating any scratch', () => {
  const context = createContext();
  try {
    const props = context.panel(1025);
    props.weights.neighbors = context.view('uint32', 1000);
    props.weights.weights = context.view('float32', 1000);
    Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {value: 65536});
    const allocation = vi.spyOn(context.graph, 'createTransientBuffer');
    expect(() =>
      new GPUSpatialPanelErrorMaximumLikelihood(props).getCommandNodes(context.graph)
    ).toThrow(/expanded neighbors/);
    expect(allocation).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});

it('preflights nested trace dots before declaring any graph resources', () => {
  const context = createContext();
  try {
    const contributor = new GPUSpatialPanelErrorMaximumLikelihood({
      ...context.panel(2),
      traceMode: 'stochastic',
      traceProbeCount: 131073,
      maximumPower: 256
    });
    const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
    const createView = vi.spyOn(context.graph, 'createDataView');
    const addNode = vi.spyOn(context.graph, 'add');
    expect(() => contributor.getCommandNodes(context.graph)).toThrow(
      /trace probe dots.*134283264 bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(addNode).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});
