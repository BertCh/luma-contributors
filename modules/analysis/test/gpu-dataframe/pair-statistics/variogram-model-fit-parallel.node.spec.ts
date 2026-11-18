// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {getParallelVariogramModelFitNodes} from '../../../src/dataframe-analysis/pair-statistics/variogram-model-fit-nodes';
import {fitVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {
  auditWGSLModule,
  getWGSLModuleUsage
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('noisy replay fixture distinguishes Cressie reweighting from pair weighting', () => {
  const distances = Float32Array.from({length: 32}, (_, index) => (index + 1) / 8);
  const pairCounts = Uint32Array.from({length: 32}, (_, index) => 10 + (index % 4) * 50);
  for (const scale of [1, 1.4]) {
    const semivariances = Float32Array.from(
      distances,
      (distance, index) =>
        scale * (0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2)) + 0.2 * Math.sin(index * 1.7))
    );
    const input = {distances, semivariances, pairCounts};
    const cressie = fitVariogramModel(input, {model: 'exponential', weighting: 'cressie'});
    const pairs = fitVariogramModel(input, {model: 'exponential', weighting: 'pairs'});
    expect(Math.abs(cressie.range - pairs.range)).toBeGreaterThan(0.1);
    expect(cressie.residual).toBeGreaterThan(0);
  }
});

it('parallel variogram search fits core binding and shader budgets at maximum search options', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const weighting of ['pairs', 'none', 'cressie'] as const) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'float32x4' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `parallel-${serial++}`, format, length);
      const contributor = new GPUVariogramModelFit({
        distances: view('float32', 256),
        semivariances: view('float32', 256),
        pairCounts: view('uint32', 256),
        weighting,
        rangeSteps: 1024,
        maximumIterations: 64,
        output: {
          parameters: view('float32x4', 3),
          diagnostics: view('float32x4', 3),
          status: view('uint32', 3)
        }
      });
      const nodes = getParallelVariogramModelFitNodes(graph, contributor);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(kernels.length).toBeGreaterThan(1);
      for (const kernel of kernels) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        const usage = getWGSLModuleUsage(kernel.source, kernel.constants);
        expect(usage.storageBuffers).toBeLessThanOrEqual(8);
        expect(usage.privateBytes).toBeLessThanOrEqual(8192);
        expect(usage.functionBytes).toBeLessThanOrEqual(8192);
        expect(usage.workgroupStorageBytes).toBeLessThanOrEqual(16384);
      }
    }
  } finally {
    device.destroy();
  }
});

it('variogram Cressie rounds reuse the candidate and refinement pipelines', () => {
  const device = createNullWebGPUDevice();
  try {
    const build = (weighting: 'pairs' | 'cressie') => {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32' | 'float32x4'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `rounds-${serial++}`, format, length);
      return getGPUCommandNodeKernelSources(
        new GPUVariogramModelFit({
          distances: view('float32', 16),
          semivariances: view('float32', 16),
          pairCounts: view('uint32', 16),
          weighting,
          output: {
            parameters: view('float32x4', 3),
            diagnostics: view('float32x4', 3),
            status: view('uint32', 3)
          }
        }).getCommandNodes(graph)
      );
    };
    const single = build('pairs');
    const repeated = build('cressie');
    expect(repeated.length).toBeGreaterThan(single.length);
    const key = (kernel: (typeof repeated)[number]) =>
      `${kernel.source}\n${JSON.stringify(kernel.constants)}\n${JSON.stringify(kernel.shaderLayout)}`;
    expect(new Set(repeated.map(key))).toEqual(new Set(single.map(key)));
    for (const kernel of repeated)
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    for (const step of ['candidates', 'refine']) {
      expect(
        new Set(repeated.filter(kernel => kernel.id.includes(`-${step}-`)).map(key)).size
      ).toBe(1);
    }
  } finally {
    device.destroy();
  }
});

it('variogram fit rejects incorrect formats before generating shader bindings', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32' | 'float32x4'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `formats-${serial++}`, format, length);
    const props = {
      distances: view('float32', 8),
      semivariances: view('float32', 8),
      pairCounts: view('uint32', 8),
      output: {
        parameters: view('float32x4', 3),
        diagnostics: view('float32x4', 3),
        status: view('uint32', 3)
      }
    };
    expect(
      () => new GPUVariogramModelFit({...props, distances: view('uint32', 8) as never})
    ).toThrow(/distances.*float32/);
    expect(
      () => new GPUVariogramModelFit({...props, pairCounts: view('float32', 8) as never})
    ).toThrow(/pairCounts.*uint32/);
    expect(
      () =>
        new GPUVariogramModelFit({
          ...props,
          output: {...props.output, parameters: view('float32', 12) as never}
        })
    ).toThrow(/output.parameters.*float32x4/);
    expect(() => new GPUVariogramModelFit({...props, rangeSteps: 1025})).toThrow(
      /rangeSteps.*1024/
    );
    expect(() => new GPUVariogramModelFit({...props, maximumIterations: 65})).toThrow(
      /maximumIterations.*64/
    );
  } finally {
    device.destroy();
  }
});

it('variogram fit preflights aligned inputs and search scratch before allocating resources', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const maximumBindingSize of [32, 2048]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32' | 'float32x4'>(
        format: Format,
        length: number
      ) =>
        createTransientView(graph, `preflight-${maximumBindingSize}-${serial++}`, format, length);
      const producer = new GPUVariogramModelFit({
        distances: view('float32', 16),
        semivariances: view('float32', 16),
        pairCounts: view('uint32', 16),
        output: {
          parameters: view('float32x4', 3),
          diagnostics: view('float32x4', 3),
          status: view('uint32', 3)
        }
      });
      Object.assign(device.limits, {maxStorageBufferBindingSize: maximumBindingSize});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => producer.getCommandNodes(graph)).toThrow(
        maximumBindingSize === 32 ? /distances.*64 bytes/ : /search scratch/
      );
      expect(allocate).not.toHaveBeenCalled();
      Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    }
  } finally {
    device.destroy();
  }
});
