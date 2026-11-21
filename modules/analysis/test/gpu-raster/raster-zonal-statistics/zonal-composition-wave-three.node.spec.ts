// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getRasterProducerCompletenessNodes} from '../../../src/raster-analysis/raster-zonal-statistics/raster-producer-completeness';
import {GPURasterZonalStatistics} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {computeZonalStatistics} from './raster-zonal-statistics-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('CMP-09 ORs each loss source and recognizes a view from another module condition', () => {
  for (const flags of [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
  ]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    recordWGSLKernelNodes(graph);
    const sources = flags.map((_, index) =>
      createTransientView(graph, `loss-${index}`, 'uint32', 1)
    );
    const incomplete = createTransientView(graph, 'incomplete', 'uint32', 1);
    const mean = createTransientView(graph, 'mean', 'float32', 1);
    // Shape and resource identity survive module boundaries; constructor identity does not.
    const foreignMean = new Proxy(mean, {getPrototypeOf: () => Object.prototype});
    const nodes = getRasterProducerCompletenessNodes(
      graph,
      'guard',
      sources[0],
      incomplete,
      {means: foreignMean},
      sources.slice(1)
    );
    const result = new Float32Array([7]);
    const status = new Uint32Array([99]);
    const resources = new Map<unknown, Uint32Array>(
      sources.map((view, index) => [view, new Uint32Array([flags[index]])])
    );
    resources.set(incomplete, status);
    resources.set(foreignMean, new Uint32Array(result.buffer));
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      const record = getWGSLKernelNodeRecord(graph, kernel.id)!;
      const bindings: Record<number, Uint32Array> = {};
      record.bindings.forEach((binding, index) => {
        const values = resources.get(binding.view);
        expect(values, `unmapped resource ${binding.view.buffer.id}`).toBeDefined();
        bindings[index] = values!;
      });
      new WgslExec(
        new WgslParser().parse(
          kernel.source
            .replace(/override (\w+): u32;/g, 'const $1: u32 = 1u;')
            .replace(/^\s*_ = .*;$/gm, '')
        )
      ).dispatchWorkgroups('main', 1, {0: bindings});
    }
    const lost = flags.some(Boolean);
    expect(Array.from(status)).toEqual([Number(lost)]);
    if (lost) expect(Number.isNaN(result[0])).toBe(true);
    else expect(result[0]).toBe(7);
  }
});

it('CMP-09 requires an incomplete status only when a loss source is supplied', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = {
    grid: {
      width: 1,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    },
    zoneCapacity: 1,
    zones: createTransientView(graph, 'zones', 'uint32', 1),
    values: {
      id: 'values',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: createTransientView(graph, 'values', 'float32', 1)}
    },
    incompleteSources: [createTransientView(graph, 'unfinished-cost-solve', 'uint32', 1)],
    output: {means: createTransientView(graph, 'means', 'float32', 1)}
  };
  expect(() => new GPURasterZonalStatistics(props)).toThrow(/incomplete/);
  expect(() => new GPURasterZonalStatistics({...props, incompleteSources: []})).not.toThrow();
});

it('CMP-10 polygon row zero survives a NO_ZONE background handoff', () => {
  const result = computeZonalStatistics([0, 0xffffffff], [7, 42], {zoneCapacity: 1});
  expect(result.means).toEqual([7]);
  expect(result.valueCounts).toEqual([1]);
  expect(result.overflow).toBe(0);
});

it('URT-03 population variance uses total weight rather than the sample divisor', () => {
  // /usr/bin/python3: R=6371007.180918475; w=R**2*pi/180*diff(sin(deg2rad([0,30,60])))[::-1]
  // np.cov([1,3], aweights=w, ddof=0/1) => .9760677434251697 / 2.
  for (const sumOrder of ['sorted', 'atomic'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = 1) =>
      createTransientView(graph, name, format, length);
    const nodes = new GPURasterZonalStatistics({
      grid: {width: 1, height: 2, affine: [1, 0, 0, 0, -30, 60], pixelInterpretation: 'area'},
      zones: view('zones', 'uint32', 2),
      zoneCapacity: 1,
      cellSizeMode: 'geographic',
      sumOrder,
      parameters: view('settings', 'float32', 7),
      values: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: view('values', 'float32', 2)}
      },
      output: {
        populationVariances: view('population', 'float32'),
        variances: view('sample', 'float32')
      }
    }).getCommandNodes(graph);
    const kernel = getGPUCommandNodeKernelSources(nodes).find(source =>
      source.id.endsWith('-variances')
    )!;
    const population = new Uint32Array(1);
    const sample = new Uint32Array(1);
    const inputs: Record<string, Uint32Array | Float32Array> = {
      valueCounts: new Uint32Array([2]),
      squaredDeviationSums: new Float32Array([0]), // replaced below with literal reference moment
      linearDeviationSums: new Float32Array([0]),
      weightSums: new Float32Array([0]),
      weightSquareSums: new Float32Array([0]),
      variances: sample,
      populationVariances: population
    };
    const weights = [259301354372.38928, 354212237308.39484];
    const total = weights[0] + weights[1];
    inputs['weightSums'][0] = total;
    inputs['weightSquareSums'][0] = weights[0] ** 2 + weights[1] ** 2;
    inputs['squaredDeviationSums'][0] = total * 0.9760677434251697;
    const bindings: Record<number, Uint32Array | Float32Array> = {};
    for (const match of kernel.source.matchAll(/@binding\((\d+)\) var<storage, \w+> (\w+):/g))
      bindings[Number(match[1])] = inputs[match[2]];
    const source = kernel.source
      .replace(/override (\w+): u32;/g, 'const $1: u32 = 1u;')
      .replace(/^\s*_ = .*;$/gm, '');
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: bindings});
    const populationValue = new Float32Array(population.buffer)[0];
    const sampleValue = new Float32Array(sample.buffer)[0];
    expect(Number.isFinite(populationValue)).toBe(true);
    expect(populationValue).toBeCloseTo(0.9760677434251697, 6);
    expect(Number.isFinite(sampleValue)).toBe(true);
    expect(sampleValue).toBeCloseTo(2, 5);
  }
});
