// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {GPUEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/gpu-empirical-bayes-rates';
import {GPUMultivariateLocalGeary} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {findStatisticsNonFiniteConstants} from './statistics-wgsl-audit';

const kinds = ['emerging', 'rates', 'geary'] as const;
function createScene(kind: (typeof kinds)[number]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `population-${serial++}`, format, length);
  const population = view('uint32', 1);
  const values = view('float32', 4);
  const construct = (output: {
    count?: GraphDataView<'uint32'>;
    includedCount?: GraphDataView<'uint32'>;
  }) => {
    if (kind === 'rates')
      return new GPUEmpiricalBayesRates({events: values, populations: view('float32', 4), output});
    if (kind === 'geary')
      return new GPUMultivariateLocalGeary({
        values,
        variableCount: 1,
        weights: {
          offsets: view('uint32', 5),
          neighbors: view('uint32', 4),
          weights: view('float32', 4)
        },
        output: {...output, statistic: view('float32', 4), status: view('uint32', 4)}
      });
    return new GPUEmergingHotSpots({
      values,
      gridWidth: 2,
      gridHeight: 1,
      sliceCount: 2,
      parameters: view('float32', 8),
      multipleTesting: 'none',
      output: {
        ...output,
        giZScores: view('float32', 4),
        trendZ: view('float32', 2),
        trendP: view('float32', 2),
        trendS: view('sint32', 2),
        category: view('uint32', 2),
        hotSliceCount: view('uint32', 2),
        coldSliceCount: view('uint32', 2)
      }
    });
  };
  return {device, graph, population, construct};
}
for (const kind of kinds) {
  it(`${kind} includedCount and legacy count keep the same pipelines and public props`, () => {
    const record = (name: 'count' | 'includedCount') => {
      const context = createScene(kind);
      try {
        const producer = context.construct({[name]: context.population});
        expect(producer.props.output[name]).toBe(context.population);
        expect(producer.props.output[name === 'count' ? 'includedCount' : 'count']).toBeUndefined();
        const nodes = producer.getCommandNodes(context.graph);
        // The requested view must actually be written; equal source alone cannot show forwarding.
        expect(
          nodes.some(node =>
            node.resources?.some(
              resource =>
                'buffer' in resource &&
                resource.buffer === context.population &&
                resource.usage.startsWith('storage') &&
                resource.usage !== 'storage-read'
            )
          )
        ).toBe(true);
        return getGPUCommandNodeKernelSources(nodes).map(({source, constants}) => {
          expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
          expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
          return {source, constants};
        });
      } finally {
        context.device.destroy();
      }
    };
    expect(record('includedCount')).toEqual(record('count'));
  });
  it(`${kind} rejects ambiguous aliases and invalid includedCount shape at construction`, () => {
    const context = createScene(kind);
    try {
      expect(() =>
        context.construct({count: context.population, includedCount: context.population})
      ).toThrow(/output.includedCount and legacy output.count are aliases; provide only one/);
      const invalid = context.graph.createDataView(context.population.buffer, {
        format: 'float32',
        length: 1
      }) as unknown as GraphDataView<'uint32'>;
      expect(() => context.construct({includedCount: invalid})).toThrow(/includedCount/);
    } finally {
      context.device.destroy();
    }
  });
  it(`${kind} preflights includedCount STORAGE usage before any declarations`, () => {
    const context = createScene(kind);
    const buffer = context.device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST});
    try {
      const invalid = importGraphBuffer(context.graph, 'invalid-population', buffer, 'uint32', 1);
      const producer = context.construct({includedCount: invalid});
      const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
      const declare = vi.spyOn(context.graph, 'createDataView');
      expect(() => producer.getCommandNodes(context.graph)).toThrow(/STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declare).not.toHaveBeenCalled();
    } finally {
      buffer.destroy();
      context.device.destroy();
    }
  });
}
