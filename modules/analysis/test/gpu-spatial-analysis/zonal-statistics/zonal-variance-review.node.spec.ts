// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {computeZonalStatistics} from './zonal-statistics-oracle';

it('atomic public sums use stable tree reductions for the two variance passes (B4)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const pointCount = 500000;
  const contributor = new GPUZonalStatistics({
    id: 'z',
    sumOrder: 'atomic',
    features: {
      kind: 'feature-rows',
      featureCount: 17,
      pointFeatureRows: createTransientView(graph, 'rows', 'uint32', pointCount)
    },
    values: createTransientView(graph, 'values', 'float32', pointCount),
    output: {variances: createTransientView(graph, 'variances', 'float32', 17)}
  });
  const nodes = contributor.getCommandNodes(graph);
  const identifiers = nodes.map(node => node.id);
  expect(identifiers).toContain('z-mean-refinement-reduce-first-linear-deviation');
  expect(identifiers).toContain('z-variance-reduce-deviation');
  expect(identifiers).toContain('z-variance-reduce-linear-deviation');
  expect(
    identifiers.some(identifier => identifier.startsWith('z-first-linear-deviation-aggregation'))
  ).toBe(false);
});

it('weighted variance distinguishes signed weights before applying its nonnegative roundoff clamp', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const scalar = (name: string) => createTransientView(graph, name, 'float32', 1);
  const contributor = new GPUZonalStatistics({
    id: 'signed',
    features: {
      kind: 'feature-rows',
      featureCount: 1,
      pointFeatureRows: createTransientView(graph, 'rows', 'uint32', 4)
    },
    values: createTransientView(graph, 'values', 'float32', 4),
    weights: createTransientView(graph, 'weights', 'float32', 4),
    output: {variances: scalar('variances'), standardDeviations: scalar('deviations')}
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const variance = sources.find(source => source.id === 'signed-variances')!.source;
  expect(variance).toContain('minimumWeights[minimumWeightsOffset + index] >= 0.0');
  expect(new WgslReflect(variance).storage.length).toBeLessThanOrEqual(8);
  // Hand calculation: weighted mean -1, squared sum -30, V1=5, V2=13,
  // denominator 5-13/5=2.4; -30/2.4=-12.5, whose real square root is undefined.
  const reference = computeZonalStatistics({
    featureRows: [0, 0, 0, 0],
    featureCount: 1,
    values: [0, 0, 0, 5],
    weights: [2, 2, 2, -1]
  });
  expect(reference.variances).toEqual([-12.5]);
  expect(reference.standardDeviations[0]).toBeNaN();
});
