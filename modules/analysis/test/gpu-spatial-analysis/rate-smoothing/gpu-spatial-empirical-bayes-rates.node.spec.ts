// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('R9 retains the validated loss buffer when the caller replaces status before lowering', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const original = view('original-overflow', 'uint32', 1);
  const weights = {
    offsets: view('offsets', 'uint32', 3),
    neighbors: view('neighbors', 'uint32', 2),
    weights: view('weights', 'float32', 2),
    status: {overflow: original}
  };
  const contributor = new GPUSpatialEmpiricalBayesRates({
    events: view('events', 'float32', 2),
    populations: view('populations', 'float32', 2),
    weights,
    output: {smoothedRates: view('out', 'float32', 2)}
  });
  weights.status = {} as typeof weights.status;
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toContain('spatial-empirical-bayes-rates-incomplete-0');
  const guard = nodes.find(node => node.id.endsWith('-incomplete-0'))!;
  expect(
    guard.resources?.some(resource => 'buffer' in resource && resource.buffer === original)
  ).toBe(true);
});

it('GPUSpatialEmpiricalBayesRates validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'seb-validation'});
  const events = createTransientView(graph, 'events', 'float32', 4);
  const populations = createTransientView(graph, 'populations', 'float32', 4);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
    weights: createTransientView(graph, 'weights', 'float32', 6)
  };
  const output = createTransientView(graph, 'out', 'float32', 4);
  expect(
    () => new GPUSpatialEmpiricalBayesRates({events, populations, weights, output: {}})
  ).toThrow(/no output/);
  expect(
    () =>
      new GPUSpatialEmpiricalBayesRates({
        events,
        populations,
        weights: {...weights, offsets: createTransientView(graph, 'short', 'uint32', 4)},
        output: {spatialRates: output}
      })
  ).toThrow(/row count/);
  expect(
    () =>
      new GPUSpatialEmpiricalBayesRates({
        events,
        populations,
        weights,
        output: {smoothedRates: createTransientView(graph, 'bad', 'float32', 3)}
      })
  ).toThrow(/length/);
  const nodes = new GPUSpatialEmpiricalBayesRates({
    events,
    populations,
    weights,
    output: {smoothedRates: output}
  }).getCommandNodes(graph);
  expect(nodes.length).toBe(1);
});

it('GPUSpatialEmpiricalBayesRates accepts uint32 counts with one conversion node per input', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'seb-uint32'});
  const nodes = new GPUSpatialEmpiricalBayesRates({
    events: createTransientView(graph, 'events', 'uint32', 4),
    populations: createTransientView(graph, 'populations', 'float32', 4),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6)
    },
    output: {smoothedRates: createTransientView(graph, 'out', 'float32', 4)}
  }).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'spatial-empirical-bayes-rates-events-to-float32',
    'spatial-empirical-bayes-rates-smooth'
  ]);
});

it('GPUSpatialEmpiricalBayesRates suppresses every dependent output for either incomplete weights flag', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'seb-incomplete'});
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const nodes = new GPUSpatialEmpiricalBayesRates({
    events: view('events', 'float32', 8),
    populations: view('populations', 'float32', 8),
    weights: {
      offsets: view('offsets', 'uint32', 9),
      neighbors: view('neighbors', 'uint32', 12),
      weights: view('weights', 'float32', 12),
      status: {
        overflow: view('overflow', 'uint32', 1),
        candidateOverflow: view('candidate', 'uint32', 1)
      }
    },
    output: {
      spatialRates: view('spatial', 'float32', 8),
      smoothedRates: view('smoothed', 'float32', 8),
      priorMeans: view('means', 'float32', 8),
      priorVariances: view('variances', 'float32', 8)
    }
  }).getCommandNodes(graph);
  const suppression = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
    kernel.source.includes('loss[lossOffset] != 0u')
  );
  expect(suppression).toHaveLength(2);
  for (const kernel of suppression) {
    for (const name of ['spatialRates', 'smoothedRates', 'priorMeans', 'priorVariances']) {
      expect(kernel.source).toContain(`${name}[${name}Offset + index] = getQuietNaN(index);`);
    }
  }
});
