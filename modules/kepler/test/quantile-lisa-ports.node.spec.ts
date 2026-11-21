// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../analysis/src/testing/gpu-contributor-test-utils';
import {GPUQuantileLISA} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {addKeplerPointQuantileLISA} from '../src/analysis/point-quantile-lisa';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import {getKeplerPointWeightSpatialContext} from '../src/analysis/point-weight-coordinates';
import type {KeplerPointWeightsBinding} from '../src/analysis/point-weights';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const source = {datasetId: 'six-points', revision: 1};
  const weightsSettings = {
    kind: 'knn' as const,
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    k: 1,
    coordinateSpace: 'longitude-latitude' as const,
    metric: 'great-circle' as const,
    units: 'meters' as const
  };
  const status = {
    overflow: view('weights-overflow', 'uint32', 1),
    requiredCount: view('weights-required', 'uint32', 1),
    invalidCount: view('weights-invalid', 'uint32', 1)
  };
  const candidateOverflow = view('candidate-overflow', 'uint32', 1);
  const pointWeights: KeplerPointWeightsBinding = {
    source,
    settings: weightsSettings,
    status,
    coordinateMask: view('coordinates', 'uint32', 6),
    weights: {
      offsets: view('offsets', 'uint32', 7),
      neighbors: view('neighbors', 'uint32', 6),
      weights: view('weights', 'float32', 6),
      status: {...status, candidateOverflow},
      spatialContext: getKeplerPointWeightSpatialContext(weightsSettings),
      information: {
        rowCount: 6,
        transform: 'binary',
        hasSelfWeights: false,
        symmetric: true,
        provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: 1}
      }
    }
  };
  return {
    device,
    graph,
    view,
    source,
    pointWeights,
    candidateOverflow,
    values: view('values', 'float32', 6),
    mask: view('mask', 'uint32', 6)
  };
}

const settings = {
  quantileCount: 2,
  selectedQuantile: 2,
  seed: 7,
  permutationCount: 39,
  significanceLevel: 0.05,
  multipleTesting: 'none' as const,
  islands: 'exclude' as const,
  quantileMethod: 'type-5' as const
};

test.each([
  'greater',
  'two-sided',
  'geoda-legacy'
] as const)('public point binding forwards optional inference ports and candidate loss for %s', alternative => {
  const fixture = createFixture();
  const {device, graph, view, source, values, mask, pointWeights, candidateOverflow} = fixture;
  const add = vi.spyOn(graph, 'add');
  const tested = view('tested', 'uint32', 6);
  const analysisCount = view('analysis-count', 'uint32', 1);
  try {
    const result = addKeplerPointQuantileLISA(graph, {
      source,
      population: 'all',
      values,
      mask,
      pointWeights,
      settings: {...settings, alternative},
      outputs: {tested, analysisCount}
    });
    try {
      const contributor = add.mock.calls
        .map(([node]) => node)
        .find(node => node instanceof GPUQuantileLISA);
      expect(contributor?.props.output.tested?.buffer.id).toBe(tested.buffer.id);
      expect(contributor?.props.output.analysisCount?.buffer.id).toBe(analysisCount.buffer.id);
      expect(contributor?.props.weights.status?.candidateOverflow?.buffer.id).toBe(
        candidateOverflow.buffer.id
      );
      expect(result.outputs.tested).toBe(tested);
      expect(result.outputs.analysisCount).toBe(analysisCount);
    } finally {
      result.destroyParameters();
    }
  } finally {
    add.mockRestore();
    device.destroy();
  }
});

test('point recipe retains candidate-only loss on the inference producer', () => {
  const {device, graph, source, values, pointWeights, candidateOverflow} = createFixture();
  const add = vi.spyOn(graph, 'add');
  try {
    const result = addKeplerAnalysisRecipe(
      graph,
      {
        schemaVersion: 1,
        algorithmVersion: 1,
        id: 'candidate-loss',
        source,
        population: 'all',
        backendPolicy: 'require-gpu',
        operation: {
          kind: 'quantile-lisa-points',
          column: 'value',
          weights: pointWeights.settings,
          ...settings,
          alternative: 'greater'
        }
      },
      {source, columns: {value: values}, pointWeights}
    );
    try {
      const contributor = add.mock.calls
        .map(([node]) => node)
        .find(node => node instanceof GPUQuantileLISA);
      expect(contributor?.props.weights.status?.candidateOverflow?.buffer.id).toBe(
        candidateOverflow.buffer.id
      );
    } finally {
      result.destroyParameters();
    }
  } finally {
    add.mockRestore();
    device.destroy();
  }
});

test('conflicting weights and topology status is rejected before graph declaration', () => {
  const {device, graph, view, source, values, mask, pointWeights} = createFixture();
  const compute = vi.spyOn(graph, 'addComputePass');
  try {
    pointWeights.weights.status = {
      ...pointWeights.weights.status,
      overflow: view('other-overflow', 'uint32', 1)
    };
    expect(() =>
      addKeplerPointQuantileLISA(graph, {
        source,
        population: 'all',
        values,
        mask,
        pointWeights,
        settings: {...settings, alternative: 'greater'}
      })
    ).toThrow(/kepler-point-quantile-lisa.*weights.status.overflow.*topology.status.overflow/);
    expect(compute).not.toHaveBeenCalled();
  } finally {
    compute.mockRestore();
    device.destroy();
  }
});

test('point recipe rejects conflicting producer and wrapper loss views', () => {
  const {device, graph, view, source, values, pointWeights} = createFixture();
  try {
    pointWeights.weights.status = {
      ...pointWeights.weights.status,
      overflow: view('other-overflow', 'uint32', 1)
    };
    expect(() =>
      addKeplerAnalysisRecipe(
        graph,
        {
          schemaVersion: 1,
          algorithmVersion: 1,
          id: 'conflicting-loss',
          source,
          population: 'all',
          backendPolicy: 'require-gpu',
          operation: {
            kind: 'quantile-lisa-points',
            column: 'value',
            weights: pointWeights.settings,
            ...settings,
            alternative: 'greater'
          }
        },
        {source, columns: {value: values}, pointWeights}
      )
    ).toThrow(/weights.status.overflow.*pointWeights.status.overflow/);
  } finally {
    device.destroy();
  }
});
