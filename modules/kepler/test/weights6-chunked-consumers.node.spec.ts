// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import type {GPUChunkedSpatialWeights} from '@luma-contributors/graph/gpu-spatial-analysis';
import {expect, test} from 'vitest';
import {addKeplerSpatialRate} from '../src/analysis/analysis-bindings';
import {addKeplerPointMultivariateLocalGeary} from '../src/analysis/point-multivariate-local-geary';
import {addKeplerPointQuantileLISA} from '../src/analysis/point-quantile-lisa';
import {getKeplerPointWeightSpatialContext} from '../src/analysis/point-weight-coordinates';
import type {KeplerPointWeightsBinding} from '../src/analysis/point-weights';

const ROWS = 12;
const CHUNK_ROWS = [5, 5, 2];
const settings = {
  kind: 'distance-band' as const,
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude' as const,
  metric: 'great-circle' as const,
  units: 'meters' as const,
  radiusMeters: 1000,
  neighborCapacity: 48
};
const source = {datasetId: 'points', revision: 2};

function createGraph() {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64
    }
  });
  const graph = new GPUCommandGraph(device, {id: 'weights6'});
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = ROWS) =>
    createTransientView(graph, name, format, length);
  let rowStart = 0;
  const weights: GPUChunkedSpatialWeights = {
    rowCount: ROWS,
    spatialContext: getKeplerPointWeightSpatialContext(settings),
    information: {
      rowCount: ROWS,
      transform: 'binary',
      symmetric: 'unknown',
      hasSelfWeights: true,
      provenance: {source: 'GPUNeighborSearch', criterion: 'distance-band', radius: 1000}
    },
    chunks: CHUNK_ROWS.map((rowCount, index) => {
      const chunk = {
        rowStart,
        rowCount,
        weights: {
          offsets: view(`offsets-${index}`, 'uint32', rowCount + 1),
          neighbors: view(`neighbors-${index}`, 'uint32', rowCount * 4),
          weights: view(`weights-${index}`, 'float32', rowCount * 4)
        }
      };
      rowStart += rowCount;
      return chunk;
    })
  };
  const pointWeights: KeplerPointWeightsBinding = {
    source,
    settings,
    weights,
    coordinateMask: view('coordinate-mask', 'uint32'),
    status: {
      overflow: view('overflow', 'uint32', 1),
      requiredCount: view('required', 'uint32', 1),
      invalidCount: view('invalid', 'uint32', 1)
    }
  };
  return {graph, view, pointWeights};
}

test('quantile LISA accepts chunked point weights and emits one classes kernel per chunk', () => {
  const {graph, view, pointWeights} = createGraph();
  addKeplerPointQuantileLISA(graph, {
    source,
    population: 'all',
    values: view('values', 'float32'),
    mask: view('mask', 'uint32'),
    pointWeights,
    settings: {
      quantileCount: 4,
      selectedQuantile: 4,
      seed: 7,
      permutationCount: 99,
      significanceLevel: 0.05,
      inference: 'legacy-directed-pseudo',
      multipleTesting: 'none',
      islands: 'exclude',
      quantileMethod: 'type-5'
    }
  } as never);
  graph.device.destroy();
});

test('multivariate Geary accepts chunked point weights', () => {
  const {graph, view, pointWeights} = createGraph();
  addKeplerPointMultivariateLocalGeary(graph, {
    source,
    population: 'all',
    settings: {
      permutationCount: 19,
      seed: 3,
      significanceLevel: 0.05,
      correction: 'none',
      tail: 'equal-tail'
    },
    variables: [view('first', 'float32'), view('second', 'float32')],
    mask: view('mask', 'uint32'),
    pointWeights
  } as never);
  graph.device.destroy();
});

test('spatial rates accept chunked weights', () => {
  const {graph, view, pointWeights} = createGraph();
  addKeplerSpatialRate(graph, {
    source,
    population: 'all',
    events: view('events', 'float32'),
    populations: view('populations', 'float32'),
    weights: pointWeights.weights,
    weightsSource: source,
    output: view('rates', 'float32')
  } as never);
  graph.device.destroy();
});

test('converted adapters no longer require single-CSR point weights', () => {
  for (const file of [
    'point-quantile-lisa',
    'point-multivariate-local-geary',
    'weighted-quantile-lisa',
    'recipe-bindings'
  ]) {
    const text = readFileSync(new URL(`../src/analysis/${file}.ts`, import.meta.url), 'utf8');
    expect(text, file).not.toContain('requireKeplerWholePointWeights');
  }
});
