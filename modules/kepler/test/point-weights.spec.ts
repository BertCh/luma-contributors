// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph, GPUMask} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerPointWeights} from '../src/analysis/point-weights';
import {addKeplerSpatialRate} from '../src/analysis/analysis-bindings';
import {importGraphBuffer} from '../../graph/src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../graph/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

test('great-circle point weights cross the antimeridian and are reusable without refilling masked neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const source = {datasetId: 'points', revision: 3};
  const longitudeBuffer = createInputBuffer(
    device,
    new Float32Array([179.9, -179.9, 170, 0, NaN, 0, 181])
  );
  const latitudeBuffer = createInputBuffer(device, new Float32Array(7));
  const validityBuffer = createInputBuffer(device, new Uint32Array([1, 1, 1, 1, 1, 0, 1]));
  const eventBuffer = createInputBuffer(device, new Float32Array([10, 30, 80, 100, 999, 999, 999]));
  const populationBuffer = createInputBuffer(device, new Float32Array(7).fill(100));
  const selectionBuffer = createInputBuffer(device, new Uint32Array([1, 0, 1, 1, 1, 1, 1]));
  const buffers = [
    longitudeBuffer,
    latitudeBuffer,
    validityBuffer,
    eventBuffer,
    populationBuffer,
    selectionBuffer
  ];
  function output(name: string, length: number) {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  }
  const offsets = output('offsets', 8);
  const neighbors = output('neighbors', 7);
  const weights = output('weights', 7);
  const overflow = output('overflow', 1);
  const requiredCount = output('required-count', 1);
  const invalidCount = output('invalid-count', 1);
  const full = output('full', 7);
  const selected = output('selected', 7);
  const selectedMask = output('selected-mask', 7);
  const pointWeights = addKeplerPointWeights(graph, {
    source,
    settings: {
      kind: 'knn',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 1,
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    },
    longitude: importGraphBuffer(graph, 'longitude', longitudeBuffer, 'float32', 7),
    latitude: importGraphBuffer(graph, 'latitude', latitudeBuffer, 'float32', 7),
    coordinateValidityMask: importGraphBuffer(
      graph,
      'coordinate-validity',
      validityBuffer,
      'uint32',
      7
    ),
    output: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: importGraphBuffer(graph, 'weight-values', weights.buffer, 'float32', 7),
      status: {
        overflow: overflow.view,
        requiredCount: requiredCount.view,
        invalidCount: invalidCount.view
      }
    }
  });
  graph.add(
    new GPUMask({
      id: 'selected-valid-points',
      inputs: [
        pointWeights.coordinateMask,
        importGraphBuffer(graph, 'selection', selectionBuffer, 'uint32', 7)
      ],
      output: selectedMask.view
    })
  );
  const rateInputs = {
    source,
    events: importGraphBuffer(graph, 'events', eventBuffer, 'float32', 7),
    populations: importGraphBuffer(graph, 'populations', populationBuffer, 'float32', 7),
    weights: pointWeights.weights,
    weightsSource: pointWeights.source,
    scale: 1000,
    islandPolicy: 'undefined' as const
  };
  addKeplerSpatialRate(graph, {
    ...rateInputs,
    id: 'full-rates',
    population: 'all',
    mask: pointWeights.coordinateMask,
    output: importGraphBuffer(graph, 'full-rates-output', full.buffer, 'float32', 7)
  });
  addKeplerSpatialRate(graph, {
    ...rateInputs,
    id: 'selected-rates',
    population: 'selected',
    selectionRevision: 2,
    mask: selectedMask.view,
    output: importGraphBuffer(graph, 'selected-rates-output', selected.buffer, 'float32', 7)
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(offsets.buffer, 8))).toEqual([0, 1, 2, 3, 4, 4, 4, 4]);
    expect(Array.from(await readUint32(neighbors.buffer, 4))).toEqual([1, 0, 0, 2]);
    expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([0]);
    expect(Array.from(await readUint32(requiredCount.buffer, 1))).toEqual([4]);
    expect(Array.from(await readUint32(invalidCount.buffer, 1))).toEqual([3]);
    const fullValues = await readFloat32(full.buffer, 7);
    const selectedValues = await readFloat32(selected.buffer, 7);
    for (const [index, expected] of [200, 200, 450, 900].entries())
      expect(fullValues[index]).toBeCloseTo(expected, 3);
    expect(Array.from(fullValues.slice(4)).every(Number.isNaN)).toBe(true);
    expect(Number.isNaN(selectedValues[0])).toBe(true);
    expect(Number.isNaN(selectedValues[1])).toBe(true);
    expect(selectedValues[2]).toBeCloseTo(450, 3);
    expect(selectedValues[3]).toBeCloseTo(900, 3);
    expect(Array.from(selectedValues.slice(4)).every(Number.isNaN)).toBe(true);
  } finally {
    compiled.destroy();
    pointWeights.destroyParameters();
    for (const buffer of buffers) buffer.destroy();
  }
});
