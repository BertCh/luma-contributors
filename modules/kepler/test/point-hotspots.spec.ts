// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerPointWeights, type KeplerPointWeightsBinding} from '../src/analysis/point-weights';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import type {
  KeplerAnalysisRecipe,
  KeplerPointWeightsSettings
} from '../src/recipes/analysis-recipe';
import {importGraphBuffer} from '../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

test('point Gi* reuses geographic kNN and matches independent moments, inference and island policies', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gpuDevice: Device = device;
  const graph = new GPUCommandGraph(device);
  const source = {datasetId: 'point-clusters', revision: 3};
  const settings: KeplerPointWeightsSettings = {
    kind: 'knn',
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    k: 1,
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  };
  const sourceValues = [1, 1, 1, 1, 100, 100];
  const longitudes = [0, 0.1, 10, 10.1, 20, 20.1];
  const selected = [1, 0, 1, 1, 1, 1];
  const buffers: Buffer[] = [];
  function input(name: string, values: number[]) {
    const buffer = createInputBuffer(gpuDevice, new Float32Array(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32', values.length);
  }
  function output<Format extends 'uint32' | 'sint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) {
    const buffer = createOutputBuffer(gpuDevice, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  }
  const longitude = input('longitude', longitudes);
  const latitude = input(
    'latitude',
    longitudes.map(() => 0)
  );
  const values = input('values', sourceValues);
  const maskBuffer = createInputBuffer(gpuDevice, new Uint32Array(selected));
  buffers.push(maskBuffer);
  const mask = importGraphBuffer(graph, 'selection', maskBuffer, 'uint32', selected.length);
  const pointStatus = {
    overflow: output('point-overflow', 'uint32', 1).view,
    requiredCount: output('point-required', 'uint32', 1).view,
    invalidCount: output('point-invalid', 'uint32', 1).view
  };
  const producer = addKeplerPointWeights(graph, {
    source,
    settings,
    longitude,
    latitude,
    status: pointStatus
  });
  const binding: KeplerPointWeightsBinding = {...producer, status: pointStatus};
  const variants = [
    {id: 'full-none', population: 'all', correction: 'none', islands: 'exclude'},
    {id: 'full-bh', population: 'all', correction: 'benjamini-hochberg', islands: 'exclude'},
    {id: 'selected-exclude', population: 'selected', correction: 'none', islands: 'exclude'},
    {id: 'selected-keep', population: 'selected', correction: 'none', islands: 'keep'}
  ] as const;
  const results = variants.map(variant => {
    const zScores = output(`${variant.id}-z`, 'float32', 6);
    const pValues = output(`${variant.id}-p`, 'float32', 6);
    const bins = output(`${variant.id}-bins`, 'sint32', 6);
    const neighborCounts = output(`${variant.id}-neighbors`, 'uint32', 6);
    const globalStatistics = output(`${variant.id}-statistics`, 'float32', 5);
    const weightsOverflow = output(`${variant.id}-overflow`, 'uint32', 1);
    const weightsRequiredCount = output(`${variant.id}-required`, 'uint32', 1);
    const invalidCount = output(`${variant.id}-undefined`, 'uint32', 1);
    const recipe: KeplerAnalysisRecipe = {
      schemaVersion: 1,
      algorithmVersion: 1,
      id: variant.id,
      source,
      population: variant.population,
      ...(variant.population === 'selected' ? {selectionRevision: 4} : {}),
      backendPolicy: 'require-gpu',
      operation: {
        kind: 'hot-spots-points',
        column: 'value',
        weights: settings,
        minimumNeighbors: 1,
        selfWeight: 1,
        multipleTesting: variant.correction,
        inference: 'analytic-two-sided',
        islands: variant.islands
      }
    };
    const result = addKeplerAnalysisRecipe(graph, recipe, {
      source,
      columns: {value: values},
      mask,
      selectionRevision: 4,
      pointWeights: binding,
      outputs: {
        hotSpots: {
          zScores: zScores.view,
          pValues: pValues.view,
          bins: bins.view,
          neighborCounts: neighborCounts.view,
          globalStatistics: globalStatistics.view,
          status: {
            weightsOverflow: weightsOverflow.view,
            weightsRequiredCount: weightsRequiredCount.view,
            invalidCount: invalidCount.view
          }
        }
      }
    });
    return {
      variant,
      result,
      zScores,
      pValues,
      bins,
      neighborCounts,
      globalStatistics,
      weightsOverflow,
      weightsRequiredCount,
      invalidCount
    };
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Every pair's angular separation is smaller than every cross-pair distance.
    const neighbors = longitudes.map(
      (longitudeValue, row) =>
        longitudes
          .map((value, other) => ({row: other, distance: Math.abs(value - longitudeValue)}))
          .filter(candidate => candidate.row !== row)
          .sort((first, second) => first.distance - second.distance || first.row - second.row)[0]
          .row
    );
    expect(neighbors).toEqual([1, 0, 3, 2, 5, 4]);
    for (const item of results) {
      const population = item.variant.population === 'all' ? selected.map(() => 1) : selected;
      const expected = referenceHotspots(
        sourceValues,
        neighbors,
        population,
        item.variant.islands,
        item.variant.correction
      );
      const actualScores = await readFloat32(item.zScores.buffer, 6);
      const actualProbabilities = await readFloat32(item.pValues.buffer, 6);
      const actualBins = Int32Array.from(await readUint32(item.bins.buffer, 6));
      for (let row = 0; row < 6; row++) {
        if (Number.isNaN(expected.zScores[row])) {
          expect(Number.isNaN(actualScores[row])).toBe(true);
          expect(Number.isNaN(actualProbabilities[row])).toBe(true);
        } else {
          expect(actualScores[row]).toBeCloseTo(expected.zScores[row], 4);
          expect(actualProbabilities[row]).toBeCloseTo(expected.pValues[row], 5);
        }
      }
      expect(Array.from(actualBins)).toEqual(expected.bins);
      expect((await readFloat32(item.globalStatistics.buffer, 5))[0]).toBe(expected.count);
      expect((await readUint32(item.weightsOverflow.buffer, 1))[0]).toBe(0);
      expect((await readUint32(item.weightsRequiredCount.buffer, 1))[0]).toBe(6);
      expect((await readUint32(item.invalidCount.buffer, 1))[0]).toBe(expected.invalidCount);
      if (!('pointWeights' in item.result) || !('inference' in item.result))
        throw new Error('Expected point hot spots');
      expect(item.result.pointWeights.parameterByteLength).toBe(0);
      const resultWeights = item.result.pointWeights.weights;
      if (!('offsets' in resultWeights) || !('offsets' in binding.weights))
        throw new Error('Expected unchunked point weights');
      expect(resultWeights.offsets).toBe(binding.weights.offsets);
      expect(item.result.inference.multipleTesting).toBe(item.variant.correction);
    }
    expect((await readUint32(results[0].bins.buffer, 6))[4]).toBe(2);
    expect((await readUint32(results[1].bins.buffer, 6))[4]).toBe(1);
  } finally {
    compiled.destroy();
    for (const result of results) result.result.destroyParameters();
    producer.destroyParameters();
    for (const buffer of buffers) buffer.destroy();
  }
});

function referenceHotspots(
  values: number[],
  neighbors: number[],
  selected: number[],
  islands: 'exclude' | 'keep',
  correction: 'none' | 'benjamini-hochberg'
) {
  const active = selected.map((include, row) =>
    Boolean(include && (islands === 'keep' || selected[neighbors[row]]))
  );
  const population = values.filter((value, row) => active[row]);
  const count = population.length;
  const mean = population.reduce((sum, value) => sum + value, 0) / count;
  const deviation = Math.sqrt(
    population.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count
  );
  const zScores = values.map((value, row) => {
    const neighbor = neighbors[row];
    if (!active[row] || !active[neighbor]) return NaN;
    return (
      (value + values[neighbor] - 2 * mean) / (deviation * Math.sqrt((count * 2 - 4) / (count - 1)))
    );
  });
  const pValues = zScores.map(value => normalTwoSided(value));
  const adjusted = [...pValues];
  if (correction === 'benjamini-hochberg') {
    const ordered = pValues
      .map((value, row) => ({value, row}))
      .filter(entry => Number.isFinite(entry.value))
      .sort((first, second) => first.value - second.value);
    let previous = 1;
    for (let index = ordered.length - 1; index >= 0; index--) {
      previous = Math.min(previous, (ordered[index].value * ordered.length) / (index + 1));
      adjusted[ordered[index].row] = previous;
    }
  }
  const bins = adjusted.map((value, row) => {
    const level = value <= 0.01 ? 3 : value <= 0.05 ? 2 : value <= 0.1 ? 1 : 0;
    return level === 0 ? 0 : level * (zScores[row] < 0 ? -1 : 1);
  });
  return {
    count,
    zScores,
    pValues,
    bins,
    invalidCount: zScores.filter((value, row) => selected[row] && !Number.isFinite(value)).length
  };
}

function normalTwoSided(value: number): number {
  const magnitude = Math.abs(value);
  const factor = 1 / (1 + 0.2316419 * magnitude);
  const polynomial =
    factor *
    (0.31938153 +
      factor *
        (-0.356563782 + factor * (1.781477937 + factor * (-1.821255978 + factor * 1.330274429))));
  return ((2 * Math.exp((-magnitude * magnitude) / 2)) / Math.sqrt(2 * Math.PI)) * polynomial;
}
