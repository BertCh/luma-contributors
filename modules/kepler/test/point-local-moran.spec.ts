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
import {computeLocalPermutationOracle} from '../../analysis/test/gpu-spatial-analysis/permutation-inference/permutation-oracle';
import {computeLocalMoranOracle} from '../../analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-oracle';

test('point local Moran reuses fixed geographic weights, preserves primary outputs and matches CPU inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gpuDevice: Device = device;
  const graph = new GPUCommandGraph(device);
  const source = {datasetId: 'point-local-moran', revision: 2};
  const settings: KeplerPointWeightsSettings = {
    kind: 'knn',
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    k: 1,
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  };
  const originalValues = new Float32Array([1, 2, 3, 4, 10, 20, 999]);
  const selection = new Uint32Array([1, 0, 1, 1, 1, 1, 1]);
  const buffers: Buffer[] = [];
  function input(name: string, data: Float32Array) {
    const buffer = createInputBuffer(gpuDevice, data);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32', data.length)};
  }
  function output<Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) {
    const buffer = createOutputBuffer(gpuDevice, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `${name}-output`, buffer, format, length)};
  }
  const values = input('values', originalValues);
  const longitude = input('longitude', new Float32Array([0, 0.1, 10, 10.1, 20, 20.1, 181]));
  const latitude = input('latitude', new Float32Array(7));
  const selectionBuffer = createInputBuffer(gpuDevice, selection);
  buffers.push(selectionBuffer);
  const mask = importGraphBuffer(graph, 'selection', selectionBuffer, 'uint32', 7);
  const pointStatus = {
    overflow: output('point-overflow', 'uint32', 1).view,
    requiredCount: output('point-required', 'uint32', 1).view,
    invalidCount: output('point-invalid', 'uint32', 1).view
  };
  const producer = addKeplerPointWeights(graph, {
    source,
    settings,
    longitude: longitude.view,
    latitude: latitude.view,
    status: pointStatus
  });
  const binding: KeplerPointWeightsBinding = {...producer, status: pointStatus};
  const variants = [
    {
      id: 'full',
      permutation: false,
      population: 'all',
      islands: 'exclude',
      multipleTesting: 'none',
      significanceLevel: 0.4
    },
    {
      id: 'selected',
      permutation: false,
      population: 'selected',
      islands: 'keep',
      multipleTesting: 'benjamini-hochberg',
      significanceLevel: 0.2
    },
    {
      id: 'permutation',
      permutation: true,
      population: 'selected',
      islands: 'keep',
      multipleTesting: 'benjamini-hochberg',
      significanceLevel: 0.8
    }
  ] as const;
  const results = variants.map(variant => {
    const localI = output(`${variant.id}-local-i`, 'float32', 7);
    const zScores = output(`${variant.id}-z-scores`, 'float32', 7);
    const spatialLag = output(`${variant.id}-lag`, 'float32', 7);
    const pValues = output(`${variant.id}-p-values`, 'float32', 7);
    const adjustedPValues = output(`${variant.id}-adjusted-p`, 'float32', 7);
    const quadrants = output(`${variant.id}-quadrants`, 'uint32', 7);
    const neighborCounts = output(`${variant.id}-neighbors`, 'uint32', 7);
    const globalStatistics = output(`${variant.id}-statistics`, 'float32', 5);
    const weightsOverflow = output(`${variant.id}-overflow`, 'uint32', 1);
    const weightsRequiredCount = output(`${variant.id}-required`, 'uint32', 1);
    const weightsInvalidCount = output(`${variant.id}-coordinate-invalid`, 'uint32', 1);
    const invalidCount = output(`${variant.id}-undefined`, 'uint32', 1);
    const inferenceOverflow = output(`${variant.id}-inference-overflow`, 'uint32', 1);
    inferenceOverflow.buffer.write(new Uint32Array([7]));
    const recipe: KeplerAnalysisRecipe = {
      schemaVersion: 1,
      algorithmVersion: 1,
      id: variant.id,
      source,
      population: variant.population,
      ...(variant.population === 'selected' ? {selectionRevision: 3} : {}),
      backendPolicy: 'require-gpu',
      operation: {
        kind: 'local-moran-points',
        column: 'value',
        weights: settings,
        minimumNeighbors: 1,
        islands: variant.islands,
        multipleTesting: variant.multipleTesting,
        significanceLevel: variant.significanceLevel,
        inference: variant.permutation
          ? 'permutation-conditional-two-sided'
          : 'analytic-conditional-normal-two-sided',
        ...(variant.permutation ? {permutationCount: 99, seed: 17} : {})
      }
    };
    const result = addKeplerAnalysisRecipe(graph, recipe, {
      source,
      columns: {value: values.view},
      mask,
      selectionRevision: 3,
      pointWeights: binding,
      outputs: {
        values: localI.view,
        localMoran: {
          zScores: zScores.view,
          spatialLag: spatialLag.view,
          pValues: pValues.view,
          adjustedPValues: adjustedPValues.view,
          quadrants: quadrants.view,
          neighborCounts: neighborCounts.view,
          globalStatistics: globalStatistics.view,
          status: {
            weightsOverflow: weightsOverflow.view,
            weightsRequiredCount: weightsRequiredCount.view,
            weightsInvalidCount: weightsInvalidCount.view,
            invalidCount: invalidCount.view,
            inferenceOverflow: inferenceOverflow.view
          }
        }
      }
    });
    return {
      variant,
      result,
      localI,
      zScores,
      spatialLag,
      pValues,
      adjustedPValues,
      quadrants,
      neighborCounts,
      globalStatistics,
      weightsOverflow,
      weightsRequiredCount,
      weightsInvalidCount,
      inferenceOverflow,
      invalidCount
    };
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Three isolated coordinate pairs give an independently known binary kNN topology.
    const oracleWeights = {
      offsets: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 6]),
      neighbors: new Uint32Array([1, 0, 3, 2, 5, 4]),
      weights: new Float32Array(6).fill(1)
    };
    for (const item of results) {
      const effectiveMask =
        item.variant.population === 'all' ? new Uint32Array(7).fill(1) : selection.slice();
      effectiveMask[6] = 0;
      const oracle = computeLocalMoranOracle(
        {
          weights: oracleWeights,
          values: originalValues,
          mask: effectiveMask,
          islands: item.variant.islands,
          minimumNeighbors: 1,
          rowStandardized: false,
          parameters: {significanceLevel: item.variant.significanceLevel}
        },
        {multipleTesting: item.variant.multipleTesting}
      );
      if (item.variant.permutation) {
        const permutationOracle = computeLocalPermutationOracle({
          weights: oracleWeights,
          values: originalValues,
          mask: effectiveMask,
          islands: item.variant.islands,
          statistic: 'localMoran',
          alternative: 'two-sided',
          rowStandardized: false,
          seed: 17,
          permutationCount: 99,
          maximumNeighbors: 1,
          multipleTesting: item.variant.multipleTesting,
          significanceLevel: item.variant.significanceLevel
        });
        oracle.pValues = permutationOracle.pseudoPValues;
        oracle.adjustedPValues = permutationOracle.adjustedPValues;
        oracle.quadrants = permutationOracle.quadrants.map((quadrant, row) =>
          permutationOracle.significant[row] ? quadrant : 0
        );
      }
      for (const name of [
        'localI',
        'zScores',
        'spatialLag',
        'pValues',
        'adjustedPValues'
      ] as const) {
        const actual = await readFloat32(item[name].buffer, 7);
        for (let row = 0; row < 7; row++) {
          if (Number.isNaN(oracle[name][row]))
            expect(Number.isNaN(actual[row]), `${item.variant.id} ${name} ${row}`).toBe(true);
          else
            expect(actual[row], `${item.variant.id} ${name} ${row}`).toBeCloseTo(
              oracle[name][row],
              4
            );
        }
      }
      expect(Array.from(await readUint32(item.quadrants.buffer, 7))).toEqual(oracle.quadrants);
      expect(Array.from(await readUint32(item.neighborCounts.buffer, 7))).toEqual(
        oracle.neighborCounts
      );
      expect((await readFloat32(item.globalStatistics.buffer, 5))[0]).toBe(oracle.moments.count);
      expect((await readUint32(item.weightsOverflow.buffer, 1))[0]).toBe(0);
      expect((await readUint32(item.weightsRequiredCount.buffer, 1))[0]).toBe(6);
      expect((await readUint32(item.weightsInvalidCount.buffer, 1))[0]).toBe(1);
      expect((await readUint32(item.invalidCount.buffer, 1))[0]).toBe(
        (item.variant.permutation ? oracle.pValues : oracle.zScores).filter(
          (value, row) => effectiveMask[row] && !Number.isFinite(value)
        ).length
      );
      if (!('localI' in item.result) || !('pointWeights' in item.result))
        throw new Error('Expected local Moran result');
      expect(item.result.localI).toBe(item.localI.view);
      expect(item.result.pointWeights.parameterByteLength).toBe(0);
      const resultWeights = item.result.pointWeights.weights;
      if (!('offsets' in resultWeights) || !('offsets' in binding.weights))
        throw new Error('Expected unchunked point weights');
      expect(resultWeights.offsets).toBe(binding.weights.offsets);
      expect((await readUint32(item.inferenceOverflow.buffer, 1))[0]).toBe(0);
      expect(item.result.parameterByteLength).toBe(item.variant.permutation ? 48 : 32);
      expect(item.result.inference.significanceLevel).toBe(item.variant.significanceLevel);
      expect(item.result.inference.multipleTesting).toBe(item.variant.multipleTesting);
      expect(item.result.inferenceMethod).toBe(
        item.variant.permutation
          ? 'permutation-conditional-two-sided'
          : 'analytic-conditional-normal-two-sided'
      );
    }
    expect((await readFloat32(results[1].localI.buffer, 7))[0]).toBe(0);
    expect((await readUint32(results[1].neighborCounts.buffer, 7))[0]).toBe(0);
    // Constant source values invalidate analytic inference without changing the reusable topology.
    values.buffer.write(new Float32Array(7).fill(5));
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readFloat32(results[0].localI.buffer, 7)).every(Number.isNaN)).toBe(
      true
    );
    expect(Array.from(await readUint32(results[0].quadrants.buffer, 7))).toEqual([
      0, 0, 0, 0, 0, 0, 0
    ]);
    expect((await readUint32(results[0].invalidCount.buffer, 1))[0]).toBe(6);
  } finally {
    compiled.destroy();
    for (const item of results) item.result.destroyParameters();
    producer.destroyParameters();
    for (const buffer of buffers) buffer.destroy();
  }
});
