// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {createGridIndexNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-grid-index';
import {GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT} from '../../../src/gpu-spatial-analysis/spatial-regression/generalized-gwr-limits';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createGeneralizedGWRReviewScene,
  computeGeneralizedGWRReviewExpectations
} from './generalized-gwr-scalability-oracle';

async function runScene(
  rows: number,
  family: 'poisson' | 'logistic',
  mode: 'boundary' | 'bandwidth' | 'invalid' | 'converged',
  indexed = true
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2'>(format: Format, data: Float32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  const scene = createGeneralizedGWRReviewScene(rows, family, mode === 'bandwidth');
  const coefficients = output('float32', rows * 2);
  const fitted = output('float32', rows);
  const codes = output('uint32', rows);
  const iterations = output('uint32', rows);
  const golden = mode === 'invalid' || mode === 'converged';
  const scores = output('float32', golden ? 48 : 2);
  const bandwidth = output('float32', 1);
  const selectionStatus = output('uint32', 1);
  const searchIterations = output('uint32', 1);
  const searchConverged = output('uint32', 1);
  const executions = output('uint32', 1);
  const selectionOutput = {
    scores: scores.view,
    bandwidth: bandwidth.view,
    status: selectionStatus.view
  };
  const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
    id: `review-${family}`,
    family,
    kernel: 'bisquare',
    ...(indexed
      ? {indexGridSize: [Math.ceil(Math.sqrt(rows / 8)), Math.ceil(Math.sqrt(rows / 8))] as const}
      : {indexGridSize: false}),
    positions: input('float32x2', scene.positions),
    predictors: input('float32', scene.predictors),
    predictorCount: 1,
    response: input('float32', scene.response),
    parameters: input('float32', Float32Array.from([0.5, 1e-6])),
    maximumIterations: 64,
    ...(mode === 'boundary'
      ? {}
      : {
          bandwidthSelection: golden
            ? {
                method: 'golden-section',
                maximumIterations: 24,
                parameters: input(
                  'float32',
                  Float32Array.from(mode === 'invalid' ? [2, 1, 1e-5] : [0.25, 0.75, 1e-5])
                ),
                output: selectionOutput,
                status: {iterationCount: searchIterations.view, converged: searchConverged.view}
              }
            : {criterion: 'aicc', bandwidths: [0.5, 2], output: selectionOutput}
        }),
    output: {
      coefficients: coefficients.view,
      fitted: fitted.view,
      status: {codes: codes.view, iterationCount: iterations.view}
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) {
    graph.add(node);
    if (golden && /-candidate-.*-focus-(init|advance)/.test(node.id)) {
      // One extra one-thread pass has exactly the control node's GPU condition. Unlike CPU encode
      // counts, this records only work the GPU actually launches after invalid bounds/convergence.
      graph.add(
        createWGSLKernelNode(graph, {
          id: `${node.id}-execution-probe`,
          operation: 'GeneralizedGWRReviewProbe',
          invocationCount: 1,
          condition: node.condition,
          extraResources:
            node.condition?.source === 'gpu'
              ? [{buffer: node.condition.buffer, usage: 'indirect'}]
              : [],
          bindings: [
            {name: 'executionCounts', view: executions.view, type: 'u32', access: 'read_write'}
          ],
          body: 'executionCounts[executionCountsOffset] += 1u;'
        })
      );
    }
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const [actualScores, selectedStatus, selectedBandwidth] = await Promise.all([
      readFloat32(scores.buffer),
      readUint32(selectionStatus.buffer, 1),
      readFloat32(bandwidth.buffer, 1)
    ]);
    if (mode === 'invalid') {
      expect(actualScores.every(Number.isNaN)).toBe(true);
      expect(selectedStatus).toEqual([1]);
      expect(await readUint32(executions.buffer, 1)).toEqual([0]);
      expect(await readUint32(searchIterations.buffer, 1)).toEqual([0]);
      const failureCodes = await readUint32(codes.buffer);
      expect(failureCodes.every(code => code === 4)).toBe(true);
      expect((await readUint32(iterations.buffer)).every(count => count === 0)).toBe(true);
      expect((await readFloat32(coefficients.buffer)).every(Number.isNaN)).toBe(true);
      expect((await readFloat32(fitted.buffer)).every(Number.isNaN)).toBe(true);
      return;
    }
    let expectedBandwidth = 0.5;
    if (mode === 'bandwidth') {
      const expectedScores = [0.5, 2].map(value =>
        computeGeneralizedGWRReviewExpectations(scene, value).score()
      );
      for (let candidate = 0; candidate < 2; candidate++) {
        expect(Number.isFinite(actualScores[candidate])).toBe(true);
        // Absolute bound is smaller than the independently checked effect of omitting the last
        // eight rows (>8 score units for either family/candidate); no nonfinite score is filtered.
        expect(Math.abs(actualScores[candidate] - expectedScores[candidate])).toBeLessThan(2);
      }
      expectedBandwidth = expectedScores[0] < expectedScores[1] ? 0.5 : 2;
      expect(expectedBandwidth).toBe(rows <= 65552 ? 0.5 : 2);
      expect(selectedBandwidth).toEqual([expectedBandwidth]);
      expect(selectedStatus).toEqual([0]);
    }
    if (mode === 'converged') {
      expect(await readUint32(searchIterations.buffer, 1)).toEqual([1]);
      expect(await readUint32(searchConverged.buffer, 1)).toEqual([1]);
      expect(await readUint32(executions.buffer, 1)).toEqual([2 * (1 + Math.ceil(rows / 65536))]);
      const expectedScore = computeGeneralizedGWRReviewExpectations(scene, 0.5).score();
      for (let candidate = 0; candidate < 2; candidate++) {
        expect(Number.isFinite(actualScores[candidate])).toBe(true);
        expect(Math.abs(actualScores[candidate] - expectedScore)).toBeLessThan(2);
      }
      expect(actualScores.slice(2).every(Number.isNaN)).toBe(true);
    }
    const expected = computeGeneralizedGWRReviewExpectations(scene, expectedBandwidth);
    const coefficientBytes = await coefficients.buffer.readAsync();
    const coefficientValues = new Float32Array(
      coefficientBytes.buffer,
      coefficientBytes.byteOffset,
      rows * 2
    );
    const fittedBytes = await fitted.buffer.readAsync();
    const fittedValues = new Float32Array(fittedBytes.buffer, fittedBytes.byteOffset, rows);
    const statusBytes = await codes.buffer.readAsync();
    const statusValues = new Uint32Array(statusBytes.buffer, statusBytes.byteOffset, rows);
    const iterationBytes = await iterations.buffer.readAsync();
    const iterationValues = new Uint32Array(iterationBytes.buffer, iterationBytes.byteOffset, rows);
    let maximumCoefficientError = 0;
    let maximumFittedError = 0;
    let invalidRows = 0;
    for (let row = 0; row < rows; row++) {
      maximumCoefficientError = Math.max(
        maximumCoefficientError,
        Math.abs(coefficientValues[2 * row] - expected.coefficients[2 * row]),
        Math.abs(coefficientValues[2 * row + 1] - expected.coefficients[2 * row + 1])
      );
      maximumFittedError = Math.max(
        maximumFittedError,
        Math.abs(fittedValues[row] - expected.fitted[row])
      );
      if (statusValues[row] !== 0 || iterationValues[row] < 1 || iterationValues[row] > 64)
        invalidRows++;
    }
    expect(maximumCoefficientError).toBeLessThan(3e-4);
    expect(maximumFittedError).toBeLessThan(0.002);
    expect(invalidRows).toBe(0);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const family of ['poisson', 'logistic'] as const) {
  for (const rows of [65535, 65536, 65537, 1048575, 1048576, 1048584]) {
    it(`${family} matches independent, spatially varying coefficients at ${rows} rows`, async () => {
      await runScene(rows, family, 'boundary');
    }, 180000);
  }
  for (const rows of [65536, 65552, 1048576, 1048584]) {
    it(`${family} uses the global bandwidth winner with every score finite at ${rows} rows`, async () => {
      await runScene(rows, family, 'bandwidth');
    }, 180000);
  }
  it(`${family} retains the small dense route with the same independent oracle`, async () => {
    await runScene(48, family, 'boundary', false);
  });
}

for (const mode of ['invalid', 'converged'] as const) {
  it(`${mode} bounds/search state stops candidate focus-control dispatches with 24 search rounds`, async () => {
    await runScene(1048584, 'poisson', mode);
  }, 180000);
}

it('profiles the shared extent stage separately near the indexed envelope with small neighborhoods', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT - 8;
  const positions = new Float32Array(rows * 2);
  for (let row = 0; row < rows; row++) {
    positions[2 * row] = row % 4096;
    positions[2 * row + 1] = Math.floor(row / 4096);
  }
  const graph = new GPUCommandGraph(device);
  const source = createInputBuffer(device, positions);
  const result = createOutputBuffer(device, 4);
  const input = importGraphBuffer(graph, 'extent-positions', source, 'float32x2', rows);
  const output = importGraphBuffer(graph, 'extent-result', result, 'float32', 4);
  const grid = createGridIndexNodes(graph, {
    id: 'review-extent',
    operation: 'GeneralizedGWRReviewExtent',
    positions: input,
    rowCount: rows,
    gridSize: [2048, 2048]
  });
  // The grid normalizer is the first extent consumer. Keep every extent producer, including
  // published reductions introduced by W2-GWR, and omit the grid/sort/neighborhood stages.
  const consumerIndex = grid.nodes.findIndex(node => node.id.endsWith('-unit-positions'));
  expect(consumerIndex).toBeGreaterThan(0);
  const extent = grid.nodes[consumerIndex].resources?.find(
    resource =>
      'buffer' in resource &&
      'format' in resource.buffer &&
      resource.buffer.format === 'float32' &&
      resource.buffer.length === 4
  );
  expect(extent && 'buffer' in extent).toBe(true);
  for (const node of grid.nodes.slice(0, consumerIndex)) graph.add(node);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'extent-publish',
      operation: 'GeneralizedGWRReviewExtent',
      invocationCount: 4,
      bindings: [
        {
          name: 'bounds',
          view: (extent as {buffer: GraphDataView}).buffer,
          type: 'f32',
          access: 'read'
        },
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      body: 'output[outputOffset + index] = bounds[boundsOffset + index];'
    })
  );
  const compiled = graph.compile();
  try {
    const started = performance.now();
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(result, 4)).toEqual([0, 0, 4095, 4095]);
    // Submission-to-readback timing includes queue/readback overhead and is not a GPU timestamp.
    console.info(
      `GGWR extent ${rows} rows, ${consumerIndex} nodes: ${performance.now() - started} ms submission-to-readback`
    );
  } finally {
    compiled.destroy();
    source.destroy();
    result.destroy();
  }
});
