// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GPUGather} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

function createCompositionFixture(device: Device, rows: number, predictorCount: number) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${serial++}`, buffer, format, values.length);
  };
  const allocate = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  const coefficients = allocate('float32', predictorCount + 1);
  const status = allocate('uint32', 1);
  const output = {
    coefficients: coefficients.view,
    standardErrors: allocate('float32', predictorCount + 1).view,
    tStatistics: allocate('float32', predictorCount + 1).view,
    summary: allocate('float32', 22).view,
    status: {code: status.view},
    residuals: allocate('float32', rows).view
  };
  return {
    graph,
    input,
    allocate,
    output,
    coefficients,
    status,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it('CMP-11 fits ordered dataframe columns with intercept, x, z coefficients [1,2,3]', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createCompositionFixture(device, 5, 2);
  const {graph, input, output} = fixture;
  const predictors = GPUOrdinaryLeastSquares.createDesign(graph, [
    input('float32', new Float32Array([0, 1, 0, 1, 2])),
    input('float32', new Float32Array([0, 0, 1, 1, 1]))
  ]);
  graph.add(
    new GPUOrdinaryLeastSquares({
      predictors,
      response: input('float32', new Float32Array([1, 3, 4, 6, 8])),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(fixture.status.buffer)).toEqual([0]);
    const coefficients = await readFloat32(fixture.coefficients.buffer);
    expect(coefficients).toHaveLength(3);
    for (const [index, expected] of [1, 2, 3].entries()) {
      expect(Number.isFinite(coefficients[index])).toBe(true);
      expect(coefficients[index]).toBeCloseTo(expected, 4);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const mode of ['incomplete', 'complete', 'unmatched'] as const) {
  it(`CMP-01 preserves joined-table completeness for ${mode} training data`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createCompositionFixture(device, 5, 1);
    const {graph, input, allocate, output} = fixture;
    const rightKeys = input(
      'uint32',
      new Uint32Array(mode === 'complete' ? [0, 1, 2, 3, 4] : [0, 1, 2, 3])
    );
    const groupOverflow = input('uint32', new Uint32Array([mode === 'incomplete' ? 1 : 0]));
    const matched = allocate('uint32', 5);
    const complete = allocate('uint32', 1);
    const candidateOverflow = allocate('uint32', 1);
    const joinedPredictors = allocate('float32', 5);
    const joinedTable = {
      matched: matched.view,
      status: {complete: complete.view, candidateOverflow: candidateOverflow.view}
    };
    graph.add(
      new GPUKeyLookup({
        leftKeys: input('uint32', new Uint32Array([4, 1, 3, 0, 2])),
        rightKeys,
        rightTable: {keys: rightKeys, status: {overflow: groupOverflow}},
        gather: [
          {
            column: input(
              'float32',
              new Float32Array(mode === 'complete' ? [0, 1, 2, 3, 4] : [0, 1, 2, 3])
            ),
            unmatchedValue: 99,
            output: joinedPredictors.view
          }
        ],
        output: joinedTable
      })
    );
    const composition = GPUOrdinaryLeastSquares.addJoinedTableFit(
      graph,
      {
        predictors: joinedPredictors.view,
        response: input('float32', new Float32Array([50, 3, 7, 1, 5])),
        mask: input('uint32', new Uint32Array([1, 1, 0, 1, 1])),
        output
      },
      joinedTable
    );
    expect(composition.status.candidateOverflow).toBe(candidateOverflow.view);
    const maskBuffer = allocate('uint32', 5);
    graph.add(
      new GPUGather({
        id: 'read-training-mask',
        source: composition.mask,
        indices: input('uint32', new Uint32Array([0, 1, 2, 3, 4])),
        output: maskBuffer.view
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const status = await readUint32(fixture.status.buffer);
      const coefficients = await readFloat32(fixture.coefficients.buffer);
      expect(coefficients).toHaveLength(2);
      expect(await readUint32(maskBuffer.buffer)).toEqual(
        mode === 'incomplete'
          ? [0, 0, 0, 0, 0]
          : mode === 'complete'
            ? [1, 1, 0, 1, 1]
            : [0, 1, 0, 1, 1]
      );
      expect(await readUint32(candidateOverflow.buffer)).toEqual([mode === 'incomplete' ? 1 : 0]);
      expect(await readUint32(complete.buffer)).toEqual([mode === 'incomplete' ? 0 : 1]);
      if (mode === 'incomplete') {
        expect(status).toEqual([2]);
        expect(coefficients.every(Number.isNaN)).toBe(true);
      } else {
        expect(status).toEqual([0]);
        expect(coefficients.every(Number.isFinite)).toBe(true);
        expect(coefficients[0]).toBeCloseTo(mode === 'complete' ? -36 / 5 : 1, 4);
        // Complete after excluding key 3: hand centered covariance=439/4, sum-squares=35/4. Genuine miss: y=1+2*x.
        expect(coefficients[1]).toBeCloseTo(mode === 'complete' ? 439 / 35 : 2, 4);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
