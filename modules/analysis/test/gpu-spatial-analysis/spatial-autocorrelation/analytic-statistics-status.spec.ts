// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPULocalGeary,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('cached Moran ungated quadrants publish with minimal outputs and missing rows in every CSR layout', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const weights = chunked
        ? {
            rowCount: 6,
            chunks: [
              {
                rowStart: 0,
                rowCount: 1,
                weights: rig.uploadWeights(
                  {offsets: [0, 1], neighbors: [1], weights: [1], distances: []},
                  1
                )
              },
              {
                rowStart: 1,
                rowCount: 2,
                weights: rig.uploadWeights(
                  {offsets: [0, 1, 2], neighbors: [2, 3], weights: [1, 1], distances: []},
                  1
                )
              },
              {
                rowStart: 3,
                rowCount: 3,
                weights: rig.uploadWeights(
                  {offsets: [0, 1, 2, 2], neighbors: [0, 0], weights: [1, 1], distances: []},
                  1
                )
              }
            ]
          }
        : rig.uploadWeights(
            {
              offsets: [0, 1, 2, 3, 4, 5, 5],
              neighbors: [1, 2, 3, 0, 0],
              weights: [1, 1, 1, 1, 1],
              distances: []
            },
            1
          );
      const quadrants = rig.output('uint32', 6);
      const zScores = rig.output('float32', 6);
      const neighborCounts = rig.output('uint32', 6);
      const parameters = rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8);
      rig.graph.add(
        new GPULocalMoran({
          weights,
          values: rig.input(Float32Array.of(1, 2, 3, 4, NaN, 8), 'float32', 6),
          parameters,
          quadrantGating: 'none',
          output: {
            quadrants: quadrants.view,
            zScores: zScores.view,
            neighborCounts: neighborCounts.view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const significanceLevel of [0.05, 0.000001]) {
          rig
            .bufferOf(parameters)
            .write(getGPUSpatialAutocorrelationParameterValues({significanceLevel}));
          submitGraph(device, compiled, undefined);
          // Four-cycle centered values [-1.5,-.5,.5,1.5], lags [-.5,.5,1.5,-1.5].
          expect(await readUint32(quadrants.buffer, 6)).toEqual([3, 2, 1, 4, 0, 0]);
          expect(await readUint32(neighborCounts.buffer, 6)).toEqual([1, 1, 1, 1, 0, 0]);
          const scores = await readFloat32(zScores.buffer, 6);
          for (const [row, expected] of [
            Math.sqrt(1.5),
            -1 / Math.sqrt(14),
            5 / Math.sqrt(14),
            -Math.sqrt(1.5)
          ].entries())
            expect(scores[row], `row ${row}`).toBeCloseTo(expected, 6);
          expect(scores.slice(4).every(Number.isNaN)).toBe(true);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});

it('Geary publishes the independent edge oracle and diagnostics into a packed writable layout', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const packed = rig.output('float32', 69);
    const statistic = rig.graph.createDataView(packed.view.buffer, {format: 'float32', length: 4});
    const globalStatistics = rig.graph.createDataView(packed.view.buffer, {
      format: 'float32',
      length: 5,
      byteOffset: 256
    });
    const invalidCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    rig.run(
      new GPULocalGeary({
        weights: rig.uploadWeights(
          {
            offsets: [0, 1, 3, 5, 6],
            neighbors: [1, 0, 2, 1, 3, 2],
            weights: [1, 1, 1, 1, 1, 1],
            distances: []
          },
          2
        ),
        values: rig.input(Float32Array.of(1, 2, 4, 8), 'float32', 4),
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
        output: {
          statistic,
          globalStatistics,
          status: {invalidCount: invalidCount.view, overflow: overflow.view}
        }
      })
    );
    const result = await readFloat32(packed.buffer, 69);
    // Population variance is 115/16; per-row squared edge differences are 1,5,20,16.
    for (const [row, numerator] of [16, 80, 320, 256].entries())
      expect(result[row], `row ${row}`).toBeCloseTo(numerator / 115, 6);
    for (const [slot, expected] of [4, 3.75, 115 / 16, Math.sqrt(115 / 16), 0].entries())
      expect(result[64 + slot], `moment ${slot}`).toBeCloseTo(expected, 6);
    expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

for (const contributorClass of [GPULocalMoran, GPUHotSpotAnalysis, GPULocalGeary]) {
  it(`${contributorClass.name} overwrites named input diagnostics on each encoding`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const chunked of [false, true]) {
      const rig = new WeightsRig(device);
      const values = rig.input(
        new Float32Array([1, 3, 5, 8, NaN, Infinity, -Infinity]),
        'float32',
        7
      );
      const mask = rig.input(new Uint32Array([1, 1, 1, 1, 1, 0, 1]), 'uint32', 7);
      const weightOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
      const first = rig.uploadWeights(
        {offsets: [0, 1, 2, 3], neighbors: [1, 2, 3], weights: [1, 1, 1], distances: []},
        2
      );
      const second = rig.uploadWeights(
        {offsets: [0, 1, 1, 1, 1], neighbors: [0], weights: [1], distances: []},
        2
      );
      const whole = rig.uploadWeights(
        {
          offsets: [0, 1, 2, 3, 4, 4, 4, 4],
          neighbors: [1, 2, 3, 0],
          weights: [1, 1, 1, 1],
          distances: []
        },
        4
      );
      const weights = chunked
        ? {
            rowCount: 7,
            status: {overflow: weightOverflow},
            chunks: [
              {rowStart: 0, rowCount: 3, weights: first},
              {rowStart: 3, rowCount: 4, weights: second}
            ]
          }
        : {...whole, status: {overflow: weightOverflow}};
      const parameters = rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8);
      const result = rig.output('float32', 7);
      const baselineResult = rig.output('float32', 7);
      const invalidCount = rig.output('uint32', 3);
      const overflow = rig.output('uint32', 3);
      const scalar = (view: typeof invalidCount.view) =>
        rig.graph.createDataView(view.buffer, {format: 'uint32', length: 1, byteOffset: 4});
      rig.graph.add(
        new contributorClass({
          weights,
          values,
          mask,
          parameters,
          multipleTesting: 'benjamini-hochberg',
          output: {
            zScores: result.view,
            statistic: result.view,
            status: {invalidCount: scalar(invalidCount.view), overflow: scalar(overflow.view)}
          }
        })
      );
      rig.graph.add(
        new contributorClass({
          id: 'baseline',
          weights,
          values,
          mask,
          parameters,
          multipleTesting: 'benjamini-hochberg',
          output: {zScores: baselineResult.view, statistic: baselineResult.view}
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const inputOverflow of [0, 9, 0]) {
          rig.bufferOf(weightOverflow).write(new Uint32Array([inputOverflow]));
          submitGraph(device, compiled, undefined);
          // Only the NaN and -Infinity rows are selected and invalid. Masked +Infinity is omitted.
          expect(await readUint32(invalidCount.buffer, 3)).toEqual([0x7f7f7f7f, 2, 0x7f7f7f7f]);
          expect(await readUint32(overflow.buffer, 3)).toEqual([
            0x7f7f7f7f,
            inputOverflow ? 1 : 0,
            0x7f7f7f7f
          ]);
          expect(await readFloat32(result.buffer, 7)).toEqual(
            await readFloat32(baselineResult.buffer, 7)
          );
        }
        rig.bufferOf(mask).write(new Uint32Array(7));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(invalidCount.buffer, 3)).toEqual([0x7f7f7f7f, 0, 0x7f7f7f7f]);
        rig.bufferOf(values).write(new Float32Array([1, 3, 5, 8, 2, 4, 6]));
        rig.bufferOf(mask).write(new Uint32Array(7).fill(1));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(invalidCount.buffer, 3)).toEqual([0x7f7f7f7f, 0, 0x7f7f7f7f]);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  });

  it(`${contributorClass.name} reports zero invalid rows and overflow for uint32 values without input status`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const invalidCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    try {
      const weights = rig.uploadWeights(
        {offsets: [0, 0, 0, 0], neighbors: [], weights: [], distances: []},
        2
      );
      rig.run(
        new contributorClass({
          weights,
          values: rig.input(new Uint32Array([0, 0xffffffff, 42]), 'uint32', 3),
          parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
          output: {
            zScores: rig.output('float32', 3).view,
            statistic: rig.output('float32', 3).view,
            status: {invalidCount: invalidCount.view, overflow: overflow.view}
          }
        })
      );
      expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    } finally {
      rig.destroy();
    }
  });
}
