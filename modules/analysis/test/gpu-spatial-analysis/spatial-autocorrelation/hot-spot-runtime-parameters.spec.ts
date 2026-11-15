// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUHotSpotAnalysisParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {computeHotSpotOracle} from './spatial-autocorrelation-oracle';

it('updates Gi/Gi* self weights in one compiled graph against the CPU oracle, including masks and optional outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const layout of ['fused', 'all', 'adjusted'] as const) {
    const rig = new WeightsRig(device);
    const selfWeightBuffer = new GPUParameterBuffer(device, {
      id: 'self-weight',
      format: 'float32',
      length: 2,
      values: new Float32Array([99, 1])
    });
    const imported = selfWeightBuffer.importToGraph(rig.graph);
    const selfWeight = rig.graph.createDataView(imported.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: 4
    });
    const values = new Float32Array([1, 3, 10, 4, 7, NaN]);
    const mask = new Uint32Array([1, 1, 1, 1, 0, 1]);
    const source = {
      offsets: [0, 2, 4, 6, 8, 10, 10],
      neighbors: [1, 4, 0, 2, 1, 3, 2, 4, 0, 3],
      weights: Array(10).fill(1),
      distances: []
    };
    const weights = rig.uploadWeights(source, 3);
    const zScores = rig.output('float32', 6),
      bins = rig.output('sint32', 6);
    const pValues = layout === 'fused' ? undefined : rig.output('float32', 6);
    const adjustedPValues = layout === 'fused' ? undefined : rig.output('float32', 6);
    const neighborCounts = layout === 'fused' ? undefined : rig.output('uint32', 6);
    const multipleTesting = layout === 'adjusted' ? 'benjamini-hochberg' : 'none';
    rig.graph.add(
      new GPUHotSpotAnalysis({
        weights,
        selfWeight,
        multipleTesting,
        values: rig.input(values, 'float32', 6),
        mask: rig.input(mask, 'uint32', 6),
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
        output: {
          zScores: zScores.view,
          bins: bins.view,
          pValues: pValues?.view,
          adjustedPValues: adjustedPValues?.view,
          neighborCounts: neighborCounts?.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const weight of [1, 0, 0.5, 1]) {
        selfWeightBuffer.write(getGPUHotSpotAnalysisParameterValues({selfWeight: weight}), 1);
        submitGraph(device, compiled, undefined);
        const oracle = computeHotSpotOracle(
          {
            values,
            mask,
            selfWeight: weight,
            weights: {
              offsets: new Uint32Array(source.offsets),
              neighbors: new Uint32Array(source.neighbors),
              weights: new Float32Array(source.weights)
            }
          },
          {multipleTesting}
        );
        const actual = await readFloat32(zScores.buffer, 6);
        for (const [row, expected] of oracle.zScores.entries()) {
          if (Number.isNaN(expected)) expect(actual[row]).toBeNaN();
          else expect(actual[row]).toBeCloseTo(expected, 4);
        }
        // All values in this fixture are far from the confidence thresholds.
        expect(await readUint32(bins.buffer, 6)).toEqual(oracle.bins.map(value => value >>> 0));
        if (neighborCounts)
          expect(await readUint32(neighborCounts.buffer, 6)).toEqual(oracle.neighborCounts);
        for (const [output, expected] of [
          [pValues, oracle.pValues],
          [adjustedPValues, oracle.adjustedPValues]
        ] as const) {
          if (!output) continue;
          const probabilities = await readFloat32(output.buffer, 6);
          for (const [row, probability] of expected.entries()) {
            if (Number.isNaN(probability)) expect(probabilities[row]).toBeNaN();
            else expect(probabilities[row]).toBeCloseTo(probability, 4);
          }
        }
      }
    } finally {
      compiled.destroy();
      selfWeightBuffer.destroy();
      rig.destroy();
    }
  }
});

it('rejects an underflowing self weight on the two-row case and keeps valid Gi/Gi* shader selection', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const selfWeightBuffer = new GPUParameterBuffer(device, {
    id: 'two-row-self-weight',
    format: 'float32',
    length: 1,
    values: getGPUHotSpotAnalysisParameterValues({selfWeight: 2 ** -126})
  });
  const properties = {
    weights: rig.uploadWeights({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [1, 1],
      distances: []
    }),
    values: rig.input(new Float32Array([1, 3]), 'float32', 2),
    parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8)
  };
  const zScores = rig.output('float32', 2);
  const output = {zScores: zScores.view};
  expect(() => new GPUHotSpotAnalysis({...properties, selfWeight: 1e-50, output})).toThrow(
    /selfWeight/
  );
  rig.graph.add(
    new GPUHotSpotAnalysis({
      ...properties,
      selfWeight: selfWeightBuffer.importToGraph(rig.graph),
      output
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const selfWeight of [2 ** -126, 0, 2 ** -126]) {
      selfWeightBuffer.write(getGPUHotSpotAnalysisParameterValues({selfWeight}));
      // Rejected packer updates must leave the last valid GPU parameter intact.
      expect(() =>
        selfWeightBuffer.write(getGPUHotSpotAnalysisParameterValues({selfWeight: 1e-50}))
      ).toThrow(/selfWeight/);
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(zScores.buffer, 2);
      if (selfWeight === 0) {
        // Gi excludes the focus from its moments and requires at least three rows.
        expect(actual).toEqual([NaN, NaN]);
      } else {
        // Population mean=2 and variance=1; with the tiny normal self weight, Gi* is [1,-1].
        expect(actual[0]).toBeCloseTo(1, 5);
        expect(actual[1]).toBeCloseTo(-1, 5);
      }
    }
  } finally {
    compiled.destroy();
    selfWeightBuffer.destroy();
    rig.destroy();
  }
});
