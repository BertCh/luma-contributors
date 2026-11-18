// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {HUFF_REVIEW_CASES} from './huff-review-cases';

it.each([
  {role: 'demand', flagName: 'overflow'},
  {role: 'facility', flagName: 'overflow'},
  {role: 'demand', flagName: 'candidateOverflow'},
  {role: 'facility', flagName: 'candidateOverflow'}
] as const)('suppresses incomplete Huff $role weights ($flagName) and recovers on the next submission', async ({
  role,
  flagName
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'huff-incomplete');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  const floats = (name: string, values: number[]) =>
    fixture.input(name, Float32Array.from(values), 'float32', values.length);
  const integers = (name: string, values: number[]) =>
    fixture.input(name, Uint32Array.from(values), 'uint32', values.length);
  try {
    const loss = fixture.output('loss', 'uint32', 1);
    const forwardOffsets = fixture.output('offsets', 'uint32', 2);
    const reverseOffsets = fixture.output('reverse-offsets', 'uint32', 3);
    const probabilities = fixture.output('probabilities', 'float32', 2);
    const tradeArea = fixture.output('trade-area', 'uint32', 1);
    const expectedDemand = fixture.output('expected-demand', 'float32', 2);
    const status = {
      overflow: flagName === 'overflow' ? loss.view : undefined,
      candidateOverflow: flagName === 'candidateOverflow' ? loss.view : undefined,
      requiredCount: integers('required-count', [2])
    };
    fixture.graph.add(
      new GPUHuffTradeAreas({
        attractiveness: floats('attractiveness', [1, 1]),
        demand: floats('demand', [10]),
        demandWeights: {
          offsets: forwardOffsets.view,
          neighbors: integers('neighbors', [0, 1]),
          weights: floats('weights', [1, 1]),
          status: role === 'demand' ? status : undefined
        },
        facilityWeights: {
          offsets: reverseOffsets.view,
          neighbors: integers('reverse-neighbors', [0, 0]),
          weights: floats('reverse-weights', [1, 1]),
          status: role === 'facility' ? status : undefined
        },
        output: {
          probabilities: probabilities.view,
          tradeArea: tradeArea.view,
          expectedDemand: expectedDemand.view
        }
      })
    );
    compiled = fixture.graph.compile();
    for (const incomplete of [1, 0]) {
      loss.buffer.write(Uint32Array.of(incomplete));
      forwardOffsets.buffer.write(
        Uint32Array.from(role === 'demand' && incomplete ? [0, 1] : [0, 2])
      );
      reverseOffsets.buffer.write(
        Uint32Array.from(role === 'facility' && incomplete ? [0, 1, 1] : [0, 1, 2])
      );
      submitGraph(device, compiled, undefined);
      const shares = await fixture.readFloat32(probabilities, 2);
      const captured = await fixture.readFloat32(expectedDemand, 2);
      if (incomplete) {
        expect(captured.every(Number.isNaN)).toBe(true);
        if (role === 'demand') {
          expect(shares.every(Number.isNaN)).toBe(true);
          expect(await fixture.readUint32(tradeArea, 1)).toEqual([0xffffffff]);
        } else expect(shares).toEqual([0.5, 0.5]);
      } else {
        expect(shares).toEqual([0.5, 0.5]);
        expect(captured).toEqual([5, 5]);
        expect(await fixture.readUint32(tradeArea, 1)).toEqual([0]);
      }
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it.each(HUFF_REVIEW_CASES)('Huff review: $name', async testCase => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'huff-review');
  const floats = (name: string, values: number[]) =>
    fixture.input(name, Float32Array.from(values), 'float32', values.length);
  const integers = (name: string, values: number[]) =>
    fixture.input(name, Uint32Array.from(values), 'uint32', values.length);
  try {
    const probabilities = fixture.output('probabilities', 'float32', 2);
    const tradeArea = fixture.output('trade-area', 'uint32', 1);
    const expectedDemand = fixture.output('expected-demand', 'float32', 2);
    fixture.graph.add(
      new GPUHuffTradeAreas({
        attractiveness: floats('attractiveness', testCase.attractiveness),
        parameters:
          testCase.exponent === undefined ? undefined : floats('parameters', [testCase.exponent]),
        demand: floats('demand', [testCase.demand]),
        demandWeights: {
          offsets: integers('offsets', [0, 2]),
          neighbors: integers('neighbors', [0, 1]),
          weights: floats('weights', testCase.weights)
        },
        facilityWeights: {
          offsets: integers('reverse-offsets', [0, 1, 2]),
          neighbors: integers('reverse-neighbors', [0, 0]),
          weights: floats('reverse-weights', testCase.weights)
        },
        output: {
          probabilities: probabilities.view,
          tradeArea: tradeArea.view,
          expectedDemand: expectedDemand.view
        }
      })
    );
    fixture.run();
    expect(await fixture.readUint32(tradeArea, 1)).toEqual([testCase.winner]);
    const shares = await fixture.readFloat32(probabilities, 2);
    const captured = await fixture.readFloat32(expectedDemand, 2);
    for (let facility = 0; facility < 2; facility++) {
      expect(shares[facility]).toBeCloseTo(testCase.probabilities[facility], 6);
      if (testCase.captured[facility] === 0) expect(captured[facility]).toBe(0);
      else
        expect(Math.abs(captured[facility] / testCase.captured[facility] - 1)).toBeLessThan(1e-4);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  false,
  true
])('zeros removed Huff edges on one compiled graph (packed=%s)', async packed => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'huff-resubmission');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const offsets = fixture.output('offsets', 'uint32', 2);
    offsets.buffer.write(Uint32Array.of(0, 2));
    const storage = fixture.output('probabilities', 'float32', packed ? 128 : 2);
    const probabilities = packed
      ? fixture.graph.createDataView(storage.view.buffer, {
          format: 'float32',
          byteOffset: 256,
          length: 2
        })
      : storage.view;
    storage.buffer.write(Float32Array.of(99, 99), packed ? 256 : 0);
    const weights = packed
      ? fixture.graph.createDataView(storage.view.buffer, {format: 'float32', length: 2})
      : fixture.input('weights', Float32Array.of(1, 1), 'float32', 2);
    if (packed) storage.buffer.write(Float32Array.of(1, 1));
    const tradeArea = fixture.output('trade-area', 'uint32', 1);
    fixture.graph.add(
      new GPUHuffTradeAreas({
        attractiveness: fixture.input('attractiveness', Float32Array.of(1, 1), 'float32', 2),
        demandWeights: {
          offsets: offsets.view,
          neighbors: fixture.input('neighbors', Uint32Array.of(0, 1), 'uint32', 2),
          weights
        },
        output: {tradeArea: tradeArea.view, probabilities}
      })
    );
    compiled = fixture.graph.compile();
    for (const [end, expected] of [
      [2, [0.5, 0.5]],
      [1, [1, 0]],
      [0, [0, 0]]
    ] as const) {
      offsets.buffer.write(Uint32Array.of(0, end));
      submitGraph(device, compiled, undefined);
      const values = await fixture.readFloat32(storage, packed ? 128 : 2);
      expect(values.slice(packed ? 64 : 0, packed ? 66 : 2)).toEqual(expected);
      if (packed) expect(values.slice(0, 2)).toEqual([1, 1]);
      expect(await fixture.readUint32(tradeArea, 1)).toEqual([end ? 0 : 0xffffffff]);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it.each([
  {scale: 1, exponent: 10, weights: [1, 1], captured: [1, 1024]},
  {scale: 10000, exponent: 10, weights: [1, 1], captured: [1, 1024]},
  {scale: 1e-20, exponent: 10, weights: [1, 1], captured: [1, 1024]},
  {scale: 10000, exponent: 3e38, weights: [1, 1], captured: [0, 1025]},
  {scale: 10000, exponent: 10, weights: [2, 1], captured: [2050 / 1026, (1025 * 1024) / 1026]},
  {scale: 10000, exponent: 10, weights: [0, 1], captured: [0, 1025]}
])('keeps Huff shares, winner and expected demand normalized: $scale, $exponent, $weights', async ({
  scale,
  exponent,
  weights,
  captured
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'huff-scale');
  try {
    const probabilities = fixture.output('probabilities', 'float32', 2);
    const tradeArea = fixture.output('trade-area', 'uint32', 1);
    const tradeAreaProbability = fixture.output('trade-probability', 'float32', 1);
    const expectedDemand = fixture.output('expected-demand', 'float32', 2);
    fixture.graph.add(
      new GPUHuffTradeAreas({
        attractiveness: fixture.input(
          'attractiveness',
          Float32Array.of(scale, 2 * scale),
          'float32',
          2
        ),
        parameters: fixture.input('parameters', Float32Array.of(exponent), 'float32', 1),
        demand: fixture.input('demand', Float32Array.of(1025), 'float32', 1),
        demandWeights: {
          offsets: fixture.input('offsets', Uint32Array.of(0, 2), 'uint32', 2),
          neighbors: fixture.input('neighbors', Uint32Array.of(0, 1), 'uint32', 2),
          weights: fixture.input('weights', Float32Array.from(weights), 'float32', 2)
        },
        facilityWeights: {
          offsets: fixture.input('reverse-offsets', Uint32Array.of(0, 1, 2), 'uint32', 3),
          neighbors: fixture.input('reverse-neighbors', Uint32Array.of(0, 0), 'uint32', 2),
          weights: fixture.input('reverse-weights', Float32Array.from(weights), 'float32', 2)
        },
        output: {
          probabilities: probabilities.view,
          tradeArea: tradeArea.view,
          tradeAreaProbability: tradeAreaProbability.view,
          expectedDemand: expectedDemand.view
        }
      })
    );
    fixture.run();
    expect(await fixture.readUint32(tradeArea, 1)).toEqual([1]);
    const shares = await fixture.readFloat32(probabilities, 2);
    expect(shares.every(Number.isFinite)).toBe(true);
    expect(shares[0]).toBeCloseTo(captured[0] / 1025, 6);
    expect(shares[1]).toBeCloseTo(captured[1] / 1025, 6);
    expect((await fixture.readFloat32(tradeAreaProbability, 1))[0]).toBeCloseTo(
      captured[1] / 1025,
      6
    );
    const actualDemand = await fixture.readFloat32(expectedDemand, 2);
    expect(actualDemand[0]).toBeCloseTo(captured[0], 3);
    expect(actualDemand[1]).toBeCloseTo(captured[1], 2);
  } finally {
    fixture.destroy();
  }
});
