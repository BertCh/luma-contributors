// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  getGPUHuffTradeAreasParameterValues,
  GPUCatchmentAccessibility,
  GPUHuffTradeAreas
} from '../../../src/gpu-spatial-analysis/catchment-accessibility/index';
import {computeCatchmentOracle, computeHuffOracle} from './catchment-oracle';
import {AnalysisRig, buildKernelWeights, createSeededRandom, transposeWeights} from './rig';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

for (const method of ['2sfca', '3sfca'] as const) {
  for (const role of ['demand', 'facility'] as const) {
    it.each([
      'overflow',
      'candidateOverflow'
    ] as const)(`${method} suppresses ${role} weights %s and restores complete scores`, async flagName => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new GraphRig(device, 'catchment-loss');
      try {
        const loss = rig.output('uint32', 1);
        const forwardOffsets = rig.output('uint32', 2);
        const reverseOffsets = rig.output('uint32', 3);
        loss.buffer.write(Uint32Array.of(1));
        forwardOffsets.buffer.write(Uint32Array.from(role === 'demand' ? [0, 1] : [0, 2]));
        reverseOffsets.buffer.write(Uint32Array.from(role === 'facility' ? [0, 1, 1] : [0, 1, 2]));
        const status = {
          overflow: flagName === 'overflow' ? loss.view : undefined,
          candidateOverflow: flagName === 'candidateOverflow' ? loss.view : undefined
        };
        const accessibility = rig.output('float32', 1);
        const ratios = rig.output('float32', 2);
        const reachable = rig.output('uint32', 1);
        const validity = rig.output('uint32', 1);
        rig.run(
          new GPUCatchmentAccessibility({
            method,
            supply: rig.input(Float32Array.of(1, 1), 'float32'),
            demand: rig.input(Float32Array.of(1), 'float32'),
            demandWeights: {
              offsets: forwardOffsets.view,
              neighbors: rig.input(Uint32Array.of(0, 1), 'uint32'),
              weights: rig.input(Float32Array.of(1, 1), 'float32'),
              status: role === 'demand' ? status : undefined
            },
            facilityWeights: {
              offsets: reverseOffsets.view,
              neighbors: rig.input(Uint32Array.of(0, 0), 'uint32'),
              weights: rig.input(Float32Array.of(1, 1), 'float32'),
              status: role === 'facility' ? status : undefined
            },
            output: {
              accessibility: accessibility.view,
              ratios: ratios.view,
              reachableFacilities: reachable.view,
              validity: validity.view
            }
          })
        );
        expect((await accessibility.readFloat32())[0]).toBeNaN();
        if (role === 'facility' || method === '3sfca')
          expect((await ratios.readFloat32()).every(Number.isNaN)).toBe(true);
        else expect(await ratios.readFloat32()).toEqual([1, 1]);
        expect(await reachable.readUint32()).toEqual([role === 'demand' ? 0xffffffff : 2]);
        expect(await validity.readUint32()).toEqual([1]);
        loss.buffer.write(Uint32Array.of(0));
        forwardOffsets.buffer.write(Uint32Array.of(0, 2));
        reverseOffsets.buffer.write(Uint32Array.of(0, 1, 2));
        rig.resubmit();
        expect((await accessibility.readFloat32())[0]).toBeCloseTo(2, 6);
        expect(await ratios.readFloat32()).toEqual(method === '2sfca' ? [1, 1] : [2, 2]);
        expect(await reachable.readUint32()).toEqual([2]);
      } finally {
        rig.destroy();
      }
    });
  }
}

function expectClose(actual: number[], expected: number[], label: string, relative = 5e-5): void {
  expectNumericArrayClose(actual, expected, {
    absoluteTolerance: 1e-6,
    relativeTolerance: relative,
    label
  });
}

function randomPoints(count: number, seed: number): number[] {
  const random = createSeededRandom(seed);
  return Array.from({length: count * 2}, () => random() * 10);
}

function randomValues(count: number, seed: number, scale: number): number[] {
  const random = createSeededRandom(seed);
  return Array.from({length: count}, () => Math.fround(1 + Math.floor(random() * scale)));
}

for (const method of ['2sfca', '3sfca'] as const) {
  it(`GPUCatchmentAccessibility ${method} matches the oracle on symmetric self-join weights`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = 60;
    const points = randomPoints(count, 11);
    const csr = buildKernelWeights(points, points, 2.5, {selfJoin: false});
    const supply = randomValues(count, 5, 20);
    const demand = randomValues(count, 6, 200);
    const expected = computeCatchmentOracle(method, supply, demand, csr, csr);
    const rig = new AnalysisRig(device);
    const accessibility = rig.output('float32', count);
    const ratios = rig.output('float32', count);
    const reachable = rig.output('uint32', count);
    rig.run(
      new GPUCatchmentAccessibility({
        method,
        supply: rig.input(Float32Array.from(supply), 'float32'),
        demand: rig.input(Float32Array.from(demand), 'float32'),
        facilityWeights: rig.weights(csr),
        output: {accessibility, ratios, reachableFacilities: reachable}
      })
    );
    const actual = await rig.readFloat(accessibility);
    expect(actual.some(value => value > 0)).toBe(true);
    expectClose(actual, expected.accessibility, `${method} accessibility`);
    expectClose(await rig.readFloat(ratios), expected.ratios, `${method} ratios`);
    expect(await rig.readUint(reachable)).toEqual(expected.reachable);
    rig.destroy();
  });
}

it('GPUCatchmentAccessibility 3sfca conserves supply and handles cross weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const facilityCount = 12;
  const demandCount = 50;
  const facilityPoints = randomPoints(facilityCount, 21);
  const demandPoints = randomPoints(demandCount, 22);
  const facilityWeights = buildKernelWeights(facilityPoints, demandPoints, 4, {selfJoin: false});
  const demandWeights = transposeWeights(facilityWeights, demandCount);
  const supply = randomValues(facilityCount, 7, 10);
  const demand = randomValues(demandCount, 8, 100);
  const expected = computeCatchmentOracle('3sfca', supply, demand, facilityWeights, demandWeights);
  const rig = new AnalysisRig(device);
  const accessibility = rig.output('float32', demandCount);
  rig.run(
    new GPUCatchmentAccessibility({
      method: '3sfca',
      supply: rig.input(Float32Array.from(supply), 'float32'),
      demand: rig.input(Float32Array.from(demand), 'float32'),
      facilityWeights: rig.weights(facilityWeights),
      demandWeights: rig.weights(demandWeights),
      output: {accessibility}
    })
  );
  const actual = await rig.readFloat(accessibility);
  expectClose(actual, expected.accessibility, '3sfca cross accessibility');
  // Conservation: sum_i P_i A_i equals the supply of facilities with demand in range.
  const reachableSupply = supply.reduce(
    (sum, value, j) =>
      sum + (facilityWeights.offsets[j + 1] > facilityWeights.offsets[j] ? value : 0),
    0
  );
  const total = actual.reduce((sum, value, i) => sum + value * demand[i], 0);
  expect(Math.abs(total - reachableSupply) / reachableSupply).toBeLessThan(1e-3);
  rig.destroy();
});

it('GPUCatchmentAccessibility treats non-finite demand and supply as missing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 40;
  const points = randomPoints(count, 41);
  const csr = buildKernelWeights(points, points, 3, {selfJoin: false});
  const supply = randomValues(count, 42, 20);
  const demand = randomValues(count, 43, 200);
  supply[3] = Number.POSITIVE_INFINITY;
  demand[5] = Number.NaN;
  demand[9] = Number.NEGATIVE_INFINITY;
  for (const method of ['2sfca', '3sfca'] as const) {
    const expected = computeCatchmentOracle(method, supply, demand, csr, csr);
    const rig = new AnalysisRig(device);
    const accessibility = rig.output('float32', count);
    rig.run(
      new GPUCatchmentAccessibility({
        method,
        supply: rig.input(Float32Array.from(supply), 'float32'),
        demand: rig.input(Float32Array.from(demand), 'float32'),
        facilityWeights: rig.weights(csr),
        output: {accessibility}
      })
    );
    const actual = await rig.readFloat(accessibility);
    expect(actual.every(Number.isFinite), `${method} finite`).toBe(true);
    expectClose(actual, expected.accessibility, `${method} missing values`);
    rig.destroy();
  }
});

it('GPUHuffTradeAreas matches the oracle with an exponent and keeps rows normalized', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const facilityCount = 10;
  const demandCount = 45;
  const demandPoints = randomPoints(demandCount, 31);
  const facilityPoints = randomPoints(facilityCount, 32);
  const demandWeights = buildKernelWeights(demandPoints, facilityPoints, 5, {selfJoin: false});
  const facilityWeights = transposeWeights(demandWeights, facilityCount);
  const attractiveness = randomValues(facilityCount, 9, 30);
  attractiveness[3] = 0;
  const demand = randomValues(demandCount, 10, 50);
  const alpha = 1.5;
  const expected = computeHuffOracle(attractiveness, demand, alpha, demandWeights, facilityWeights);
  const rig = new AnalysisRig(device);
  const probabilities = rig.output('float32', demandWeights.neighbors.length + 3);
  const tradeArea = rig.output('uint32', demandCount);
  const tradeAreaProbability = rig.output('float32', demandCount);
  const expectedDemand = rig.output('float32', facilityCount);
  rig.run(
    new GPUHuffTradeAreas({
      attractiveness: rig.input(Float32Array.from(attractiveness), 'float32'),
      demand: rig.input(Float32Array.from(demand), 'float32'),
      demandWeights: rig.weights(demandWeights),
      facilityWeights: rig.weights(facilityWeights),
      parameters: rig.input(getGPUHuffTradeAreasParameterValues({exponent: alpha}), 'float32'),
      output: {probabilities, tradeArea, tradeAreaProbability, expectedDemand}
    })
  );
  const probabilityValues = (await rig.readFloat(probabilities)).slice(
    0,
    demandWeights.neighbors.length
  );
  expect(probabilityValues.some(value => value > 0)).toBe(true);
  expectClose(probabilityValues, expected.probabilities, 'huff probabilities');
  for (let i = 0; i < demandCount; i++) {
    let sum = 0;
    for (let s = demandWeights.offsets[i]; s < demandWeights.offsets[i + 1]; s++) {
      sum += probabilityValues[s];
    }
    if (demandWeights.offsets[i + 1] > demandWeights.offsets[i]) {
      expect(Math.abs(sum - 1)).toBeLessThan(1e-4);
    }
  }
  expect(await rig.readUint(tradeArea)).toEqual(expected.tradeArea);
  expectClose(await rig.readFloat(tradeAreaProbability), expected.tradeAreaProbability, 'trade p');
  const expectedDemandValues = await rig.readFloat(expectedDemand);
  expectClose(expectedDemandValues, expected.expectedDemand, 'expected demand');
  expect(expectedDemandValues[3]).toBe(0);
  // Demand is conserved over demand locations that reach any facility.
  const served = demand.reduce(
    (sum, value, i) => sum + (expected.tradeArea[i] !== 0xffffffff ? value : 0),
    0
  );
  const captured = expectedDemandValues.reduce((sum, value) => sum + value, 0);
  expect(Math.abs(captured - served) / served).toBeLessThan(1e-3);
  rig.destroy();
});

it('GPUHuffTradeAreas skips a non-finite demand and reports the row invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Three demand rows reach both facilities with weight 1. Attractiveness 1 and 3 give every row
  // the probabilities 1/4 and 3/4. Only row 1 has a usable demand (10), so by hand the expected
  // demand is 10 / 4 = 2.5 and 3 * 10 / 4 = 7.5, the result GPUCatchmentAccessibility's rule
  // (a non-finite demand is missing and adds nothing) gives.
  const demandWeights = {
    offsets: [0, 2, 4, 6],
    neighbors: [0, 1, 0, 1, 0, 1],
    weights: Array(6).fill(1)
  };
  const facilityWeights = {
    offsets: [0, 3, 6],
    neighbors: [0, 1, 2, 0, 1, 2],
    weights: Array(6).fill(1)
  };
  const rig = new AnalysisRig(device);
  const tradeArea = rig.output('uint32', 3);
  const expectedDemand = rig.output('float32', 2);
  const validity = rig.output('uint32', 3);
  rig.run(
    new GPUHuffTradeAreas({
      attractiveness: rig.input(Float32Array.of(1, 3), 'float32'),
      demand: rig.input(Float32Array.of(Number.NaN, 10, Number.POSITIVE_INFINITY), 'float32'),
      demandWeights: rig.weights(demandWeights),
      facilityWeights: rig.weights(facilityWeights),
      output: {tradeArea, expectedDemand, validity}
    })
  );
  const captured = await rig.readFloat(expectedDemand);
  // Relative-log normalization uses float32 transcendental operations; allow their rounding
  // error while still rejecting NaN, infinity, and any contribution from the missing rows.
  expect(captured).toHaveLength(2);
  expect(captured[0]).toBeCloseTo(2.5, 6);
  expect(captured[1]).toBeCloseTo(7.5, 6);
  expect(await rig.readUint(validity)).toEqual([0, 1, 0]);
  // The trade area does not depend on the demand value.
  expect(await rig.readUint(tradeArea)).toEqual([1, 1, 1]);
  rig.destroy();
});

it('GPUCatchmentAccessibility skips a non-finite demand and reports the row invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One facility with supply 6 reaches three demand rows with weight 1. Only row 1 has a usable
  // demand (3), so by hand R = 6 / 3 = 2 and every row's accessibility is 1 * R = 2: a row with a
  // missing demand still has access, it only adds nothing to the facility's load.
  const facilityWeights = {offsets: [0, 3], neighbors: [0, 1, 2], weights: [1, 1, 1]};
  const demandWeights = {offsets: [0, 1, 2, 3], neighbors: [0, 0, 0], weights: [1, 1, 1]};
  const rig = new AnalysisRig(device);
  const accessibility = rig.output('float32', 3);
  const ratios = rig.output('float32', 1);
  const validity = rig.output('uint32', 3);
  rig.run(
    new GPUCatchmentAccessibility({
      method: '2sfca',
      supply: rig.input(Float32Array.of(6), 'float32'),
      demand: rig.input(Float32Array.of(Number.NaN, 3, Number.NEGATIVE_INFINITY), 'float32'),
      facilityWeights: rig.weights(facilityWeights),
      demandWeights: rig.weights(demandWeights),
      output: {accessibility, ratios, validity}
    })
  );
  expect(await rig.readFloat(ratios)).toEqual([2]);
  expect(await rig.readFloat(accessibility)).toEqual([2, 2, 2]);
  expect(await rig.readUint(validity)).toEqual([0, 1, 0]);
  rig.destroy();
});
