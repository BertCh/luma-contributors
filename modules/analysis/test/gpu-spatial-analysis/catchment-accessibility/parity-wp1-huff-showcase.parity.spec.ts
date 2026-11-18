// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUHuffTradeAreas,
  getGPUHuffTradeAreasParameterValues
} from '../../../src/gpu-spatial-analysis/catchment-accessibility/index';
import fixtureJson from '../../oracles/fixtures/parity-wp1-huff-trade-areas.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';

it.each([
  'demand',
  'facility'
] as const)('showcase described weights suppress %s loss and recover without edge-probability output', async role => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device, 'showcase-described-weights');
  try {
    const loss = rig.output('uint32', 1);
    const forwardOffsets = rig.output('uint32', 2);
    const reverseOffsets = rig.output('uint32', 3);
    loss.buffer.write(Uint32Array.of(1));
    forwardOffsets.buffer.write(Uint32Array.from(role === 'demand' ? [0, 1] : [0, 2]));
    reverseOffsets.buffer.write(Uint32Array.from(role === 'facility' ? [0, 1, 1] : [0, 1, 2]));
    const facilities = rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2);
    const demands = rig.input(Float32Array.of(0, 0), 'float32x2', 1);
    const parameters = rig.input(new Float32Array(12), 'float32');
    // Resident CSR and loss views stand in for the search output so the flag can change on
    // resubmission. Pass the real producer's described port, as both trade-area demos must do.
    const forwardSearch = new GPUNeighborSearch({
      mode: 'radius',
      gridSize: [2, 2],
      positions: facilities,
      queryPositions: demands,
      parameters,
      output: {
        offsets: forwardOffsets.view,
        neighbors: rig.input(Uint32Array.of(0, 1), 'uint32'),
        weights: rig.input(Float32Array.of(1, 1), 'float32'),
        status: {overflow: role === 'demand' ? loss.view : rig.input(Uint32Array.of(0), 'uint32')}
      }
    });
    const reverseSearch = new GPUNeighborSearch({
      mode: 'radius',
      gridSize: [2, 2],
      positions: demands,
      queryPositions: facilities,
      parameters,
      output: {
        offsets: reverseOffsets.view,
        neighbors: rig.input(Uint32Array.of(0, 0), 'uint32'),
        weights: rig.input(Float32Array.of(1, 1), 'float32'),
        status: {overflow: role === 'facility' ? loss.view : rig.input(Uint32Array.of(0), 'uint32')}
      }
    });
    const tradeArea = rig.output('uint32', 1);
    const tradeProbability = rig.output('float32', 1);
    const expectedDemand = rig.output('float32', 2);
    rig.run(
      new GPUHuffTradeAreas({
        attractiveness: rig.input(Float32Array.of(1, 1), 'float32'),
        demand: rig.input(Float32Array.of(10), 'float32'),
        demandWeights: forwardSearch.outputWeights,
        facilityWeights: reverseSearch.outputWeights,
        output: {
          tradeArea: tradeArea.view,
          tradeAreaProbability: tradeProbability.view,
          expectedDemand: expectedDemand.view
        }
      })
    );
    if (role === 'demand') {
      expect(await tradeArea.readUint32()).toEqual([0xffffffff]);
      expect((await tradeProbability.readFloat32())[0]).toBeNaN();
    } else {
      expect(await tradeArea.readUint32()).toEqual([0]);
      expect((await tradeProbability.readFloat32())[0]).toBeCloseTo(0.5, 6);
    }
    expect((await expectedDemand.readFloat32()).every(Number.isNaN)).toBe(true);
    loss.buffer.write(Uint32Array.of(0));
    forwardOffsets.buffer.write(Uint32Array.of(0, 2));
    reverseOffsets.buffer.write(Uint32Array.of(0, 1, 2));
    rig.resubmit();
    expect(await tradeArea.readUint32()).toEqual([0]);
    expect((await tradeProbability.readFloat32())[0]).toBeCloseTo(0.5, 6);
    const captured = await expectedDemand.readFloat32();
    expect(captured[0]).toBeCloseTo(5, 6);
    expect(captured[1]).toBeCloseTo(5, 6);
  } finally {
    rig.destroy();
  }
});

type HuffCase = {
  name: string;
  reference: string;
  n: number;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  facilityOffsets: number[];
  facilityNeighbors: number[];
  facilityWeights: number[];
  attractiveness: number[];
  demand: number[];
  exponent: number;
  expected: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{cases: HuffCase[]}>(fixtureJson);

function compareValues(testCase: HuffCase, metric: string, actual: number[], exact: boolean): void {
  const expected = testCase.expected[metric];
  const scale = Math.max(1, ...expected.filter(Number.isFinite).map(Math.abs));
  const absolute = exact ? 0 : 1e-6 * scale;
  const relative = exact ? 0 : 1e-4;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  expected.forEach((value, row) => {
    if (!Number.isFinite(value) || !Number.isFinite(actual[row])) {
      if (!Object.is(actual[row], value)) mismatches++;
      return;
    }
    const error = Math.abs(actual[row] - value);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(value), Number.MIN_VALUE));
    if (error > absolute + relative * Math.abs(value)) mismatches++;
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUHuffTradeAreas',
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.n,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {profile: exact ? 'E' : 'S', absolute, relative}
    })
  );
  expect.soft(mismatches, `${testCase.name}: ${metric}`).toBe(0);
}

for (const testCase of fixture.cases) {
  it(`GPUHuffTradeAreas showcase omitted-probability SciPy parity: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device, 'parity-wp1-huff');
    try {
      const outputs = {
        tradeArea: rig.output('uint32', testCase.n),
        tradeAreaProbability: rig.output('float32', testCase.n),
        expectedDemand: rig.output('float32', testCase.attractiveness.length)
      };
      rig.run(
        new GPUHuffTradeAreas({
          attractiveness: rig.input(Float32Array.from(testCase.attractiveness), 'float32'),
          demand: rig.input(Float32Array.from(testCase.demand), 'float32'),
          parameters: rig.input(
            getGPUHuffTradeAreasParameterValues({exponent: testCase.exponent}),
            'float32'
          ),
          demandWeights: {
            offsets: rig.input(Uint32Array.from(testCase.offsets), 'uint32'),
            neighbors: rig.input(Uint32Array.from(testCase.neighbors), 'uint32'),
            weights: rig.input(Float32Array.from(testCase.weights), 'float32')
          },
          facilityWeights: {
            offsets: rig.input(Uint32Array.from(testCase.facilityOffsets), 'uint32'),
            neighbors: rig.input(Uint32Array.from(testCase.facilityNeighbors), 'uint32'),
            weights: rig.input(Float32Array.from(testCase.facilityWeights), 'float32')
          },
          output: {
            tradeArea: outputs.tradeArea.view,
            tradeAreaProbability: outputs.tradeAreaProbability.view,
            expectedDemand: outputs.expectedDemand.view
          }
        })
      );
      for (const [metric, output] of Object.entries(outputs)) {
        const exact = metric === 'tradeArea';
        const actual = exact ? await output.readUint32() : await output.readFloat32();
        compareValues(testCase, metric, actual, exact);
        if (exact)
          compareValues(
            testCase,
            'served',
            actual.map(value => Number(value !== 0xffffffff)),
            true
          );
      }
    } finally {
      rig.destroy();
    }
  });
}
