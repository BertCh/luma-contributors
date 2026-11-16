// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURankMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {GPUTransitionMatrix} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {computeRankMarkovOnCPU} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation';
import {
  GPURankMarkovOperation,
  type RankMarkovOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRankMobilityNodes} from '../../../src/gpu-spatial-analysis/distribution-dynamics/rank-markov-sorted';

for (const mode of ['full-rank', 'geographic-rank'] as const) {
  for (const tieOrder of ['giddy', 'row-id'] as const) {
    for (const rows of [63, 64, 65, 255, 256, 257]) {
      it(`sorted rank ${mode}/${tieOrder} at ${rows} rows matches the CPU sorting oracle`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const periodCount = 3;
        const values = Float32Array.from({length: rows * periodCount}, (_, index) => {
          const row = index % rows;
          const period = Math.floor(index / rows);
          return row % 7 === 0 ? (row % 2 ? -0 : 0) : ((row * (period + 1)) % 11) - 5;
        });
        values[rows + 3] = NaN;
        values[2 * rows + 4] = Infinity;
        const mask = Uint32Array.from({length: rows}, (_, row) => (row % 13 === 0 ? 0 : 1));
        const output = {
          ranks: 'ranks',
          counts: 'counts',
          probabilities: 'probabilities',
          rowTotals: 'rowTotals',
          ignored: 'ignored',
          mobility: 'mobility'
        };
        const props: RankMarkovOperationProps = {
          values: 'values',
          mask: 'mask',
          rowCount: rows,
          periodCount,
          mode,
          tieOrder,
          rankingMethod: 'sorted',
          output
        };
        const expected = computeRankMarkovOnCPU(values, props, mask);
        const actual = await runOperationGraph(device, {
          inputs: {values: {format: 'float32', values}, mask: {format: 'uint32', values: mask}},
          operations: [new GPURankMarkovOperation(props)],
          outputs: Object.values(output)
        });
        for (const name of ['ranks', 'counts', 'rowTotals', 'ignored'] as const)
          expect([...actual[name]]).toEqual([...expected[name]]);
        for (const name of ['probabilities', 'mobility'] as const) {
          for (let index = 0; index < expected[name].length; index++)
            expect(actual[name][index]).toBeCloseTo(expected[name][index], 5);
        }
        // Explicit signed-zero tie order, independent of the CPU implementation.
        const includedZeroRows = Array.from({length: rows}, (_, row) => row).filter(
          row => mask[row] && row !== 3 && row !== 4 && values[row] === 0
        );
        for (let index = 1; index < includedZeroRows.length; index++) {
          const first = actual['ranks'][includedZeroRows[index - 1]];
          const second = actual['ranks'][includedZeroRows[index]];
          expect(tieOrder === 'row-id' ? first < second : first > second).toBe(true);
        }
      });
    }
  }
}

it('keeps large integer displacement totals exact and rounds mobility once on Metal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const rows = 65535;
    const ranks = Uint32Array.from({length: rows * 2}, (_, index) =>
      index < rows ? index : 2 * rows - 1 - index
    );
    const mobility = rig.output('float32', 1);
    rig.run({
      getCommandNodes: graph =>
        createRankMobilityNodes(graph, 'large-mobility', {
          values: createTransientView(graph, 'unused-values', 'float32', ranks.length),
          periodCount: 2,
          output: {
            ranks: rig.input(ranks, 'uint32'),
            counts: createTransientView(graph, 'unused-counts', 'uint32', 1),
            mobility
          }
        })
    });
    // Sum of distances in an odd-population reversal is (n*n - 1)/2, an exact integer.
    const expected = Math.fround((rows * rows - 1) / (2 * rows));
    expect((await rig.readFloat(mobility))[0]).toBe(expected);
  } finally {
    rig.destroy();
  }
});

it('counts and ranks 4,096 states with a closed-form reversal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 4096;
  const values = Float32Array.from({length: rows * 2}, (_, index) =>
    index < rows ? index : 2 * rows - 1 - index
  );
  const actual = await runOperationGraph(device, {
    inputs: {values: {format: 'float32', values}},
    operations: [
      new GPURankMarkovOperation({
        values: 'values',
        rowCount: rows,
        periodCount: 2,
        output: {ranks: 'ranks', counts: 'counts', mobility: 'mobility'}
      })
    ],
    outputs: ['ranks', 'counts', 'mobility']
  });
  let wrongCounts = 0;
  for (let row = 0; row < rows; row++) {
    expect(actual['ranks'][row]).toBe(rows - 1 - row);
    expect(actual['ranks'][rows + row]).toBe(row);
    for (let destination = 0; destination < rows; destination++) {
      if (actual['counts'][row * rows + destination] !== (destination === rows - 1 - row ? 1 : 0))
        wrongCounts++;
    }
  }
  expect(wrongCounts).toBe(0);
  expect(actual['mobility'][0]).toBe(rows / 2);
});

for (const rows of [63, 64, 65, 255, 256, 257]) {
  for (const population of ['excluded', 'single'] as const) {
    it(`ranks ${population} panels at ${rows} rows with nonzero input and output offsets`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const periodCount = 3;
        const values = Float32Array.from({length: rows * periodCount}, (_, index) => index % rows);
        const mask = new Uint32Array(rows);
        if (population === 'single') mask[rows - 1] = 1;
        const paddedValues = new Float32Array(values.length + 16).fill(NaN);
        paddedValues.set(values, 16);
        const paddedMask = new Uint32Array(rows + 16);
        paddedMask.set(mask, 16);
        const valuesStorage = rig.input(paddedValues, 'float32');
        const maskStorage = rig.input(paddedMask, 'uint32');
        const ranksStorage = rig.output('uint32', values.length + 16);
        const countsStorage = rig.output('uint32', rows * rows + 16);
        const mobilityStorage = rig.output('float32', periodCount - 1 + 16);
        const ranks = rig.graph.createDataView(ranksStorage.buffer, {
          format: 'uint32',
          length: values.length,
          byteOffset: 64
        });
        const counts = rig.graph.createDataView(countsStorage.buffer, {
          format: 'uint32',
          length: rows * rows,
          byteOffset: 64
        });
        const mobility = rig.graph.createDataView(mobilityStorage.buffer, {
          format: 'float32',
          length: periodCount - 1,
          byteOffset: 64
        });
        rig.run(
          new GPURankMarkov({
            values: rig.graph.createDataView(valuesStorage.buffer, {
              format: 'float32',
              length: values.length,
              byteOffset: 64
            }),
            mask: rig.graph.createDataView(maskStorage.buffer, {
              format: 'uint32',
              length: rows,
              byteOffset: 64
            }),
            periodCount,
            output: {ranks, counts, mobility}
          })
        );
        const actualRanks = (await rig.readUint(ranksStorage)).slice(16);
        const actualCounts = (await rig.readUint(countsStorage)).slice(16);
        expect(actualRanks).toEqual(
          Array.from({length: values.length}, (_, index) =>
            population === 'single' && index % rows === rows - 1 ? 0 : 0xffffffff
          )
        );
        expect(actualCounts).toEqual(
          Array.from({length: rows * rows}, (_, index) =>
            population === 'single' && index === 0 ? periodCount - 1 : 0
          )
        );
        const actualMobility = (await rig.readFloat(mobilityStorage)).slice(16);
        for (const value of actualMobility) {
          expect(value).toBeNaN();
        }
        expect((await rig.readUint(ranksStorage)).slice(0, 16)).toEqual(
          new Array(16).fill(0x7f7f7f7f)
        );
      } finally {
        rig.destroy();
      }
    });
  }
}

it('counts more than 64 classes with multiple conditions, lag two, masks and invalid classes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const classCount = 65,
      conditionCount = 3;
    const counts = rig.output('uint32', conditionCount * classCount * classCount);
    const probabilities = rig.output('float32', counts.length);
    const rowTotals = rig.output('uint32', conditionCount * classCount);
    const ignored = rig.output('uint32', 1);
    rig.run(
      new GPUTransitionMatrix({
        classCount,
        conditionCount,
        periodCount: 4,
        periodLag: 2,
        classes: rig.input(
          Uint32Array.of(0, 64, 1, 2, 65, 1, 0, 64, 2, 0, 64, 0, 2, 3, 1, 0, 64, 1, 4, 2),
          'uint32'
        ),
        conditionClasses: rig.input(
          Uint32Array.of(0, 1, 2, 2, 1, 1, 2, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
          'uint32'
        ),
        mask: rig.input(Uint32Array.of(1, 1, 0, 1, 1), 'uint32'),
        output: {counts, probabilities, rowTotals, ignored}
      })
    );
    const expectedCounts = new Array(counts.length).fill(0);
    const expectedTotals = new Array(rowTotals.length).fill(0);
    // Enumerated valid (condition, from, to) triples; the other four pairs are ignored.
    for (const [condition, from, to] of [
      [0, 0, 64],
      [1, 64, 0],
      [2, 2, 3],
      [1, 1, 0],
      [2, 0, 64],
      [1, 0, 2]
    ]) {
      expectedCounts[(condition * classCount + from) * classCount + to]++;
      expectedTotals[condition * classCount + from]++;
    }
    expect(await rig.readUint(counts)).toEqual(expectedCounts);
    expect(await rig.readUint(rowTotals)).toEqual(expectedTotals);
    expect(await rig.readUint(ignored)).toEqual([4]);
    const expectedProbabilities = expectedCounts.map(
      (count, index) => count / (expectedTotals[Math.floor(index / classCount)] || 1)
    );
    expect(await rig.readFloat(probabilities)).toEqual(expectedProbabilities);
  } finally {
    rig.destroy();
  }
});
