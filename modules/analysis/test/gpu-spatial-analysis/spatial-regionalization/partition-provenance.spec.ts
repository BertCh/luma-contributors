// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {compactRegionLabels} from '../../../src/gpu-spatial-analysis/spatial-regionalization/compact-region-labels';
import {createPartitionInput} from './partition-budget-fixtures';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';
import {createMaxPRefinementInput} from './regionalization-operation-test-utils';
import {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {
  createSparsePartitionComparison,
  createStagedPartitionInput
} from './partition-provenance-fixtures';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('evaluates sparse labels at capacities two and four and after compaction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const outputs = createSparsePartitionComparison(rig);
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    for (let route = 0; route < outputs.length; route++) {
      const {summary, sizes, capacity} = outputs[route];
      const words = await readFloat32(summary.buffer, 9);
      expect(words[7]).toBe(route === 0 ? 2 : 0);
      expect(await readUint32(sizes.buffer, capacity)).toEqual(
        route === 0 ? [2, 0] : route === 1 ? [2, 0, 2, 0] : [2, 2]
      );
      // Within SSD=0. One region has undefined pseudo-F; two use the documented value 1.
      if (route === 0) expect(Number.isNaN(words[8])).toBe(true);
      else {
        expect(Number.isFinite(words[8])).toBe(true);
        expect(words[8]).toBe(1);
      }
    }
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});

for (const withSolverWord of [true, false]) {
  it(`keeps label provenance independent of adjacency loss, solver word ${withSolverWord}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const labelsComplete = rig.input(Uint32Array.of(0), 'uint32', 1);
    const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const summary = rig.output('float32', 9);
    const complete = rig.output('uint32', 1);
    rig.graph.add(
      new GPURegionPartitionEvaluation({
        values: rig.input(Float32Array.of(0, 2, 10, 14), 'float32', 4),
        labels: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32', 4),
        labelsComplete: withSolverWord ? labelsComplete : undefined,
        weights: {
          offsets: rig.input(Uint32Array.of(0, 1, 3, 5, 6), 'uint32', 5),
          neighbors: rig.input(Uint32Array.of(1, 0, 2, 1, 3, 2), 'uint32', 6),
          weights: rig.input(Float32Array.of(1, 1, 1, 1, 1, 1), 'float32', 6),
          status: {overflow}
        },
        output: {summary: summary.view, labelsComplete: complete.view}
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      for (const [labelFlag, loss] of [
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 0]
      ]) {
        rig.bufferOf(labelsComplete).write(Uint32Array.of(labelFlag));
        rig.bufferOf(overflow).write(Uint32Array.of(loss));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(complete.buffer, 1)).toEqual([withSolverWord ? labelFlag : 1]);
        const words = await readFloat32(summary.buffer, 9);
        // Regional means 1,12; SSD=10; between=121; pseudo-F=121/(10/2)=24.2.
        expect(Number.isFinite(words[8])).toBe(true);
        expect(words[8]).toBeCloseTo(24.2, 4);
        if (loss) expect(Number.isNaN(words[6])).toBe(true);
        else {
          expect(Number.isFinite(words[6])).toBe(true);
          expect(words[6]).toBeCloseTo(1 / 3);
        }
      }
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}

it('reduces multiple row and label tile stages with literal summary expectations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const input = createStagedPartitionInput();
  const summary = rig.output('float32', 9);
  rig.graph.add(
    new GPURegionPartitionEvaluation({
      values: rig.input(input.values, 'float32', input.values.length),
      columnCount: input.columnCount,
      labels: rig.input(input.labels, 'uint32', input.rowCount),
      labelCapacity: input.rowCount,
      output: {summary: summary.view}
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    // Independent rational calculation (Python fractions.Fraction): x alternates 0,2 for
    // 65536 rows then x=4096; y=2*x+3. n0=32769, n1=32768. Within=5*4096**2*32768/32769;
    // total=5*(4*32768+4096**2-69632**2/65537); between=total-within; F=between*65535/within.
    const actual = await readFloat32(summary.buffer, 9);
    const expected = [
      2, 83883520.07812262, 288005.5663225067, 84171525.64444512, 32768, 32769, 0, 0,
      225.00778187857736
    ];
    const tolerance = [0, 64, 0.5, 64, 0, 0, 0, 0, 0.001];
    for (let word = 0; word < expected.length; word++) {
      expect(Number.isFinite(actual[word])).toBe(true);
      expect(Math.abs(actual[word] - expected[word])).toBeLessThanOrEqual(tolerance[word]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});

it('reports exhausted Max-p improvement independently of the six status words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const input = createMaxPRefinementInput(4);
  const labels = rig.output('uint32', 9);
  const objective = rig.output('float32', 1);
  const status = rig.output('uint32', 6);
  const evidence = rig.output('uint32', 5);
  rig.graph.add(
    new GPUMaxPRegions({
      weights: {
        offsets: rig.input(input.offsets, 'uint32', input.offsets.length),
        neighbors: rig.input(input.neighbors, 'uint32', input.neighbors.length),
        weights: rig.input(input.weights, 'float32', input.weights.length)
      },
      values: rig.input(input.values, 'float32', input.values.length),
      extensiveValues: rig.input(input.extensiveValues, 'float32', 9),
      threshold: rig.input(input.threshold, 'float32', 1),
      parameters: rig.input(input.parameters, 'uint32', 1),
      standardize: false,
      startCount: 1,
      maximumIterations: 1,
      output: {
        labels: labels.view,
        objective: objective.view,
        status: status.view,
        evidence: evidence.view
      }
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    // First successful round moves row 4; a second round is still needed to move row 0.
    expect(await readUint32(labels.buffer, 9)).toEqual([1, 0, 0, 1, 1, 2, 1, 2, 2]);
    expect((await readUint32(status.buffer, 6))[5]).toBe(1);
    expect((await readUint32(evidence.buffer, 5)).slice(2)).toEqual([1, 0, 1]);
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});

for (const maximumGrowthSteps of [3, 4])
  it(`reports enclave construction completion at budget ${maximumGrowthSteps}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const input = {
      ...createPartitionInput(
        [[1], [0, 2], [1, 3], [2, 4], [3]],
        [0, 1, 2, 3, 4],
        [5, 0, 0, 0, 0],
        1
      ),
      parameters: Uint32Array.of(12)
    };
    const labels = rig.output('uint32', 5),
      evidence = rig.output('uint32', 5);
    rig.graph.add(
      new GPUMaxPRegions({
        weights: {
          offsets: rig.input(input.offsets, 'uint32', 6),
          neighbors: rig.input(input.neighbors, 'uint32', 8),
          weights: rig.input(input.weights, 'float32', 8)
        },
        values: rig.input(input.values, 'float32', 5),
        extensiveValues: rig.input(input.extensiveValues, 'float32', 5),
        threshold: rig.input(input.threshold, 'float32', 1),
        parameters: rig.input(input.parameters, 'uint32', 1),
        startCount: 1,
        maximumGrowthSteps,
        standardize: false,
        output: {
          labels: labels.view,
          objective: rig.output('float32', 1).view,
          status: rig.output('uint32', 6).view,
          evidence: evidence.view
        }
      })
    );
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(labels.buffer, 5)).toEqual(
        maximumGrowthSteps === 3 ? [0, 0, 0, 0, 1] : [0, 0, 0, 0, 0]
      );
      expect((await readUint32(evidence.buffer, 5)).slice(0, 2)).toEqual(
        maximumGrowthSteps === 3 ? [1, 0] : [0, 1]
      );
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });

it('evaluates an all-missing compacted column using a positive allocation capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const compact = compactRegionLabels(Uint32Array.of(0xffffffff, 0xffffffff));
  const summary = rig.output('float32', 9),
    sizes = rig.output('uint32', Math.max(1, compact.count));
  rig.graph.add(
    new GPURegionPartitionEvaluation({
      values: rig.input(Float32Array.of(1, 2), 'float32', 2),
      labels: rig.input(compact.labels, 'uint32', 2),
      labelCapacity: Math.max(1, compact.count),
      output: {summary: summary.view, regionSizes: sizes.view}
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(summary.buffer, 9);
    expect(compact.count).toBe(0);
    expect(actual[0]).toBe(0);
    expect(actual[7]).toBe(2);
    expect(Number.isNaN(actual[8])).toBe(true);
    expect(await readUint32(sizes.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});

for (const maximumGrowthSteps of [3, 4])
  it(`carries enclave completion through GPU operation budget ${maximumGrowthSteps}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = {
      ...createPartitionInput(
        [[1], [0, 2], [1, 3], [2, 4], [3]],
        [0, 1, 2, 3, 4],
        [5, 0, 0, 0, 0],
        1
      ),
      parameters: Uint32Array.of(12)
    };
    try {
      const result = await runOperationGraph(device, {
        inputs: getRegionalizationOperationInputs(input),
        operations: [
          new GPUMaxPRegionsOperation({
            rowCount: 5,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            values: 'values',
            extensiveValues: 'extensiveValues',
            threshold: 'threshold',
            parameters: 'parameters',
            standardize: false,
            startCount: 1,
            maximumGrowthSteps,
            output: {
              labels: 'labels',
              objective: 'objective',
              status: 'status',
              evidence: 'evidence'
            }
          })
        ],
        outputs: ['labels', 'evidence']
      });
      expect(Array.from(result['labels'])).toEqual(
        maximumGrowthSteps === 3 ? [0, 0, 0, 0, 1] : [0, 0, 0, 0, 0]
      );
      expect(Array.from(result['evidence'])).toEqual(
        maximumGrowthSteps === 3 ? [1, 0, 0, 0, 0] : [0, 1, 0, 0, 0]
      );
    } finally {
      device.destroy();
    }
  });
