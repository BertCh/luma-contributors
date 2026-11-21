// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPUAZPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  createPartitionInput,
  inspectPartition,
  inspectBudgetPartition
} from './partition-budget-fixtures';

for (let maximumGrowthSteps = 1; maximumGrowthSteps <= 8; maximumGrowthSteps++) {
  for (const maximumIterations of [0, 4]) {
    it(`GPU seed 619 partition at growth ${maximumGrowthSteps}, improvement ${maximumIterations}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      const input = createPartitionInput();
      const labels = rig.output('uint32', 8),
        objective = rig.output('float32', 1),
        status = rig.output('uint32', 6);
      const evidence = rig.output('uint32', 5);
      rig.graph.add(
        new GPUMaxPRegions({
          weights: {
            offsets: rig.input(input.offsets, 'uint32', 9),
            neighbors: rig.input(input.neighbors, 'uint32', input.neighbors.length),
            weights: rig.input(input.weights, 'float32', input.weights.length)
          },
          values: rig.input(input.values, 'float32', 8),
          extensiveValues: rig.input(input.extensiveValues, 'float32', 8),
          threshold: rig.input(input.threshold, 'float32', 1),
          parameters: rig.input(input.parameters, 'uint32', 1),
          standardize: false,
          maximumGrowthSteps,
          maximumIterations,
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
        const labelValues = await readUint32(labels.buffer, 8);
        const [score] = await readFloat32(objective.buffer, 1);
        const words = await readUint32(status.buffer, 6);
        const evidenceWords = await readUint32(evidence.buffer, 5);
        if (maximumGrowthSteps === 2) expect(evidenceWords.slice(0, 2)).toEqual([1, 0]);
        expect(evidenceWords[2]).toBe(Number(maximumIterations > 0));
        if (maximumIterations === 0) expect(evidenceWords.slice(3)).toEqual([0, 0]);
        inspectBudgetPartition(labelValues, score, words, maximumGrowthSteps, maximumIterations);
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy();
      }
    });
  }
}

for (const model of ['AZP', 'Max-p'] as const) {
  it(`recovers the unique enumerable optimum for ${model}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const input = createPartitionInput([[1], [0, 2], [1, 3], [2]], [0, 1, 10, 11], [1, 1, 1, 1], 2);
    const labels = rig.output('uint32', 4),
      objective = rig.output('float32', 1),
      status = rig.output('uint32', model === 'AZP' ? 4 : 6);
    const common = {
      weights: {
        offsets: rig.input(input.offsets, 'uint32', 5),
        neighbors: rig.input(input.neighbors, 'uint32', 6),
        weights: rig.input(input.weights, 'float32', 6)
      },
      values: rig.input(input.values, 'float32', 4),
      extensiveValues: rig.input(input.extensiveValues, 'float32', 4),
      parameters: rig.input(input.parameters, 'uint32', 1),
      standardize: false,
      maximumIterations: 4,
      output: {labels: labels.view, objective: objective.view, status: status.view}
    };
    rig.graph.add(
      model === 'AZP'
        ? new GPUAZPRegions({
            ...common,
            regionCapacity: 2,
            initialLabels: rig.input(Uint32Array.of(0, 1, 1, 1), 'uint32', 4),
            extensiveThreshold: rig.input(input.threshold, 'float32', 1)
          })
        : new GPUMaxPRegions({...common, threshold: rig.input(input.threshold, 'float32', 1)})
    );
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      const result = inspectPartition(
        await readUint32(labels.buffer, 4),
        [[1], [0, 2], [1, 3], [2]],
        [0, 1, 10, 11],
        [1, 1, 1, 1]
      );
      expect(result.masses).toEqual([2, 2]);
      expect(result.objective).toBe(1);
      expect(await readFloat32(objective.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy();
    }
  });
}
