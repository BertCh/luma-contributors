// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {computeAZPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {
  computeWardRegionsOnCPU,
  CPUWardRegionsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {GPUWardRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createMaxPGrowthInput} from './regionalization-operation-test-utils';
import {runMaxPRegionsMirror} from './max-p-construction-mirror';
import {makeLikelihoodCPUInputs} from '../spatial-interpolation/likelihoods-operation-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  createRegionalizationBudgetPath,
  createMaxPEnclaveBudgetPath,
  createWardLossPath
} from './regionalization-final1-fixtures';

it('spatial-3:F7 publishes only one AZP boundary move at a one-round budget', () => {
  const result = computeAZPRegionsOnCPU(createRegionalizationBudgetPath(), {
    standardize: false,
    regionCapacity: 2,
    maximumIterations: 1
  });
  expect(Array.from(result.labels)).toEqual([0, 0, 0, 0, 1, 1]);
  expect(Array.from(result.status)).toEqual([1, 0, 1, 0]);
  expect(result.objective[0]).toBe(50);
});

it('spatial-3:F7 publishes the untouched Ward partition with a zero-round budget', () => {
  const input = createRegionalizationBudgetPath();
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    const result = computeWardRegionsOnCPU(input, {
      standardize: false,
      targetRegionCount: 4,
      maximumIterations: 0,
      mergeSchedule
    });
    expect(Array.from(result.labels)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(result.status)).toEqual([0, 0, 1, 0]);
    expect(result.objective[0]).toBe(0);
  }
});

it('spatial-3:F7 bounds Max-p growth and enclave assignment to one step each', () => {
  const input = createMaxPGrowthInput();
  const expected = runMaxPRegionsMirror({
    ...input,
    columnCount: 2,
    extensive: input.extensiveValues,
    threshold: input.threshold[0],
    seed: 3,
    startCount: 1,
    standardize: false,
    maximumGrowthSteps: 1
  });
  const result = computeMaxPRegionsOnCPU(input, {
    columnCount: 2,
    standardize: false,
    startCount: 1,
    maximumGrowthSteps: 1
  });
  expect(expected.infeasibleRegionCount).toBeGreaterThan(0);
  expect(Array.from(result.labels)).toEqual(Array.from(expected.labels));
  expect(Array.from(result.status)).toEqual([
    expected.regionCount,
    0,
    expected.infeasibleRegionCount,
    0,
    expected.selectedStart,
    0
  ]);
});

it('spatial-3:F7 stops enclave propagation at one edge per permitted step', () => {
  const result = computeMaxPRegionsOnCPU(createMaxPEnclaveBudgetPath(), {
    standardize: false,
    startCount: 1,
    maximumGrowthSteps: 1
  });
  expect(result.labels[1]).toBe(result.labels[0]);
  expect(result.labels[2]).not.toBe(result.labels[0]);
  expect(result.status[1]).toBe(0);
});

it('spatial-3:F8 suppresses Ward inference after either adjacency loss flag', () => {
  const complete = computeWardRegionsOnCPU(createWardLossPath(false), {
    standardize: false,
    targetRegionCount: 2,
    maximumIterations: 1
  });
  const unflagged = computeWardRegionsOnCPU(createWardLossPath(true), {
    standardize: false,
    targetRegionCount: 2,
    maximumIterations: 1
  });
  expect(complete.objective[0]).toBe(0.5);
  expect(unflagged.objective[0]).toBe(40.5);
  for (const reason of ['overflow', 'candidateOverflow'] as const) {
    const result = computeWardRegionsOnCPU(
      {...createWardLossPath(true), status: {[reason]: Uint32Array.of(1)}},
      {
        standardize: false,
        targetRegionCount: 2,
        maximumIterations: 1
      }
    );
    expect(Array.from(result.labels)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    expect(Number.isNaN(result.objective[0])).toBe(true);
    expect(result.status[1]).toBe(0);
  }
});

it('spatial-3:F8 lowers Ward loss flags into readers in both schedules', () => {
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `ward-loss-${serial++}`, format, length);
    const candidateOverflow = view('uint32', 1);
    let currentLoss = candidateOverflow;
    let lossReads = 0;
    const contributor = new GPUWardRegions({
      weights: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 2),
        weights: view('float32', 2),
        status: Object.defineProperty({overflow: view('uint32', 1)}, 'candidateOverflow', {
          enumerable: false,
          get() {
            lossReads++;
            return currentLoss;
          }
        })
      },
      values: view('float32', 3),
      parameters: view('uint32', 1),
      targetRegionCount: 2,
      maximumIterations: 1,
      mergeSchedule,
      output: {labels: view('uint32', 3), objective: view('float32', 1), status: view('uint32', 5)}
    });
    currentLoss = view('uint32', 1);
    expect(contributor.props.weights.status?.candidateOverflow).toBe(candidateOverflow);
    const nodes = contributor.getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes)
      .map(kernel => kernel.source)
      .join('\n');
    expect(sources.includes('inputOverflow[')).toBe(true);
    expect(sources.includes('inputCandidateOverflow[')).toBe(true);
    expect(lossReads).toBe(1);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  }
  const operation = new GPUWardRegionsOperation({
    rowCount: 3,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: Object.create({
        get overflow() {
          return 'loss';
        },
        get candidateOverflow() {
          return 'candidateLoss';
        }
      })
    },
    values: 'values',
    parameters: 'parameters',
    targetRegionCount: 2,
    maximumIterations: 1,
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  });
  expect(operation.inputs).toMatchObject({
    weightsOverflow: 'loss',
    weightsCandidateOverflow: 'candidateLoss'
  });
});

it('spatial-3:G4 retains getter-backed Ward loss ports and recovers after caller-map mutation', async () => {
  let lossName = 'loss';
  let lossReads = 0;
  const props = {
    rowCount: 3,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: Object.defineProperty({}, 'candidateOverflow', {
        get() {
          lossReads++;
          return lossName;
        },
        configurable: true,
        enumerable: false
      }) as {candidateOverflow?: string}
    },
    values: 'values',
    parameters: 'parameters',
    standardize: false,
    targetRegionCount: 2,
    maximumIterations: 1,
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  };
  const gpu = new GPUWardRegionsOperation(props);
  const cpu = new CPUWardRegionsOperation(props);
  lossName = 'unbound';
  delete props.weights.status.candidateOverflow;
  expect(lossReads).toBe(2);
  const scene = createWardLossPath(true);
  const loss = Uint32Array.of(1);
  const arguments_ = {
    inputs: makeLikelihoodCPUInputs({
      offsets: {format: 'uint32', values: scene.offsets},
      neighbors: {format: 'uint32', values: scene.neighbors},
      weights: {format: 'float32', values: scene.weights},
      values: {format: 'float32', values: scene.values},
      parameters: {format: 'uint32', values: scene.parameters},
      weightsCandidateOverflow: {format: 'uint32', values: loss}
    })
  };
  const result = await cpu.execute(arguments_);
  expect(Array.from(result['labels'] as Uint32Array)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  expect(Number.isNaN((result['objective'] as Float32Array)[0])).toBe(true);
  expect(Array.from(result['status'] as Uint32Array)).toEqual([0, 0, 0, 0]);
  loss[0] = 0;
  const recovered = await cpu.execute(arguments_);
  expect(Array.from(recovered['labels'] as Uint32Array)).toEqual([0, 1, 1]);
  expect((recovered['objective'] as Float32Array)[0]).toBe(40.5);
  expect(Array.from(recovered['status'] as Uint32Array)).toEqual([1, 1, 0, 0]);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = gpu.createContributor(
    createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 4},
      neighbors: {format: 'uint32', length: 2},
      weights: {format: 'float32', length: 2},
      values: {format: 'float32', length: 3},
      parameters: {format: 'uint32', length: 1},
      weightsCandidateOverflow: {format: 'uint32', length: 1},
      labels: {format: 'uint32', length: 3},
      objective: {format: 'float32', length: 1},
      status: {format: 'uint32', length: 4}
    })
  );
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))
    .map(kernel => kernel.source)
    .join('\n');
  expect(sources.includes('inputCandidateOverflow[')).toBe(true);
});
