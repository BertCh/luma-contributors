// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createSparsePartitionComparison,
  createStagedPartitionInput
} from './partition-provenance-fixtures';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  CPUMaxPRegionsOperation,
  type CPUMaxPRegionsResult
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {CPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {CPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {CPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {Buffer} from 'node:buffer';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUWardRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {
  GPUMaxPRegions,
  decodeGPUMaxPRegionsEvidence
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {compactRegionLabels} from '../../../src/gpu-spatial-analysis/spatial-regionalization/compact-region-labels';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {computeWardRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {createPartitionInput} from './partition-budget-fixtures';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createRig(rowCount = 5) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const uint = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const float = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', length);
  const weights = {
    offsets: uint('offsets', rowCount + 1),
    neighbors: uint('neighbors', rowCount),
    weights: float('weights', rowCount)
  };
  const output = {
    labels: uint('labels', rowCount),
    objective: float('objective', 1),
    status: uint('status', 6)
  };
  return {device, graph, uint, float, weights, output, rowCount};
}

it('B1 reports the Ward caller ID, unknown schedule and both accepted values', () => {
  const rig = createRig();
  try {
    expect(
      () =>
        new GPUWardRegions({
          id: 'census',
          weights: rig.weights,
          values: rig.float('values', 5),
          parameters: rig.uint('parameters', 1),
          targetRegionCount: 2,
          maximumIterations: 1,
          mergeSchedule: 'unknown' as 'sequential',
          output: rig.output
        })
    ).toThrow(/census.*mergeSchedule.*sequential.*mutual-nearest.*unknown/);
  } finally {
    rig.device.destroy();
  }
});

for (const maximumIterations of [0, 1])
  it(`B2 evidence shader has no reserved identifiers, refinement ${maximumIterations}`, () => {
    const rig = createRig();
    try {
      new GPUMaxPRegions({
        id: 'probe',
        weights: rig.weights,
        values: rig.float('values', 5),
        extensiveValues: rig.float('extensive', 5),
        threshold: rig.float('threshold', 1),
        parameters: rig.uint('parameters', 1),
        startCount: 1,
        maximumGrowthSteps: 3,
        maximumIterations,
        output: {...rig.output, evidence: rig.uint('evidence', 5)}
      }).getCommandNodes(rig.graph);
      const record = getWGSLKernelNodeRecord(rig.graph, 'probe-evidence')!;
      const identifiers = [...record.body.matchAll(/\b(?:let|var)\s+(\w+)/g)].map(
        match => match[1]
      );
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    } finally {
      rig.device.destroy();
    }
  });

it('B3 includes unfinished enclave propagation in the selected construction evidence', () => {
  const input = {
    ...createPartitionInput(
      [[1], [0, 2], [1, 3], [2, 4], [3]],
      [0, 1, 2, 3, 4],
      [5, 0, 0, 0, 0],
      1
    ),
    parameters: Uint32Array.of(12)
  };
  for (const maximumGrowthSteps of [3, 4]) {
    const result = computeMaxPRegionsOnCPU(input, {
      standardize: false,
      startCount: 1,
      maximumGrowthSteps
    });
    // One adjacent row can join the completed region each enclave round.
    expect(Array.from(result.labels)).toEqual(
      maximumGrowthSteps === 3 ? [0, 0, 0, 0, 1] : [0, 0, 0, 0, 0]
    );
    expect(Array.from(result.evidence.slice(0, 2))).toEqual(
      maximumGrowthSteps === 3 ? [1, 0] : [0, 1]
    );
  }
});

it('B4 refuses a label completion input without a provenance output', () => {
  const rig = createRig();
  try {
    expect(
      () =>
        new GPURegionPartitionEvaluation({
          id: 'provenance',
          values: rig.float('values', 5),
          labels: rig.output.labels,
          labelsComplete: rig.uint('complete', 1),
          output: {summary: rig.float('summary', 9)}
        })
    ).toThrow(/provenance.*output.labelsComplete/);
    expect(() =>
      new GPURegionPartitionEvaluation({
        values: rig.float('ordinary-values', 5),
        labels: rig.output.labels,
        output: {
          summary: rig.float('ordinary-summary', 9),
          labelsComplete: rig.uint('ordinary-complete', 1)
        }
      }).getCommandNodes(rig.graph)
    ).not.toThrow();
  } finally {
    rig.device.destroy();
  }
});

it('B5 reads guarded unaligned Buffer subviews without reading backing allocation bytes', () => {
  const bytes = Buffer.alloc(49, 0xa5);
  const labels = bytes.subarray(5, 25);
  [0, 0, 2, 2, 0xffffffff].forEach((value, index) => labels.writeUInt32LE(value, index * 4));
  expect(Array.from(compactRegionLabels(labels).labels)).toEqual([0, 0, 1, 1, 0xffffffff]);
  const evidence = bytes.subarray(25, 45);
  [0, 1, 1, 0, 1].forEach((value, index) => evidence.writeUInt32LE(value, index * 4));
  expect(decodeGPUMaxPRegionsEvidence(evidence)).toEqual({
    constructionLimitReached: false,
    constructionSettled: true,
    improvementEnabled: true,
    improvementConverged: false,
    iterationLimit: true
  });
});

it('S1 retains admission at the existing command budget without evidence', () => {
  const rig = createRig(1412);
  try {
    const contributor = new GPUMaxPRegions({
      weights: {
        ...rig.weights,
        status: {overflow: rig.uint('overflow', 1), candidateOverflow: rig.uint('candidate', 1)}
      },
      values: rig.float('values', 1412),
      extensiveValues: rig.float('extensive', 1412),
      threshold: rig.float('threshold', 1),
      parameters: rig.uint('parameters', 1),
      maximumGrowthSteps: 2825,
      maximumIterations: 256,
      output: rig.output
    });
    expect(contributor.estimatedCommandNodeCount).toBe(32768);
  } finally {
    rig.device.destroy();
  }
});

it('S2 preserves the failed Ward CSR operand in structured details', () => {
  try {
    computeWardRegionsOnCPU(
      {
        offsets: Uint32Array.of(1, 0),
        neighbors: new Uint32Array(),
        values: Float32Array.of(0),
        parameters: Uint32Array.of(0)
      },
      {id: 'ward-csr', targetRegionCount: 1}
    );
    throw new Error('expected CSR rejection');
  } catch (error) {
    expect(error).toMatchObject({received: 1, path: 'offsets[0]'});
  }
});

it('S3 rejects overlaps for the new outputs and accepts disjoint slices', () => {
  const rig = createRig();
  try {
    const storage = rig.uint('storage', 16);
    const slice = (offset: number, length: number) =>
      rig.graph.createDataView(storage.buffer, {format: 'uint32', byteOffset: offset * 4, length});
    const values = rig.float('values', 5);
    const summary = rig.float('summary', 9);
    for (const offset of [0, 5]) {
      const construct = () =>
        new GPURegionPartitionEvaluation({
          values,
          labels: rig.output.labels,
          output: {summary, regionSizes: slice(0, 5), labelsComplete: slice(offset, 1)}
        });
      if (offset === 0) expect(construct).toThrow(/overlap/);
      else expect(construct).not.toThrow();
    }
    for (const offset of [0, 5]) {
      const construct = () =>
        new GPUMaxPRegions({
          weights: rig.weights,
          values,
          extensiveValues: rig.float(`extensive-${offset}`, 5),
          threshold: rig.float(`threshold-${offset}`, 1),
          parameters: rig.uint(`parameters-${offset}`, 1),
          output: {...rig.output, labels: slice(0, 5), evidence: slice(offset, 5)}
        });
      if (offset === 0) expect(construct).toThrow(/overlap/);
      else expect(construct).not.toThrow();
    }
  } finally {
    rig.device.destroy();
  }
});

it('B7 replays the three device comparison declarations with distinct resource IDs', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    expect(() => createSparsePartitionComparison(rig)).not.toThrow();
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('B6 composes an all-missing partition without changing its true count', () => {
  const compact = compactRegionLabels(Uint32Array.of(0xffffffff, 0xffffffff));
  expect(compact.count).toBe(0);
  const rig = createRig(2);
  try {
    const props = {
      values: rig.float('values', 2),
      labels: rig.output.labels,
      output: {summary: rig.float('summary', 9)}
    };
    expect(
      () => new GPURegionPartitionEvaluation({...props, labelCapacity: compact.count})
    ).toThrow(/labelCapacity/);
    expect(() =>
      new GPURegionPartitionEvaluation({
        ...props,
        labelCapacity: Math.max(1, compact.count)
      }).getCommandNodes(rig.graph)
    ).not.toThrow();
  } finally {
    rig.device.destroy();
  }
});

it('B8 makes the final tile numerically distinguishable from an omitted tile', () => {
  const input = createStagedPartitionInput();
  let sum = 0;
  for (let row = 0; row < input.rowCount; row++)
    sum += input.values[row * input.columnCount] - input.values[0];
  const last = input.values[(input.rowCount - 1) * input.columnCount] - input.values[0];
  const completeMean = sum / input.rowCount;
  const omittedMean = (sum - last) / (input.rowCount - 1);
  expect(Math.abs(completeMean - omittedMean)).toBeGreaterThan(0.00001);
});

it('S2 uses the resolved default operation IDs in all CPU computations', async () => {
  const common = {
    rowCount: 2,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    values: 'values',
    parameters: 'parameters',
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  };
  const operations = [
    [
      new CPUMaxPRegionsOperation({
        ...common,
        extensiveValues: 'extensiveValues',
        threshold: 'threshold'
      }),
      'max-p-regions'
    ],
    [
      new CPUAZPRegionsOperation({
        ...common,
        initialLabels: 'initialLabels',
        regionCapacity: 2,
        maximumIterations: 1
      }),
      'azp-regions'
    ],
    [
      new CPUWardRegionsOperation({...common, targetRegionCount: 1, maximumIterations: 1}),
      'ward-regions'
    ],
    [
      new CPUSkaterRegionsOperation({
        ...common,
        edgeCount: 2,
        treeEdgeFlags: 'treeEdgeFlags',
        componentLabels: 'componentLabels',
        maximumRegionCount: 2
      }),
      'skater-regions'
    ]
  ] as const;
  const vector = (values: Uint32Array | Float32Array): GPUCPUVector => ({
    format: values instanceof Float32Array ? 'float32' : 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const inputs = Object.fromEntries(
    Object.entries({
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 0),
      weights: Float32Array.of(1, 1),
      values: new Float32Array(),
      parameters: Uint32Array.of(2, 1),
      initialLabels: Uint32Array.of(0, 1),
      extensiveValues: Float32Array.of(1, 1),
      threshold: Float32Array.of(1),
      treeEdgeFlags: Uint32Array.of(1, 1),
      componentLabels: Uint32Array.of(0, 0)
    }).map(([name, values]) => [name, vector(values)])
  );
  for (const [operation, id] of operations) {
    await expect(async () => operation.execute({inputs})).rejects.toMatchObject({id});
  }
});

it('S4 keeps old typed result fixtures valid while compute returns required evidence', () => {
  const legacy: CPUMaxPRegionsResult = {
    labels: Uint32Array.of(0),
    objective: Float32Array.of(0),
    status: new Uint32Array(6)
  };
  expect(legacy.evidence).toBeUndefined();
  const evidence: Uint32Array = computeMaxPRegionsOnCPU(createPartitionInput()).evidence;
  expect(evidence).toHaveLength(5);
});

it('B3 carries the enclave evidence through the CPU operation output', async () => {
  const input = {
    ...createPartitionInput(
      [[1], [0, 2], [1, 3], [2, 4], [3]],
      [0, 1, 2, 3, 4],
      [5, 0, 0, 0, 0],
      1
    ),
    parameters: Uint32Array.of(12)
  };
  const inputs = Object.fromEntries(
    Object.entries(input).map(([name, values]) => [
      name,
      {
        format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      }
    ])
  );
  for (const maximumGrowthSteps of [3, 4]) {
    const operation = new CPUMaxPRegionsOperation({
      rowCount: 5,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      values: 'values',
      extensiveValues: 'extensiveValues',
      threshold: 'threshold',
      parameters: 'parameters',
      standardize: false,
      startCount: 1,
      maximumGrowthSteps,
      output: {labels: 'labels', objective: 'objective', status: 'status', evidence: 'evidence'}
    });
    const result = await operation.execute({inputs});
    expect(result['evidence']).toEqual(
      maximumGrowthSteps === 3 ? Uint32Array.of(1, 0, 0, 0, 0) : Uint32Array.of(0, 1, 0, 0, 0)
    );
  }
});
