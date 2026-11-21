// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUSpatialWeightsLossViews} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {createPartitionLossPorts} from './partition-loss-ports';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {
  computeAZPRegionsOnCPU,
  computeMaxPRegionsOnCPU,
  computeSkaterRegionsOnCPU,
  computeWardRegionsOnCPU,
  CPUAZPRegionsOperation,
  CPUMaxPRegionsOperation,
  CPUSkaterRegionsOperation,
  CPUWardRegionsOperation,
  GPUAZPRegions,
  GPUAZPRegionsOperation,
  GPUMaxPRegions,
  GPUMaxPRegionsOperation,
  GPURegionPartitionEvaluation,
  GPUSkaterRegions,
  GPUSkaterRegionsOperation,
  GPUSpatialWeightsMinimumSpanningTree,
  GPUWardRegions,
  GPUWardRegionsOperation,
  type WardRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';

const UNSUPPORTED_STATUS_FIELDS = [
  'complete',
  'converged',
  'iterationLimitReached',
  'unresolvedCount'
] as const;
type UnsupportedStatusField = (typeof UNSUPPORTED_STATUS_FIELDS)[number];

it('exposes two distinct loss sources simultaneously in partition recovery fixtures', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    const status = createPartitionLossPorts(rig);
    const losses = getGPUSpatialWeightsLossViews({
      offsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
      neighbors: rig.input(Uint32Array.of(0), 'uint32', 1),
      weights: rig.input(Float32Array.of(1), 'float32', 1),
      status
    });
    expect(losses).toHaveLength(2);
    expect(losses[0]).not.toBe(losses[1]);
    expect(losses[0].buffer).not.toBe(losses[1].buffer);
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('refuses unsupported completeness evidence in the CPU regionalization functions', () => {
  const input = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(1, 0),
    weights: Float32Array.of(1, 1),
    values: Float32Array.of(0, 1),
    parameters: Uint32Array.of(0)
  };
  type CPUStatus = NonNullable<Parameters<typeof computeWardRegionsOnCPU>[0]['status']>;
  type ProducerStatus = CPUStatus & Partial<Record<UnsupportedStatusField, Uint32Array>>;
  const overflow = Uint32Array.of(1);
  const requiredCount = Uint32Array.of(3);
  const valid = {requiredCount, overflow} satisfies CPUStatus;
  // A caller without TypeScript can still pass a requirement without the final overflow view.
  const unqualified = {
    candidateOverflow: Uint32Array.of(0),
    requiredCount
  } as unknown as CPUStatus;
  const createContributors = [
    (status?: ProducerStatus) =>
      computeAZPRegionsOnCPU(
        {...input, status, initialLabels: Uint32Array.of(0, 1)},
        {id: 'cpu-azp', regionCapacity: 2}
      ),
    (status?: ProducerStatus) =>
      computeMaxPRegionsOnCPU(
        {...input, status, threshold: Float32Array.of(1), extensiveValues: Float32Array.of(1, 1)},
        {id: 'cpu-max-p', startCount: 1}
      ),
    (status?: ProducerStatus) =>
      computeWardRegionsOnCPU({...input, status}, {id: 'cpu-ward', targetRegionCount: 1}),
    (status?: ProducerStatus) =>
      computeSkaterRegionsOnCPU(
        {
          ...input,
          status,
          parameters: Uint32Array.of(2, 1),
          treeEdgeFlags: Uint32Array.of(1, 0),
          componentLabels: Uint32Array.of(0, 0)
        },
        {id: 'cpu-skater', maximumRegionCount: 2}
      )
  ];
  for (const create of createContributors) {
    expect(() => create(undefined)).not.toThrow();
    expect(() => create(valid)).not.toThrow();
    expect(() => create(unqualified)).toThrow(/weights\.status\.requiredCount/);
    for (const field of UNSUPPORTED_STATUS_FIELDS) {
      const status: ProducerStatus = {overflow: Uint32Array.of(0), [field]: Uint32Array.of(1)};
      expect(() => create(status)).toThrow(
        new RegExp(`^cpu-\\S+ weights\\.status\\.${field}.*omit`)
      );
    }
  }
});

it('refuses unsupported completeness evidence in named regionalization inputs before lowering', () => {
  const weights = {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'};
  const props = {id: 'named-regions', rowCount: 2, values: 'values', parameters: 'parameters'};
  const output = {labels: 'labels', status: 'status', objective: 'objective'};
  type NamedStatus = NonNullable<WardRegionsOperationProps['weights']['status']>;
  type ProducerStatus = NamedStatus & Partial<Record<UnsupportedStatusField, string>>;
  const valid = {overflow: 'overflow', requiredCount: 'required-count'} satisfies NamedStatus;
  const unqualified = {
    candidateOverflow: 'candidate-overflow',
    requiredCount: 'required-count'
  } as unknown as NamedStatus;
  const createContributors = [
    (cpu: boolean, status?: ProducerStatus) =>
      new (cpu ? CPUAZPRegionsOperation : GPUAZPRegionsOperation)({
        ...props,
        weights: {...weights, status},
        initialLabels: 'initial-labels',
        regionCapacity: 2,
        maximumIterations: 1,
        output
      }),
    (cpu: boolean, status?: ProducerStatus) =>
      new (cpu ? CPUMaxPRegionsOperation : GPUMaxPRegionsOperation)({
        ...props,
        weights: {...weights, status},
        threshold: 'threshold',
        extensiveValues: 'extensive-values',
        output
      }),
    (cpu: boolean, status?: ProducerStatus) =>
      new (cpu ? CPUWardRegionsOperation : GPUWardRegionsOperation)({
        ...props,
        weights: {...weights, status},
        targetRegionCount: 1,
        maximumIterations: 1,
        output
      }),
    (cpu: boolean, status?: ProducerStatus) =>
      new (cpu ? CPUSkaterRegionsOperation : GPUSkaterRegionsOperation)({
        ...props,
        weights: {...weights, status},
        edgeCount: 2,
        maximumRegionCount: 2,
        treeEdgeFlags: 'tree-edges',
        componentLabels: 'component-labels',
        output: {labels: 'labels'}
      })
  ];
  for (const cpu of [false, true]) {
    for (const create of createContributors) {
      expect(() => create(cpu)).not.toThrow();
      expect(() => create(cpu, valid)).not.toThrow();
      expect(() => create(cpu, unqualified)).toThrow(/weights\.status\.requiredCount/);
      for (const field of UNSUPPORTED_STATUS_FIELDS) {
        const status: ProducerStatus = {overflow: 'overflow', [field]: field};
        expect(() => create(cpu, status)).toThrow(
          new RegExp(`^named-regions weights\\.status\\.${field}.*omit`)
        );
      }
    }
  }
});

it('refuses CSR requirement counts without final overflow and unsupported iteration evidence', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    const weights: GPUSpatialWeights = {
      offsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
      neighbors: rig.input(Uint32Array.of(1, 0), 'uint32', 2),
      weights: rig.input(Float32Array.of(1, 1), 'float32', 2),
      information: {rowCount: 2, transform: 'binary', symmetric: true}
    };
    const uint = (length = 1) => rig.output('uint32', length).view;
    const float = (length = 1) => rig.output('float32', length).view;
    const values = float(2);
    const labels = uint(2);
    const parameters = uint();
    const initialLabels = uint(2);
    const treeEdgeFlags = uint(2);
    const componentLabels = uint(2);
    const partitionOutput = {labels, status: uint(6), objective: float()};
    const createContributors = [
      (weights: GPUSpatialWeights) =>
        new GPUAZPRegions({
          weights,
          values,
          initialLabels,
          parameters,
          regionCapacity: 2,
          maximumIterations: 1,
          output: partitionOutput
        }),
      (weights: GPUSpatialWeights) =>
        new GPUWardRegions({
          weights,
          values,
          parameters,
          targetRegionCount: 1,
          maximumIterations: 1,
          output: partitionOutput
        }),
      (weights: GPUSpatialWeights) =>
        new GPUMaxPRegions({
          weights,
          values,
          parameters,
          threshold: float(),
          extensiveValues: float(2),
          output: partitionOutput
        }),
      (weights: GPUSpatialWeights) =>
        new GPUSpatialWeightsMinimumSpanningTree({
          weights,
          values,
          output: {treeEdgeFlags, componentLabels}
        }),
      (weights: GPUSpatialWeights) =>
        new GPUSkaterRegions({
          weights,
          values,
          treeEdgeFlags,
          componentLabels,
          maximumRegionCount: 2,
          parameters: uint(2),
          output: {labels}
        }),
      (weights: GPUSpatialWeights) =>
        new GPURedcapSingleLinkage({weights, values, parameters, output: {labels}}),
      (weights: GPUSpatialWeights) =>
        new GPURegionPartitionEvaluation({weights, values, labels, output: {summary: float(32)}})
    ];
    // A complete two-edge CSR fits its allocation. A requirement of three proves that the
    // same allocation could be incomplete even when a candidate-overflow flag reads zero.
    const requiredCount = rig.input(Uint32Array.of(3), 'uint32', 1);
    const candidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const overflow = rig.input(Uint32Array.of(1), 'uint32', 1);
    const complete = rig.input(Uint32Array.of(0), 'uint32', 1);
    for (const create of createContributors) {
      expect(() => create(weights)).not.toThrow();
      expect(() => create({...weights, status: {requiredCount, overflow}})).not.toThrow();
    }
    for (const create of createContributors) {
      expect(() => create({...weights, status: {requiredCount, candidateOverflow}})).toThrow(
        /weights.status.requiredCount requires weights.status.overflow/
      );
      for (const field of UNSUPPORTED_STATUS_FIELDS) {
        const status: GPUSpatialWeights['status'] = {
          overflow: candidateOverflow,
          [field]: field === 'complete' || field === 'converged' ? complete : overflow
        };
        expect(() => create({...weights, status})).toThrow(
          new RegExp(`weights\\.status\\.${field}.*omit`)
        );
      }
    }
    const summary = float(32);
    const createChunked = (status: GPUSpatialWeights['status'], atMatrixLevel: boolean) =>
      new GPURegionPartitionEvaluation({
        weights: {
          rowCount: 2,
          status: atMatrixLevel ? status : undefined,
          chunks: [
            {
              rowStart: 0,
              rowCount: 2,
              weights: {...weights, status: atMatrixLevel ? undefined : status}
            }
          ]
        },
        values,
        labels,
        output: {summary}
      });
    for (const atMatrixLevel of [false, true]) {
      expect(() => createChunked({requiredCount, overflow}, atMatrixLevel)).not.toThrow();
      expect(() => createChunked({requiredCount, candidateOverflow}, atMatrixLevel)).toThrow(
        /weights(?:\.chunks\[0\]\.weights)?\.status\.requiredCount requires/
      );
      const path = atMatrixLevel ? 'weights' : 'weights\\.chunks\\[0\\]\\.weights';
      for (const field of UNSUPPORTED_STATUS_FIELDS) {
        expect(() =>
          createChunked({overflow: candidateOverflow, [field]: overflow}, atMatrixLevel)
        ).toThrow(new RegExp(`^region-partition-evaluation ${path}\\.status\\.${field}.*omit`));
      }
    }
  } finally {
    rig.destroy();
    device.destroy();
  }
});
