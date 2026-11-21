// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import type {GPUAZPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import type {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createAuditAZPProps} from './regionalization-audit-azp-fixtures';
import {createAuditMaxPInput, createAuditMaxPProps} from './regionalization-audit-max-p-fixtures';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';
import {
  getGPUAZPMaximumTableWordCount,
  getGPUAZPRegionsParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {
  getGPUMaxPMaximumSlotCount,
  getGPUMaxPRoundCounts,
  getGPUMaxPRegionsParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {
  getGPURegionPartitionEvaluationMaximumLabelCapacity,
  getGPURegionPartitionEvaluationMaximumRegionValueCount,
  getGPURegionPartitionEvaluationMaximumRowCount
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {
  countCPURegionInvalid,
  getCPURegionAdjacency,
  getCPURegionObjective,
  getCPURegionTable,
  validateCPURegionInput
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-region-inputs';
import {createAuditAZPInput} from './regionalization-audit-azp-fixtures';

it('audits population scaling and direct SSD with hand-derived symmetric and missing data', () => {
  // Eight entries: mean zero, sum of squares 40, population variance 5. Constant second
  // column is zero after scaling. Two labels containing {-3,-1,1,3} each have SSD 20/5 = 4.
  const values = Float32Array.of(-3, 7, -1, 7, 1, 7, 3, 7, -3, 7, -1, 7, 1, 7, 3, 7);
  const table = getCPURegionTable(values, 2);
  expect(Array.from(table)).toEqual(
    Array.from(values, (value, index) => (index % 2 ? 0 : value / Math.sqrt(5)))
  );
  expect(getCPURegionObjective(table, Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1), 2)).toBeCloseTo(
    8,
    14
  );
  const missing = getCPURegionTable(
    Float32Array.of(NaN, -3, -1, 1, 3, NaN, Infinity, -Infinity),
    1
  );
  expect(Array.from(missing)).toEqual([
    NaN,
    -3 / Math.sqrt(5),
    -1 / Math.sqrt(5),
    1 / Math.sqrt(5),
    3 / Math.sqrt(5),
    NaN,
    Infinity,
    -Infinity
  ]);
  expect(getCPURegionTable(values, 2, false)).toEqual(Float64Array.from(values));
});

it('keeps every nonconstant CPU table entry at tiny and large float32 scales', () => {
  // Mean zero and population variance five in the first column; constant second column.
  // Multiplying either column by a positive power of two leaves its z-scores unchanged.
  // Even the smallest float32 inputs have representable nonzero squared deviations in float64.
  const values = Float32Array.of(-3, 7, -1, 7, 1, 7, 3, 7, -3, 7, -1, 7, 1, 7, 3, 7);
  const expected = [
    -1.3416407864998738, 0, -0.4472135954999579, 0, 0.4472135954999579, 0, 1.3416407864998738, 0,
    -1.3416407864998738, 0, -0.4472135954999579, 0, 0.4472135954999579, 0, 1.3416407864998738, 0
  ];
  for (const exponent of [-149, -80, 64, 120]) {
    const table = getCPURegionTable(
      values.map(value => value * 2 ** exponent),
      2
    );
    expected.forEach((value, index) => {
      expect(Number.isFinite(table[index])).toBe(true);
      expect(Math.abs(table[index] - value)).toBeLessThanOrEqual(2e-15);
    });
  }
});

it('audits directed adjacency union and used-slot invalid counts independently of weights', () => {
  const input = createAuditAZPInput();
  expect(validateCPURegionInput(input, 2, 1)).toBe(9);
  expect(getCPURegionAdjacency(input)).toEqual([
    [1],
    [2, 0],
    [3, 1],
    [4, 2],
    [5, 3],
    [6, 4],
    [7, 5],
    [6],
    []
  ]);
  input.values[0] = NaN;
  input.values[3] = Infinity;
  input.neighbors = Uint32Array.of(1, 2, 3, 4, 5, 6, 99, 99);
  input.weights = new Float32Array(8);
  // Two bad feature entries plus one used bad neighbor. The unused tail slot is ignored.
  expect(countCPURegionInvalid(input)).toBe(3);
});

it('audits public memory limits by bytes per record and seed packing by literal words', () => {
  // A deliberately unaligned byte limit exercises floor division and the trailing CSR offset.
  const device = {limits: {maxStorageBufferBindingSize: 103}} as Device;
  expect(getGPUAZPMaximumTableWordCount(device)).toBe(25); // four-byte float
  expect(getGPUMaxPMaximumSlotCount(device)).toBe(6); // sixteen-byte nested adjacency slot
  expect(getGPURegionPartitionEvaluationMaximumRowCount(device)).toBe(12); // two uint32 links
  expect(getGPURegionPartitionEvaluationMaximumRegionValueCount(device)).toBe(25);
  expect(getGPURegionPartitionEvaluationMaximumLabelCapacity(device)).toBe(24); // one extra offset
  expect(getGPUMaxPRoundCounts(9, 4)).toEqual({growthRounds: 4, enclaveRounds: 4});
  expect(getGPUMaxPRoundCounts(9, 100)).toEqual({growthRounds: 20, enclaveRounds: 10});
  for (const pack of [getGPUAZPRegionsParameterValues, getGPUMaxPRegionsParameterValues]) {
    expect(Array.from(pack())).toEqual([0]);
    expect(Array.from(pack({seed: 4294967295}))).toEqual([4294967295]);
    expect(() => pack({seed: -1})).toThrow(/seed/);
    expect(() => pack({seed: 1.5})).toThrow(/seed/);
  }
});

it('retains loss bindings after caller-map mutation and budgets every suppression node', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const maxP of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const input = maxP ? createAuditMaxPInput() : createAuditAZPInput();
      const options = maxP ? createAuditMaxPProps() : createAuditAZPProps();
      const status: {overflow?: string; candidateOverflow?: string} = {
        overflow: 'weightsOverflow',
        candidateOverflow: 'weightsCandidateOverflow'
      };
      const weights = {...options.weights, status};
      const operation = maxP
        ? new GPUMaxPRegionsOperation({...createAuditMaxPProps(), weights})
        : new GPUAZPRegionsOperation({...createAuditAZPProps(), weights});
      status.overflow = undefined;
      status.candidateOverflow = undefined;
      const arrays = getRegionalizationOperationInputs({
        ...input,
        status: {overflow: Uint32Array.of(1), candidateOverflow: Uint32Array.of(0)}
      });
      const views = new Map<string, GraphDataView>();
      const getView = <Format extends GPUGraphVectorFormat>(
        port: string
      ): GraphDataView<Format> => {
        if (!views.has(port)) {
          const output = operation.outputs[port];
          views.set(
            port,
            createTransientView(
              graph,
              port,
              output?.format ?? arrays[port].format,
              output?.length ?? arrays[port].values.length
            )
          );
        }
        return views.get(port)! as GraphDataView<Format>;
      };
      const contributor = operation.createContributor(getView) as GPUAZPRegions | GPUMaxPRegions;
      expect(contributor.props.weights.status?.overflow).toBe(views.get('weightsOverflow'));
      expect(contributor.props.weights.status?.candidateOverflow).toBe(
        views.get('weightsCandidateOverflow')
      );
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(
        kernels.filter(kernel => kernel.source.includes('if (loss[lossOffset] != 0u)'))
      ).toHaveLength(6);
      operation.destroy();
    }
  } finally {
    device.destroy();
  }
});
