// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUWardRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {computeWardRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {CPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {computeSkaterRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {CPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createStandardizeColumnsNode} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-inputs';
import {getCPURegionTable} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-region-inputs';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {getWorkgroupArgminWGSL} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-wgsl';

useCPUExecutorTestHooks();

it('retains the negative tiny residual in the CPU standardized table (review note 3)', () => {
  // Exact mean 2^-54; population deviation rounds to sqrt(1/2).
  const score = getCPURegionTable(Float32Array.of(1, -1, 2 ** -52, 0), 1)[3];
  expect(Number.isFinite(score)).toBe(true);
  expect(score).toBeLessThan(0);
  expect(Math.abs(score - -7.850462293418875e-17)).toBeLessThan(1e-30);
});

it('review 2 R1 keeps small Ward differences beside a distant first row in every form', async () => {
  for (const data of [
    [1000000, 0, 2 / 128, 3 / 128],
    [2 ** 60, 0, 2, 3]
  ]) {
    const expectedObjective = data[2] === 2 ? 0.5 : 1 / 32768;
    for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
      const {device, graph, memory, view} = createScene();
      try {
        const input = {
          offsets: Uint32Array.of(0, 1, 3, 5, 6),
          neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2),
          values: Float32Array.from(data),
          parameters: Uint32Array.of(0)
        };
        const props = {
          id: 'ward-origins',
          rowCount: 4,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          values: 'values',
          parameters: 'parameters',
          standardize: false,
          targetRegionCount: 3,
          maximumIterations: 1,
          mergeSchedule,
          output: {labels: 'labels', objective: 'objective', status: 'status'}
        };
        const nodes = new GPUWardRegions({
          ...props,
          weights: {
            offsets: view('uint32', 5, input.offsets),
            neighbors: view('uint32', 6, input.neighbors),
            weights: view('float32', 6, new Float32Array(6).fill(1))
          },
          values: view('float32', 4, input.values),
          parameters: view('uint32', 1, input.parameters),
          output: {
            labels: view('uint32', 4),
            objective: view('float32', 1),
            status: view('uint32', 4)
          }
        }).getCommandNodes(graph);
        const node = (suffix: string) => nodes.find(candidate => candidate.id.endsWith(suffix))!;
        const binding = (suffix: string, name: string): GraphDataView =>
          getWGSLKernelNodeRecord(graph, node(suffix).id)!.bindings.find(
            candidate => candidate.name === name
          )!.view;
        executeCommandNodesOnCPU([node('-initialize-status'), node('-initialize-regions')], memory);
        memory.write(binding('-contract-0-costs', 'counts'), Uint32Array.of(3, 0));
        memory.write(binding('-contract-0-costs', 'edges'), Uint32Array.of(0, 1, 1, 2, 2, 3));
        memory.write(binding('-contract-0-costs', 'previousEdges'), Uint32Array.of(0, 1, 2));
        memory.write(binding('-contract-0-costs', 'previousCosts'), Float32Array.of(NaN, NaN, NaN));
        executeCommandNodesOnCPU([node('-contract-0-begin'), node('-contract-0-costs')], memory);
        const costs = memory.readFloat32(binding('-contract-0-costs', 'costs'), 3);
        // Independent singleton Ward cost is (a-b)^2 / 2; costs[2] is the unique minimum.
        expect(costs[1]).toBe(data[2] ** 2 / 2);
        expect(costs[2]).toBe(expectedObjective);
        if (mergeSchedule === 'sequential') {
          memory.write(binding('-iteration-0-apply', 'first'), Uint32Array.of(2));
          memory.write(binding('-iteration-0-apply', 'second'), Uint32Array.of(3));
          memory.write(binding('-iteration-0-apply', 'cost'), Float32Array.of(expectedObjective));
          executeCommandNodesOnCPU([node('-iteration-0-apply')], memory);
          memory.write(binding('-iteration-0-apply', 'dispatch'), Uint32Array.of(1, 1, 1));
        } else {
          memory.write(
            binding('-round-0-merge-regions', 'regionOther'),
            Uint32Array.of(0xffffffff, 0xffffffff, 3, 2)
          );
          memory.write(binding('-round-0-merge-regions', 'mergeFlag'), Uint32Array.of(0, 0, 1, 0));
          executeCommandNodesOnCPU([node('-round-0-merge-regions')], memory);
        }
        memory.write(binding('-contract-0-costs', 'counts'), Uint32Array.of(1, 0));
        memory.write(binding('-contract-0-costs', 'edges'), Uint32Array.of(1, 2));
        memory.write(binding('-contract-0-costs', 'previousCosts'), Float32Array.of(NaN));
        executeCommandNodesOnCPU([node('-contract-0-begin'), node('-contract-0-costs')], memory);
        const mergedCost = memory.readFloat32(binding('-contract-0-costs', 'costs'), 1)[0];
        // Cost of joining singleton 0 to the two-row mean 2.5: (2/3)*2.5^2, scaled by 1/128^2.
        const expectedMergedCost = data[2] === 2 ? 25 / 6 : 25 / 98304;
        expect(Number.isFinite(mergedCost)).toBe(true);
        expect(Math.abs(mergedCost / expectedMergedCost - 1)).toBeLessThan(1e-6);
        const cpu = computeWardRegionsOnCPU(input, props);
        expect(Array.from(cpu.labels)).toEqual([0, 1, 2, 2]);
        expect(Array.from(cpu.status)).toEqual([1, 1, 0, 0]);
        expect(Array.from(cpu.objective)).toEqual([expectedObjective]);
        const result = await new CPUWardRegionsOperation(props).execute({
          inputs: Object.fromEntries(
            Object.entries({...input, weights: new Float32Array(6).fill(1)}).map(
              ([name, values]) => [
                name,
                {
                  format:
                    values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
                  length: values.length,
                  chunks: [{values, length: values.length}],
                  pack: () => values
                }
              ]
            )
          )
        });
        expect(Array.from(result['labels'] as Uint32Array)).toEqual([0, 1, 2, 2]);
        expect(Array.from(result['status'] as Uint32Array)).toEqual([1, 1, 0, 0]);
        expect(Array.from(result['objective'] as Float32Array)).toEqual([expectedObjective]);
      } finally {
        device.destroy();
      }
    }
  }
});

it('W5-17 preserves the sign of a genuine sub-ULP standardized residual', () => {
  const {device, graph, memory, view} = createScene();
  try {
    const values = view('float32', 4, Float32Array.of(1, 2, -0.6, 0.8));
    const scaled = view('float32', 4);
    createStandardizeColumnsNode(graph, {
      id: 'sign-standardize',
      operation: 'ResidualSignReview',
      values,
      scaled,
      rowCount: 4,
      columnCount: 1
    });
    const record = getWGSLKernelNodeRecord(graph, 'sign-standardize')!;
    const node = createWGSLKernelNode(graph, {
      id: 'sign-standardize-serial',
      operation: 'ResidualSignReview',
      invocationCount: 1,
      workgroupSize: 2,
      bindings: record.bindings,
      body: record.body.replaceAll('storageBarrier();', ''),
      declarations: record
        .declarations!.replace('const LANES: u32 = 256u;', 'const LANES: u32 = 1u;')
        .replaceAll('workgroupBarrier();', '')
        .replaceAll('storageBarrier();', '')
    });
    executeCommandNodesOnCPU([node], memory);
    const actual = memory.readFloat32(scaled)[3];
    // Float64 population standardization of the stored f32 inputs: mean=0.7999999970197678,
    // last residual=1.4901161193847656e-8, deviation=0.9273618585478383.
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeGreaterThan(0);
    expect(Math.abs(actual - 1.6068335e-8)).toBeLessThan(1e-13);
    const cpuScore = getCPURegionTable(Float32Array.of(1, 2, -0.6, 0.8), 1)[3];
    expect(Number.isFinite(cpuScore)).toBe(true);
    expect(cpuScore).toBeGreaterThan(0);
    expect(Math.abs(cpuScore - 1.6068335e-8)).toBeLessThan(1e-13);
  } finally {
    device.destroy();
  }
});

it('review 2 R3 retains a residual mean correction for nearly centered repeated values', () => {
  const {device, graph, memory, view} = createScene();
  try {
    const rowCount = 30000;
    const data = new Float32Array(rowCount).fill(0.3);
    data[0] = 0;
    data[1] = 0.5999;
    const values = view('float32', rowCount, data);
    const scaled = view('float32', rowCount);
    createStandardizeColumnsNode(graph, {
      id: 'residual-standardize',
      operation: 'ResidualMeanReview',
      values,
      scaled,
      rowCount,
      columnCount: 1
    });
    const record = getWGSLKernelNodeRecord(graph, 'residual-standardize')!;
    const node = createWGSLKernelNode(graph, {
      id: 'residual-standardize-serial',
      operation: 'ResidualMeanReview',
      invocationCount: 1,
      workgroupSize: 2,
      bindings: record.bindings,
      body: record.body.replaceAll('storageBarrier();', ''),
      declarations: record
        .declarations!.replace('const LANES: u32 = 256u;', 'const LANES: u32 = 1u;')
        .replaceAll('workgroupBarrier();', '')
        .replaceAll('storageBarrier();', '')
    });
    executeCommandNodesOnCPU([node], memory);
    const actual = memory.readFloat32(scaled);
    expect(Number.isFinite(actual[2])).toBe(true);
    expect(actual[2]).toBeGreaterThan(0);
    expect(Math.abs(actual[2] - 1.36128029e-6)).toBeLessThan(1e-8);
  } finally {
    device.destroy();
  }
});

it('review 2 R4 ignores an extreme constant feature separately in each forest component', async () => {
  for (const withIsland of [false, true]) {
    const {device, graph, memory, view} = createScene();
    try {
      const rowCount = withIsland ? 3 : 2;
      const input = {
        offsets: Uint32Array.from(withIsland ? [0, 1, 2, 2] : [0, 1, 2]),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1),
        values: Float32Array.from(
          withIsland ? [2e38, -1e19, 2e38, 1e19, -2e38, 7] : [2e38, -1e19, 2e38, 1e19]
        ),
        treeEdgeFlags: Uint32Array.of(1, 0),
        componentLabels: Uint32Array.from(withIsland ? [0, 0, 2] : [0, 0]),
        parameters: Uint32Array.of(rowCount, 1)
      };
      const output = {
        labels: view('uint32', rowCount),
        regionCount: view('uint32', 1),
        cutEdges: view('uint32', rowCount - 1),
        cutGains: view('float32', rowCount - 1)
      };
      const nodes = new GPUSkaterRegions({
        columnCount: 2,
        maximumRegionCount: rowCount,
        weights: {
          offsets: view('uint32', input.offsets.length, input.offsets),
          neighbors: view('uint32', 2, input.neighbors),
          weights: view('float32', 2, input.weights)
        },
        values: view('float32', input.values.length, input.values),
        treeEdgeFlags: view('uint32', 2, input.treeEdgeFlags),
        componentLabels: view('uint32', rowCount, input.componentLabels),
        parameters: view('uint32', 2, input.parameters),
        output
      }).getCommandNodes(graph);
      executeCommandNodesOnCPU(nodes, memory);
      const expectedLabels = withIsland ? [0, 1, 2] : [0, 1];
      const expectedEdges = withIsland ? [0, 0xffffffff] : [0];
      expect(Array.from(memory.readUint32(output.labels))).toEqual(expectedLabels);
      expect(Array.from(memory.readUint32(output.regionCount))).toEqual([rowCount]);
      expect(Array.from(memory.readUint32(output.cutEdges))).toEqual(expectedEdges);
      const gains = memory.readFloat32(output.cutGains);
      expect(Number.isFinite(gains[0])).toBe(true);
      expect(Math.abs(gains[0] / 1.9999999360571385e38 - 1)).toBeLessThan(2e-6);
      if (withIsland) expect(gains[1]).toBe(0);
      const cpu = computeSkaterRegionsOnCPU(input, {columnCount: 2, maximumRegionCount: rowCount});
      expect(Array.from(cpu.labels)).toEqual(expectedLabels);
      expect(Array.from(cpu.regionCount)).toEqual([rowCount]);
      expect(Array.from(cpu.cutEdges)).toEqual(expectedEdges);
      expect(Array.from(cpu.cutGains)).toEqual(
        withIsland ? [1.9999999360571385e38, 0] : [1.9999999360571385e38]
      );
      const result = await new CPUSkaterRegionsOperation({
        rowCount,
        edgeCount: 2,
        columnCount: 2,
        maximumRegionCount: rowCount,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        values: 'values',
        treeEdgeFlags: 'treeEdgeFlags',
        componentLabels: 'componentLabels',
        parameters: 'parameters',
        output: {
          labels: 'labels',
          regionCount: 'regionCount',
          cutEdges: 'cutEdges',
          cutGains: 'cutGains'
        }
      }).execute({
        inputs: Object.fromEntries(
          Object.entries(input).map(([name, values]) => [
            name,
            {
              format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
              length: values.length,
              chunks: [{values, length: values.length}],
              pack: () => values
            }
          ])
        )
      });
      expect(Array.from(result['labels'] as Uint32Array)).toEqual(expectedLabels);
      expect(Array.from(result['regionCount'] as Uint32Array)).toEqual([rowCount]);
      expect(Array.from(result['cutEdges'] as Uint32Array)).toEqual(expectedEdges);
      expect(Array.from(result['cutGains'] as Float32Array)).toEqual(Array.from(cpu.cutGains));
    } finally {
      device.destroy();
    }
  }
});

it('review 2 R4 does not remove a nonfinite column as a harmless constant', () => {
  for (const invalid of [Infinity, NaN]) {
    const {device, graph, memory, view} = createScene();
    try {
      const output = {
        labels: view('uint32', 2),
        regionCount: view('uint32', 1),
        cutEdges: view('uint32', 1),
        cutGains: view('float32', 1)
      };
      const nodes = new GPUSkaterRegions({
        weights: {
          offsets: view('uint32', 3, Uint32Array.of(0, 1, 2)),
          neighbors: view('uint32', 2, Uint32Array.of(1, 0)),
          weights: view('float32', 2, Float32Array.of(1, 1))
        },
        values: view('float32', 4, Float32Array.of(invalid, -1, invalid, 1)),
        treeEdgeFlags: view('uint32', 2, Uint32Array.of(1, 0)),
        componentLabels: view('uint32', 2, Uint32Array.of(0, 0)),
        parameters: view('uint32', 2, Uint32Array.of(2, 1)),
        columnCount: 2,
        maximumRegionCount: 2,
        output
      }).getCommandNodes(graph);
      executeCommandNodesOnCPU(nodes, memory);
      // Without treeStatus, any nonfinite feature suppresses cuts in this initial tree.
      expect(Array.from(memory.readUint32(output.labels))).toEqual([0, 0]);
      expect(Array.from(memory.readUint32(output.regionCount))).toEqual([1]);
      expect(Array.from(memory.readUint32(output.cutEdges))).toEqual([0xffffffff]);
      expect(Array.from(memory.readFloat32(output.cutGains))).toEqual([0]);
    } finally {
      device.destroy();
    }
  }
});

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(
    format: Format,
    length: number,
    data?: Uint32Array | Float32Array
  ) => {
    const result = createTransientView(graph, `review-${serial++}`, format, length);
    if (data) memory.write(result, data);
    return result;
  };
  return {device, graph, memory, view};
}

it('R1 preserves raw Ward candidate costs after an exactly representable translation', async () => {
  for (const shift of [0, 1000000]) {
    const {device, graph, memory, view} = createScene();
    try {
      const input = {
        offsets: Uint32Array.of(0, 1, 3, 5, 6),
        neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2),
        values: Float32Array.from([0, 1 / 16, 3 / 16, 6 / 16], value => value + shift),
        parameters: Uint32Array.of(0)
      };
      const nodes = new GPUWardRegions({
        id: 'review-ward',
        weights: {
          offsets: view('uint32', 5, input.offsets),
          neighbors: view('uint32', 6, input.neighbors),
          weights: view('float32', 6, new Float32Array(6).fill(1))
        },
        values: view('float32', 4, input.values),
        parameters: view('uint32', 1, input.parameters),
        standardize: false,
        targetRegionCount: 2,
        maximumIterations: 2,
        output: {
          labels: view('uint32', 4),
          objective: view('float32', 1),
          status: view('uint32', 4)
        }
      }).getCommandNodes(graph);
      const node = (suffix: string) => nodes.find(candidate => candidate.id.endsWith(suffix))!;
      const binding = (suffix: string, name: string): GraphDataView =>
        getWGSLKernelNodeRecord(graph, node(suffix).id)!.bindings.find(
          candidate => candidate.name === name
        )!.view;
      executeCommandNodesOnCPU([node('-initialize-status'), node('-initialize-regions')], memory);
      // Supply the independently known first winner (0,1), cost (1/16)^2 / 2 = 1/512.
      memory.write(binding('-iteration-0-apply', 'first'), Uint32Array.of(0));
      memory.write(binding('-iteration-0-apply', 'second'), Uint32Array.of(1));
      memory.write(binding('-iteration-0-apply', 'cost'), Float32Array.of(1 / 512));
      executeCommandNodesOnCPU([node('-iteration-0-apply')], memory);
      // Evaluate both remaining adjacency costs using the actual emitted kernel, without
      // interpreting the cooperative argmin. Force recomputation rather than cache reuse.
      memory.write(binding('-contract-1-costs', 'counts'), Uint32Array.of(2, 0));
      memory.write(binding('-contract-1-costs', 'edges'), Uint32Array.of(0, 2, 2, 3));
      memory.write(binding('-contract-1-costs', 'previousEdges'), Uint32Array.of(0, 1));
      memory.write(binding('-contract-1-costs', 'previousCosts'), Float32Array.of(NaN, NaN));
      executeCommandNodesOnCPU([node('-contract-1-begin'), node('-contract-1-costs')], memory);
      const costs = memory.readFloat32(binding('-contract-1-costs', 'costs'), 2);
      // NumPy direct centered-SSD differences, independently derived in the adjacent oracle.
      [0.016276041666666668, 0.017578125].forEach((expected, index) => {
        expect(Number.isFinite(costs[index])).toBe(true);
        expect(Math.abs(costs[index] - expected)).toBeLessThan(1e-8);
      });
      const cpu = computeWardRegionsOnCPU(input, {
        standardize: false,
        targetRegionCount: 2,
        maximumIterations: 2
      });
      expect(Array.from(cpu.labels)).toEqual([0, 0, 0, 3]);
      expect(Array.from(cpu.status)).toEqual([2, 1, 0, 0]);
      expect(Number.isFinite(cpu.objective[0])).toBe(true);
      expect(Math.abs(cpu.objective[0] - 0.018229166666666668)).toBeLessThan(1e-8);
      const cpuOperation = new CPUWardRegionsOperation({
        rowCount: 4,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        values: 'values',
        parameters: 'parameters',
        standardize: false,
        targetRegionCount: 2,
        maximumIterations: 2,
        output: {labels: 'labels', objective: 'objective', status: 'status'}
      });
      const result = await cpuOperation.execute({
        inputs: Object.fromEntries(
          Object.entries({...input, weights: new Float32Array(6).fill(1)}).map(([name, data]) => [
            name,
            {
              format: data instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
              length: data.length,
              chunks: [{values: data, length: data.length}],
              pack: () => data
            }
          ])
        )
      });
      expect(Array.from(result['labels'] as Uint32Array)).toEqual([0, 0, 0, 3]);
      expect(Array.from(result['status'] as Uint32Array)).toEqual([2, 1, 0, 0]);
      expect(Number.isFinite((result['objective'] as Float32Array)[0])).toBe(true);
      expect(
        Math.abs((result['objective'] as Float32Array)[0] - 0.018229166666666668)
      ).toBeLessThan(1e-8);
    } finally {
      device.destroy();
    }
  }
});

for (const port of ['skater-count', 'redcap-count', 'redcap-converged', 'redcap-limit'] as const) {
  it(`R2 rejects an empty packed scalar before writing its adjacent canary: ${port}`, () => {
    const {device, graph, memory, view} = createScene();
    try {
      const weights = {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      };
      const backing = view('uint32', 2, Uint32Array.of(81, 82));
      const empty = graph.createDataView(backing.buffer, {
        format: 'uint32',
        length: 0,
        byteOffset: 4
      });
      const values = view('float32', 2);
      const labels = view('uint32', 2);
      const construct = () =>
        port === 'skater-count'
          ? new GPUSkaterRegions({
              weights,
              values,
              treeEdgeFlags: view('uint32', 2),
              componentLabels: view('uint32', 2),
              parameters: view('uint32', 2),
              maximumRegionCount: 2,
              output: {labels, regionCount: empty}
            })
          : new GPURedcapSingleLinkage({
              weights,
              values,
              parameters: view('uint32', 1),
              output: {
                labels,
                regionCount: port === 'redcap-count' ? empty : undefined,
                status: {
                  converged: port === 'redcap-converged' ? empty : undefined,
                  iterationLimitReached: port === 'redcap-limit' ? empty : undefined
                }
              }
            });
      expect(construct).toThrow(
        /output\.(regionCount|status\.(converged|iterationLimitReached)).*(length|rows|1)/
      );
      expect(Array.from(memory.readUint32(backing))).toEqual([81, 82]);
    } finally {
      device.destroy();
    }
  });
}

it('R3 keeps repeated above-mean values positive with the emitted pairwise lane arithmetic', () => {
  const {device, graph, memory, view} = createScene();
  try {
    const rowCount = 4096;
    const data = new Float32Array(rowCount).fill(0.1);
    data[0] = 0;
    const values = view('float32', rowCount, data);
    const scaled = view('float32', rowCount);
    createStandardizeColumnsNode(graph, {
      id: 'review-standardize',
      operation: 'ReviewStandardize',
      values,
      scaled,
      rowCount,
      columnCount: 1
    });
    const record = getWGSLKernelNodeRecord(graph, 'review-standardize')!;
    // One active lane executes the actual emitted lane summation over 4096 terms. The full
    // 256-lane cooperative kernel has its own 65536-row headless case. Barriers are unnecessary
    // only in this serial specialization; no claim is made about device scheduling here.
    const node = createWGSLKernelNode(graph, {
      id: 'review-standardize-serial',
      operation: 'ReviewStandardize',
      invocationCount: 1,
      workgroupSize: 2,
      bindings: record.bindings,
      body: record.body.replaceAll('storageBarrier();', ''),
      declarations: record
        .declarations!.replace('const LANES: u32 = 256u;', 'const LANES: u32 = 1u;')
        .replaceAll('workgroupBarrier();', '')
    });
    executeCommandNodesOnCPU([node], memory);
    const actual = memory.readFloat32(scaled);
    // Closed form for [0,c,...,c], c>0: z(0)=-sqrt(n-1), z(c)=1/sqrt(n-1).
    expect(Number.isFinite(actual[1])).toBe(true);
    expect(Math.abs(actual[1] - 0.015626907697949846)).toBeLessThan(3e-5);
    expect(actual[1]).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('R4 retains the representable SKATER gain when the unweighted square overflows', () => {
  const {device, graph, memory, view} = createScene();
  try {
    const input = {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 0),
      weights: Float32Array.of(1, 1),
      values: Float32Array.of(-1e19, 1e19),
      treeEdgeFlags: Uint32Array.of(1, 0),
      componentLabels: Uint32Array.of(0, 0),
      parameters: Uint32Array.of(2, 1)
    };
    const output = {
      labels: view('uint32', 2),
      regionCount: view('uint32', 1),
      cutEdges: view('uint32', 1),
      cutGains: view('float32', 1)
    };
    const nodes = new GPUSkaterRegions({
      weights: {
        offsets: view('uint32', 3, input.offsets),
        neighbors: view('uint32', 2, input.neighbors),
        weights: view('float32', 2, input.weights)
      },
      values: view('float32', 2, input.values),
      treeEdgeFlags: view('uint32', 2, input.treeEdgeFlags),
      componentLabels: view('uint32', 2, input.componentLabels),
      parameters: view('uint32', 2, input.parameters),
      maximumRegionCount: 2,
      output
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.labels))).toEqual([0, 1]);
    expect(Array.from(memory.readUint32(output.regionCount))).toEqual([2]);
    expect(Array.from(memory.readUint32(output.cutEdges))).toEqual([0]);
    const gain = memory.readFloat32(output.cutGains)[0];
    expect(Number.isFinite(gain)).toBe(true);
    expect(Math.abs(gain / 1.9999999360571385e38 - 1)).toBeLessThan(2e-6);
    const cpu = computeSkaterRegionsOnCPU(input, {maximumRegionCount: 2});
    expect(Array.from(cpu.labels)).toEqual([0, 1]);
    expect(Array.from(cpu.regionCount)).toEqual([2]);
    expect(Array.from(cpu.cutEdges)).toEqual([0]);
    expect(Array.from(cpu.cutGains)).toEqual([1.9999999360571385e38]);
  } finally {
    device.destroy();
  }
});

it('R5 preserves the zero-cost edge on the connected tiny-cost cycle', () => {
  const {device, graph, memory, view} = createScene();
  try {
    const labels = view('uint32', 4);
    const regionCount = view('uint32', 1);
    const converged = view('uint32', 1);
    const iterationCount = view('uint32', 1);
    const iterationLimitReached = view('uint32', 1);
    const invalidCount = view('uint32', 1);
    const nodes = new GPURedcapSingleLinkage({
      weights: {
        offsets: view('uint32', 5, Uint32Array.of(0, 2, 4, 6, 8)),
        neighbors: view('uint32', 8, Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2)),
        weights: view('float32', 8, new Float32Array(8).fill(1))
      },
      values: view('float32', 4, Float32Array.of(0, 0, 1e-18, 3e-18)),
      parameters: view('uint32', 1, Uint32Array.of(3)),
      standardize: false,
      componentIterations: 8,
      output: {
        labels,
        regionCount,
        status: {converged, iterationCount, iterationLimitReached, invalidCount}
      }
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    // The unique lowest cost is the exact zero on edge 0--1. Three regions retain only it.
    expect(Array.from(memory.readUint32(labels))).toEqual([0, 0, 2, 3]);
    expect(Array.from(memory.readUint32(regionCount))).toEqual([3]);
    expect(Array.from(memory.readUint32(converged))).toEqual([1]);
    expect(Array.from(memory.readUint32(iterationCount))).toEqual([8]);
    expect(Array.from(memory.readUint32(iterationLimitReached))).toEqual([0]);
    expect(Array.from(memory.readUint32(invalidCount))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

for (const Backend of [GPUWardRegionsOperation, CPUWardRegionsOperation]) {
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    it(`R6 ${Backend.name} rejects a supplied empty ${name} name`, () => {
      const props = {
        id: 'review-ward',
        rowCount: 2,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        values: 'values',
        parameters: 'parameters',
        standardize: false,
        targetRegionCount: 1,
        maximumIterations: 1,
        output: {labels: 'labels', objective: 'objective', status: 'status'}
      };
      expect(
        () => new Backend({...props, weights: {...props.weights, status: {[name]: ''}}})
      ).toThrow(new RegExp(`^review-ward.*${name}.*non-empty`));
      expect(() => new Backend(props)).not.toThrow();
      expect(
        () => new Backend({...props, weights: {...props.weights, status: {[name]: undefined}}})
      ).not.toThrow();
      expect(
        () => new Backend({...props, weights: {...props.weights, status: {[name]: 'loss'}}})
      ).not.toThrow();
    });
  }
}

it('R7 rejects the six-lane tree that loses a unique minimum in lane 2', () => {
  // The old six-lane strides 3,1 retain min(0,3,1,4) and discard min(2,5).
  // [9,8,-1,7,6,5] therefore incorrectly returned 6 instead of -1. Reject this unsupported
  // topology before generating a shader; callers must pad to a power of two.
  expect(() => getWorkgroupArgminWGSL(6)).toThrow(/workgroupSize.*power of two/);
});

it('R7 emits and executes a valid one-lane argmin instead of the integer literal 0.5u', () => {
  const {device, graph, memory, view} = createScene();
  try {
    const source = getWorkgroupArgminWGSL(1);
    expect(source).not.toMatch(/\d+\.\d+u/);
    const result = view('uint32', 1);
    // Execute the one logical lane without cooperative barriers (the executor is serial).
    const node = createWGSLKernelNode(graph, {
      id: 'review-argmin-one',
      operation: 'ArgminReview',
      invocationCount: 1,
      workgroupSize: 2,
      bindings: [{name: 'result', view: result, type: 'u32', access: 'read_write'}],
      declarations: source
        .replaceAll('workgroupBarrier();', '')
        .replace('workgroupUniformLoad(&argminValues[0])', 'argminValues[0]'),
      body: 'result[resultOffset] = reduceArgmin(0u, 7.0, 3u, 42u);'
    });
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(result))).toEqual([42]);
  } finally {
    device.destroy();
  }
});
