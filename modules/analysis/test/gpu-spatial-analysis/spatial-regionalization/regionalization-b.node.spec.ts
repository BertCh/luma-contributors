// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {computeWardRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {createStandardizeColumnsNode} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-inputs';
import {getWardMutualSelectionNodes} from '../../../src/gpu-spatial-analysis/spatial-regionalization/ward-mutual-selection';
import {
  createWGSLKernelNode,
  createWGSLDispatchGate,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  AUDIT_STANDARDIZED,
  AUDIT_WARD_OBJECTIVES,
  AUDIT_WARD_PREFIXES,
  createRegionalizationAuditInput
} from './regionalization-b-fixtures';

it('standardizes large finite values in the emitted WGSL arithmetic against NumPy', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const values = createTransientView(graph, 'scale-values', 'float32', 9);
    const scaled = createTransientView(graph, 'scale-output', 'float32', 9);
    createStandardizeColumnsNode(graph, {
      id: 'standardize',
      operation: 'ScaleAudit',
      values,
      scaled,
      rowCount: 9,
      columnCount: 1
    });
    const record = getWGSLKernelNodeRecord(graph, 'standardize')!;
    // The node executor cannot synchronize workgroups. Specialize the actual emitted algorithm
    // to one lane: its reductions become identities and barriers are unnecessary. This checks
    // float32 arithmetic, not the GPU reduction tree or synchronization (the headless case does).
    const declarations = record
      .declarations!.replace('const LANES: u32 = 256u;', 'const LANES: u32 = 1u;')
      .replaceAll('workgroupBarrier();', '');
    expect(declarations).toContain('const LANES: u32 = 1u;');
    const node = createWGSLKernelNode(graph, {
      id: 'serial-standardize',
      operation: 'ScaleAudit',
      invocationCount: 1,
      workgroupSize: 2,
      declarations,
      body: record.body.replaceAll('storageBarrier();', ''),
      bindings: record.bindings
    });
    for (const [scale, shift] of [
      [1, 0],
      [2 ** -80, 0],
      [2 ** 64, 64],
      [2 ** 120, 64]
    ]) {
      const memory = createCPUGraphMemory();
      memory.write(values, createRegionalizationAuditInput(scale, shift).values);
      executeCommandNodesOnCPU([node], memory);
      const actual = memory.readFloat32(scaled);
      AUDIT_STANDARDIZED.forEach((expected, index) => {
        expect(Number.isFinite(actual[index])).toBe(true);
        expect(Math.abs(actual[index] - expected)).toBeLessThan(2e-5);
      });
    }
  } finally {
    device.destroy();
  }
});

it('selects the exact Ward mutual-pair budget by cost, key and root through all radix digits', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const budget of [0, 1, 2, 3, 4]) {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      let serial = 0;
      const input = (values: number[]) => {
        const view = createTransientView(graph, `selection-${serial++}`, 'uint32', values.length);
        memory.write(view, Uint32Array.from(values));
        return view;
      };
      const gate = createWGSLDispatchGate(graph, {
        id: 'gate',
        operation: 'SelectionAudit',
        shapes: [
          [8, 2],
          [2, 2]
        ]
      });
      memory.write(gate.view, Uint32Array.of(4, 1, 1, 1, 1, 1));
      const mergeFlag = input(new Array(8).fill(99));
      const summary = input(new Array(7).fill(99));
      const histogram = input(new Array(256).fill(99));
      const nodes = getWardMutualSelectionNodes(graph, {
        id: 'select',
        rowCount: 8,
        targetRegionCount: 8 - budget,
        rowWorkgroupSize: 2,
        gate,
        // Four reciprocal pairs. Pair 6 is more expensive; 2 and 4 tie in cost and key.
        // Independent lexicographic order is roots [2,4,0,6], not their CSR or invocation order.
        regionCost: input([
          0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x40000000,
          0x40000000
        ]),
        regionKey: input([17, 17, 5, 5, 5, 5, 1, 1]),
        regionOther: input([1, 0, 3, 2, 5, 4, 7, 6]),
        mergeFlag,
        summary,
        histogram,
        state: input([8, 0, 0, 0, 0]),
        status: input([0, 0, 0, 0])
      });
      executeCommandNodesOnCPU(nodes, memory);
      const expected = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0]
      ][budget];
      expect(Array.from(memory.readUint32(mergeFlag))).toEqual(expected);
      expect(Array.from(memory.readUint32(summary)).slice(0, 2)).toEqual([4, budget]);
      expect(memory.readUint32(summary)[6]).toBe(budget > 0 && budget < 4 ? 12 : 0);
      expect(Array.from(memory.readUint32(histogram))).toEqual(new Array(256).fill(0));
    }
  } finally {
    device.destroy();
  }
});

it('matches every raw Ward prefix and infeasible status against NumPy centered SSD', () => {
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    let previousObjective = -1;
    for (let maximumIterations = 0; maximumIterations <= 7; maximumIterations++) {
      const mergeCount = Math.min(
        6,
        mergeSchedule === 'sequential'
          ? maximumIterations
          : maximumIterations === 0
            ? 0
            : maximumIterations === 1
              ? 4
              : 6
      );
      const exhausted = maximumIterations > (mergeSchedule === 'sequential' ? 6 : 2);
      const result = computeWardRegionsOnCPU(createRegionalizationAuditInput(), {
        targetRegionCount: 1,
        maximumIterations,
        mergeSchedule,
        standardize: false,
        statusLength: 5
      });
      expect(Array.from(result.labels)).toEqual(AUDIT_WARD_PREFIXES[mergeCount]);
      expect(Array.from(result.objective)).toEqual([AUDIT_WARD_OBJECTIVES[mergeCount]]);
      expect(Array.from(result.status)).toEqual([
        mergeCount,
        0,
        Number(!exhausted),
        0,
        mergeSchedule === 'sequential' ? 0 : Math.min(maximumIterations, 3)
      ]);
      expect(result.objective[0]).toBeGreaterThanOrEqual(previousObjective);
      previousObjective = result.objective[0];
    }
  }
});

it('keeps the population-standardized Ward partition under large finite rescalings', () => {
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    for (const [scale, shift] of [
      [1, 0],
      [2 ** -80, 0],
      [2 ** 64, 64],
      [2 ** 120, 64]
    ]) {
      const result = computeWardRegionsOnCPU(createRegionalizationAuditInput(scale, shift), {
        targetRegionCount: 5,
        maximumIterations: 4,
        mergeSchedule,
        statusLength: 5
      });
      expect(Array.from(result.labels)).toEqual(AUDIT_WARD_PREFIXES[4]);
      expect(Array.from(result.status)).toEqual([
        4,
        1,
        0,
        0,
        mergeSchedule === 'sequential' ? 0 : 1
      ]);
      expect(Number.isFinite(result.objective[0])).toBe(true);
      expect(Math.abs(result.objective[0] - 0.0346855462616689)).toBeLessThan(2e-9);
    }
  }
});
