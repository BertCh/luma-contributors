// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {getWorkgroupArgminWGSL} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-wgsl';
import {getWardMutualSelectionNodes} from '../../../src/gpu-spatial-analysis/spatial-regionalization/ward-mutual-selection';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  AUDIT_FOREST_COSTS,
  AUDIT_FOREST_SLOTS,
  AUDIT_NEIGHBORS,
  AUDIT_OFFSETS,
  AUDIT_STANDARDIZED,
  AUDIT_WARD_OBJECTIVES,
  AUDIT_WARD_PREFIXES,
  createRegionalizationAuditInput
} from './regionalization-b-fixtures';

// All expectations are literals from regionalization-b.oracle.py. Population z-scores and edge
// costs allow 2e-5 absolute error for float32 reduction/division; IDs and statuses are exact.
function expectFloats(actual: number[], expected: number[]) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-5);
  });
}

it('reduces the independent lexicographic argmin through all tie fields', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const scores = rig.input(Float32Array.of(-2, -2, 3, 0, -2, 10, -2, 0), 'float32', 8);
    const keys = rig.input(Uint32Array.of(9, 4, 0, 0, 4, 0, 4, 0), 'uint32', 8);
    const values = rig.input(Uint32Array.of(0, 6, 4, 2, 5, 3, 1, 0), 'uint32', 8);
    const output = rig.output('uint32', 1);
    rig.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'audit-argmin',
          operation: 'RegionalizationArgminAudit',
          invocationCount: 8,
          workgroupSize: 8,
          guardIndex: false,
          declarations: getWorkgroupArgminWGSL(8),
          bindings: [
            {name: 'scores', view: scores, type: 'f32', access: 'read'},
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'values', view: values, type: 'u32', access: 'read'},
            {name: 'result', view: output.view, type: 'u32', access: 'read_write'}
          ],
          body: `let winner = reduceArgmin(localInvocationIndex, scores[scoresOffset + index], keys[keysOffset + index], values[valuesOffset + index]);
        if (localInvocationIndex == 0u) { result[resultOffset] = winner; }`
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // By inspection: minimum score -2; minimum key among those is 4; their values are 6,5,1.
      // Ignoring keys would incorrectly choose value 0 from the score=-2,key=9 lane.
      expect(await readUint32(output.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('selects the independent Ward radix threshold on a small GPU fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const regionCost = rig.input(
      Uint32Array.of(
        0x3f800000,
        0x3f800000,
        0x3f800000,
        0x3f800000,
        0x3f800000,
        0x3f800000,
        0x40000000,
        0x40000000
      ),
      'uint32',
      8
    );
    const regionKey = rig.input(Uint32Array.of(17, 17, 5, 5, 5, 5, 1, 1), 'uint32', 8);
    const regionOther = rig.input(Uint32Array.of(1, 0, 3, 2, 5, 4, 7, 6), 'uint32', 8);
    const state = rig.input(Uint32Array.of(8, 0, 0, 0, 0), 'uint32', 5);
    const status = rig.input(Uint32Array.of(0, 0, 0, 0), 'uint32', 4);
    const mergeFlag = rig.output('uint32', 8);
    rig.graph.add({
      getCommandNodes: graph => {
        const gate = createWGSLDispatchGate(graph, {
          id: 'radix-gate',
          operation: 'RadixAudit',
          shapes: [
            [8, 2],
            [2, 2]
          ]
        });
        return [
          createWGSLKernelNode(graph, {
            id: 'radix-arm',
            operation: 'RadixAudit',
            invocationCount: 1,
            bindings: [gate.binding],
            body: gate.armWGSL
          }),
          ...getWardMutualSelectionNodes(graph, {
            id: 'audit-radix',
            rowCount: 8,
            targetRegionCount: 6,
            rowWorkgroupSize: 2,
            gate,
            regionCost,
            regionKey,
            regionOther,
            mergeFlag: mergeFlag.view,
            state,
            status,
            summary: createTransientView(graph, 'radix-summary', 'uint32', 7),
            histogram: createTransientView(graph, 'radix-histogram', 'uint32', 256)
          })
        ];
      }
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // By inspection of (cost,key,root), reciprocal roots order [2,4,0,6]; budget is two.
      expect(await readUint32(mergeFlag.buffer, 8)).toEqual([0, 0, 1, 0, 1, 0, 0, 0]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

for (const [scale, shift] of [
  [1, 0],
  [2 ** -80, 0],
  [2 ** 64, 64],
  [2 ** 120, 64]
]) {
  for (const capacity of [2, 6]) {
    it(`audits every forest port with NumPy: scale=${scale}, compact capacity=${capacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      try {
        const flags = rig.output('uint32', 23);
        const labels = rig.output('uint32', 9);
        const standardized = rig.output('float32', 9);
        const ids = rig.output('uint32', capacity);
        const count = rig.output('uint32', 1);
        const requiredCount = rig.output('uint32', 1);
        const overflow = rig.output('uint32', 1);
        const candidateOverflow = rig.output('uint32', 1);
        const status = rig.output('uint32', 2);
        const endpoints = rig.output('uint32', 2 * capacity);
        const costs = rig.output('float32', capacity);
        rig.graph.add(
          new GPUSpatialWeightsMinimumSpanningTree({
            weights: rig.uploadWeights(
              {
                offsets: AUDIT_OFFSETS,
                neighbors: AUDIT_NEIGHBORS,
                weights: new Array(20).fill(1),
                distances: []
              },
              3
            ),
            values: rig.input(createRegionalizationAuditInput(scale, shift).values, 'float32', 9),
            output: {
              treeEdgeFlags: flags.view,
              componentLabels: labels.view,
              standardizedValues: standardized.view,
              status: status.view,
              edges: {
                ids: ids.view,
                status: {
                  count: count.view,
                  requiredCount: requiredCount.view,
                  overflow: overflow.view,
                  candidateOverflow: candidateOverflow.view
                }
              },
              edgeEndpoints: endpoints.view,
              edgeDissimilarities: costs.view
            }
          })
        );
        const compiled = rig.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(flags.buffer, 23)).toEqual([
            1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0
          ]);
          expect(await readUint32(labels.buffer, 9)).toEqual([0, 0, 0, 0, 4, 4, 4, 4, 8]);
          expectFloats(await readFloat32(standardized.buffer, 9), AUDIT_STANDARDIZED);
          expect(await readUint32(ids.buffer, capacity)).toEqual(
            AUDIT_FOREST_SLOTS.slice(0, capacity)
          );
          expect(await readUint32(count.buffer, 1)).toEqual([capacity]);
          expect(await readUint32(requiredCount.buffer, 1)).toEqual([6]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([Number(capacity < 6)]);
          expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
          expect(await readUint32(status.buffer, 2)).toEqual([0, 0]);
          expect(await readUint32(endpoints.buffer, 2 * capacity)).toEqual(
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7].slice(0, 2 * capacity)
          );
          expectFloats(
            await readFloat32(costs.buffer, capacity),
            AUDIT_FOREST_COSTS.slice(0, capacity)
          );
        } finally {
          compiled.destroy();
        }
      } finally {
        rig.destroy();
      }
    });
  }
}

for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
  for (const maximumIterations of [0, 1, 2, 3, 4, 5, 6, 7]) {
    it(`audits the Ward budget/infeasible prefix: ${mergeSchedule}, ${maximumIterations}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = createRegionalizationAuditInput();
      const result = await runOperationGraph(device, {
        inputs: Object.fromEntries(
          Object.entries({...input, weights: new Float32Array(20).fill(1)}).map(
            ([name, values]) => [
              name,
              {
                format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
                values
              }
            ]
          )
        ),
        operations: [
          new GPUWardRegionsOperation({
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            rowCount: 9,
            values: 'values',
            parameters: 'parameters',
            targetRegionCount: 1,
            maximumIterations,
            standardize: false,
            mergeSchedule,
            statusLength: 5,
            output: {labels: 'labels', objective: 'objective', status: 'status'}
          })
        ],
        outputs: ['labels', 'objective', 'status']
      });
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
      expect(Array.from(result['labels'])).toEqual(AUDIT_WARD_PREFIXES[mergeCount]);
      expectFloats(Array.from(result['objective']), [AUDIT_WARD_OBJECTIVES[mergeCount]]);
      expect(Array.from(result['status']).slice(0, 4)).toEqual([
        mergeCount,
        0,
        Number(!exhausted),
        0
      ]);
      if (mergeSchedule === 'mutual-nearest')
        expect(result['status'][4]).toBe(Math.min(maximumIterations, 3));
    });
  }
  for (const scale of [2 ** -80, 2 ** 64]) {
    it(`retains the NumPy Ward objective after finite rescaling ${scale}: ${mergeSchedule}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = createRegionalizationAuditInput(scale, 64);
      const result = await runOperationGraph(device, {
        inputs: Object.fromEntries(
          Object.entries({...input, weights: new Float32Array(20).fill(1)}).map(
            ([name, values]) => [
              name,
              {
                format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
                values
              }
            ]
          )
        ),
        operations: [
          new GPUWardRegionsOperation({
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            rowCount: 9,
            values: 'values',
            parameters: 'parameters',
            targetRegionCount: 5,
            maximumIterations: 4,
            mergeSchedule,
            statusLength: 5,
            output: {labels: 'labels', objective: 'objective', status: 'status'}
          })
        ],
        outputs: ['labels', 'objective', 'status']
      });
      expect(Array.from(result['labels'])).toEqual(AUDIT_WARD_PREFIXES[4]);
      expectFloats(Array.from(result['objective']), [0.0346855462616689]);
      expect(Array.from(result['status']).slice(0, 4)).toEqual([4, 1, 0, 0]);
      if (mergeSchedule === 'mutual-nearest') expect(result['status'][4]).toBe(1);
    });
  }
}
