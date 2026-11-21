// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {getStudyWindowSummaryNodes} from '../../../src/gpu-spatial-analysis/study-window/study-window-summary';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();

for (const mode of ['positive', 'cancellation', 'empty'] as const) {
  const empty = mode === 'empty';
  it(`CORE-118 reduces ring summaries in bounded stages (${mode})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const ringCount = 514;
    const geometry = {
      positions: createTransientView(graph, 'positions', 'float32x2', 3 * ringCount),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', ringCount + 1),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', ringCount + 1)
    };
    const createView = vi.spyOn(graph, 'createDataView');
    const preparation = new GPUStudyWindowPreparation({...geometry, samplingGrid: [2, 2]});
    try {
      const nodes = preparation.getCommandNodes(graph);
      const summaryNodes = nodes.filter(node =>
        /-summary-(partials|reduce-)|-summarize$/.test(node.id)
      );
      // A ring table of this size must be split, regardless of the four-cell sampling grid.
      expect(summaryNodes.length).toBeGreaterThan(3);
      for (const kernel of getGPUCommandNodeKernelSources(summaryNodes)) {
        expect(kernel.source).not.toMatch(/for\s*\(var ring/);
        expect(() => new WgslReflect(kernel.source)).not.toThrow();
      }
      const rings = createView.mock.results.find(
        result => result.type === 'return' && result.value.buffer.id === 'study-window-rings'
      )!.value;
      const records = new Float32Array(10 * ringCount);
      if (!empty) {
        for (let ring = 0; ring < 513; ring++) {
          records.set(
            [
              ring === 0 ? 16777216 : 1,
              ring === 0 ? 16777216 : 1,
              3,
              ring === 512 ? 1 : 0,
              -2,
              -3,
              8,
              9,
              1,
              1
            ],
            10 * ring
          );
        }
        // Rejected rings must not affect totals, flags or bounds.
        records.set([999, 999, 999, 1, -999, -999, 999, 999, 1, 0], 5130);
      }
      if (mode === 'cancellation') records[10] = -16777216;
      const memory = createCPUGraphMemory();
      memory.write(rings, records);
      executeGeometryNodesOnCPU(summaryNodes, memory);
      const result = Array.from(memory.readFloat32(preparation.getViews(graph).summary));
      if (empty) {
        expect(result.slice(0, 4).every(Number.isNaN)).toBe(true);
        expect(result.slice(4)).toEqual([0, 0, 0, 2]);
      } else {
        // Closed form: 2^24 + 512 unit rings; 513 accepted rings times 3 edges.
        expect(result.every(Number.isFinite)).toBe(true);
        expect(result).toEqual([
          -2,
          -3,
          8,
          9,
          mode === 'cancellation' ? 511 : 16777728,
          16777728,
          1539,
          1
        ]);
      }
      expect(nodes.indexOf(summaryNodes.at(-1)!)).toBeLessThan(
        nodes.findIndex(node => node.id.endsWith('-sample-parameters'))
      );
      const commit = nodes.at(-1)!;
      expect(commit.id).toContain('commit');
      for (const node of summaryNodes) expect(commit.dependsOn).toContain(node.id);
    } finally {
      createView.mockRestore();
      preparation.destroy();
      device.destroy();
    }
  });
}

for (const [ringCount, expected] of [
  [1, [0, 0, 1, 1, 1, 4, 3, 0]],
  [3, [0, 0, 3, 3, 6, 12, 9, 0]]
] as const) {
  it(`CORE-118 initializes an odd ring tail without reading past ${ringCount} rings`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const rings = createTransientView(graph, 'rings', 'float32', 10 * ringCount);
    const summary = createTransientView(graph, 'summary', 'float32', 8);
    const records = new Float32Array(10 * ringCount);
    for (let ring = 0; ring < ringCount; ring++)
      records.set([ring + 1, 4, 3, 0, 0, 0, ring + 1, ring + 1, 1, 1], 10 * ring);
    const memory = createCPUGraphMemory();
    memory.write(rings, records);
    try {
      executeGeometryNodesOnCPU(
        getStudyWindowSummaryNodes(graph, {
          id: 'summary',
          operation: 'GPUStudyWindowPreparation',
          rings,
          ringStride: 10,
          summary
        }),
        memory
      );
      // Literal sums of one or three accepted records: areas [1] or [1,2,3], perimeter 4 each.
      expect(Array.from(memory.readFloat32(summary))).toEqual(expected);
    } finally {
      device.destroy();
    }
  });
}

it('B3 retains a right partial correction after a rejected prefix ring', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rings = createTransientView(graph, 'correction-rings', 'float32', 40);
  const summary = createTransientView(graph, 'correction-summary', 'float32', 8);
  const createView = vi.spyOn(graph, 'createDataView');
  const nodes = getStudyWindowSummaryNodes(graph, {
    id: 'correction',
    operation: 'GPUStudyWindowPreparation',
    rings,
    ringStride: 10,
    summary
  });
  const partials = createView.mock.results.find(
    result => result.type === 'return' && result.value.buffer.id === 'correction-summary-partials'
  )!.value;
  const records = new Float32Array(40);
  for (const [ring, area] of [0, 1, 16777216, 1].entries())
    records.set([area, 4, 3, 0, 0, 0, 1, 1, 1, ring === 0 ? 0 : 1], 10 * ring);
  const run = (dropRightCorrection: boolean) => {
    const memory = createCPUGraphMemory();
    memory.write(rings, records);
    executeGeometryNodesOnCPU(nodes.slice(0, 1), memory);
    const values = memory.readFloat32(partials);
    expect(values[12]).toBe(1); // Right partial = (2^24, 1).
    if (dropRightCorrection) {
      values[12] = 0;
      memory.write(partials, values);
    }
    executeGeometryNodesOnCPU(nodes.slice(1), memory);
    return memory.readFloat32(summary)[4];
  };
  try {
    // Closed form: rejected zero, then 1 + 2^24 + 1 = 16777218, exactly representable.
    const actual = run(false);
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBe(16777218);
    expect(run(true)).toBe(16777216); // Mutation control proves the new ordering needs the right correction.
  } finally {
    createView.mockRestore();
    device.destroy();
  }
});
