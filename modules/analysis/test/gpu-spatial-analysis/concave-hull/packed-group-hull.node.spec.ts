// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createPackedGroupHullNodes} from '../../../src/gpu-spatial-analysis/concave-hull/packed-group-hull';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createConcaveHullValidationProps} from './concave-hull-validation-cases';

useCPUExecutorTestHooks();

for (const capacity of [0, 4]) {
  it(`packed group publication compacts retained rows and preserves empty group ranges (capacity ${capacity})`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const scalar = (name: string, length = 1) =>
        createTransientView(graph, name, 'uint32', length);
      const output = {
        ...createConcaveHullValidationProps(graph, 0, 3, capacity).output,
        groups: {
          triangleOffsets: scalar('output-triangle-offsets', 3),
          triangleCounts: scalar('output-triangle-counts', 3),
          edgeOffsets: scalar('output-edge-offsets', 3),
          edgeCounts: scalar('output-edge-counts', 3)
        },
        status: {
          converged: scalar('output-converged', 3),
          count: scalar('output-count'),
          requiredCount: scalar('output-required'),
          overflow: scalar('output-overflow')
        }
      };
      const props = {
        id: 'packed',
        groupCount: 3,
        triangles: createTransientView(graph, 'input-triangles', 'uint32x3', capacity),
        triangleValidity: scalar('input-triangle-validity', capacity),
        edges: createTransientView(graph, 'input-edges', 'uint32x2', 3 * capacity),
        edgeValidity: scalar('input-edge-validity', 3 * capacity),
        groupOffsets: scalar('group-offsets', 3),
        groupCounts: scalar('group-counts', 3),
        groupConverged: scalar('group-converged', 3),
        output
      };
      const memory = createCPUGraphMemory();
      memory.write(props.groupConverged, Uint32Array.of(1, 1, 1));
      // Seed caller buffers to verify publication clears padding and resets counts.
      for (const view of [
        output.triangles,
        output.triangleValidity,
        output.boundaryEdges,
        output.edgeValidity,
        output.status.count,
        output.status.requiredCount,
        output.status.overflow
      ]) {
        memory.write(view, new Uint32Array((view.length * view.rowByteLength) / 4).fill(99));
      }
      if (capacity) {
        memory.write(props.triangles, Uint32Array.of(0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6));
        memory.write(props.triangleValidity, Uint32Array.of(0, 1, 1, 0));
        memory.write(
          props.edges,
          Uint32Array.from({length: 24}, (_, index) => 10 + index)
        );
        memory.write(props.edgeValidity, Uint32Array.of(0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0));
        memory.write(props.groupOffsets, Uint32Array.of(0, 2, 2));
        memory.write(props.groupCounts, Uint32Array.of(2, 0, 2));
      }
      const nodes = createPackedGroupHullNodes(graph, props);
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(output.status.converged))).toEqual([1, 1, 1]);
      expect(Array.from(memory.readUint32(output.status.count))).toEqual([capacity ? 3 : 0]);
      expect(Array.from(memory.readUint32(output.status.requiredCount))).toEqual([
        capacity ? 3 : 0
      ]);
      expect(Array.from(memory.readUint32(output.status.overflow))).toEqual([0]);
      expect(Array.from(memory.readUint32(output.groups.triangleOffsets))).toEqual(
        capacity ? [0, 1, 1] : [0, 0, 0]
      );
      expect(Array.from(memory.readUint32(output.groups.triangleCounts))).toEqual(
        capacity ? [1, 0, 1] : [0, 0, 0]
      );
      expect(Array.from(memory.readUint32(output.groups.edgeOffsets))).toEqual(
        capacity ? [0, 2, 2] : [0, 0, 0]
      );
      expect(Array.from(memory.readUint32(output.groups.edgeCounts))).toEqual(
        capacity ? [2, 0, 1] : [0, 0, 0]
      );
      if (capacity) {
        expect(Array.from(memory.readUint32(output.triangles))).toEqual([
          1, 3, 2, 4, 5, 6, 0, 0, 0, 0, 0, 0
        ]);
        expect(Array.from(memory.readUint32(output.triangleValidity))).toEqual([1, 1, 0, 0]);
        expect(Array.from(memory.readUint32(output.boundaryEdges))).toEqual([
          16,
          17,
          20,
          21,
          24,
          25,
          ...new Array(18).fill(0)
        ]);
        expect(Array.from(memory.readUint32(output.edgeValidity))).toEqual([
          1,
          1,
          1,
          ...new Array(9).fill(0)
        ]);
        // Reuse the same nodes and caller outputs after all triangles are removed. Geometry,
        // group ranges and totals must clear rather than preserving the previous compact prefix.
        memory.write(props.triangleValidity, new Uint32Array(capacity));
        memory.write(props.edgeValidity, new Uint32Array(3 * capacity));
        executeCommandNodesOnCPU(nodes, memory);
        for (const view of [
          output.triangles,
          output.triangleValidity,
          output.boundaryEdges,
          output.edgeValidity,
          output.status.count,
          output.status.requiredCount,
          output.status.overflow,
          ...Object.values(output.groups)
        ]) {
          expect(Array.from(memory.readUint32(view))).toEqual(
            new Array((view.length * view.rowByteLength) / 4).fill(0)
          );
        }
      }
    } finally {
      device.destroy();
    }
  });
}
