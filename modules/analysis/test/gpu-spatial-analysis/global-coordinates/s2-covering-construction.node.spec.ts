// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('captures S2 covering construction fields and prepares ring orientation in both routes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['serial', 'level-synchronous'] as const) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const hemisphereCenter: [number, number, number] = [1, 0, 0];
      const output = {
        cells: createTransientView(graph, 'cells', 'uint32x2', 8),
        counts: createTransientView(graph, 'counts', 'uint32x2', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      };
      const props = {
        positions: createTransientView(graph, 'positions', 'float32x3', 8),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 3),
        maximumLevel: 2,
        maximumVisitedCells: 64,
        capacity: 8,
        refinement,
        hemisphereCenter,
        output
      };
      const contributor = new GPUS2PolygonCovering(props);
      props.maximumLevel = 15;
      props.capacity = 99;
      hemisphereCenter[0] = -1;
      output.status = createTransientView(graph, 'replacement', 'uint32', 1);
      expect(contributor.props.maximumLevel).toBe(2);
      expect(contributor.props.capacity).toBe(8);
      expect(contributor.props.hemisphereCenter).toEqual([1, 0, 0]);
      expect(contributor.props.output.status).not.toBe(output.status);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      for (const [suffix, count] of [
        ['prepare-vertices', 8],
        ['prepare-centers', 2],
        ['prepare-area', 8],
        ['orient-rings', 2],
        ['copy-oriented', 8],
        ['initialize', 8],
        ...[1, 2, 4].flatMap(distance => [
          [`reduce-center-${distance}`, 8],
          [`reduce-area-${distance}`, 8]
        ])
      ] as const) {
        const node = nodes.find(candidate => candidate.id === `s2-polygon-covering-${suffix}`);
        expect(node, String(suffix)).toBeDefined();
        expect(node?.workload?.maximumInvocationCount, String(suffix)).toBe(count);
      }
      for (const stage of ['reduce-center-4', 'reduce-area-4'])
        expect(getWGSLKernelNodeRecord(graph, `s2-polygon-covering-${stage}`)?.body).toContain(
          'owners[ownersOffset+index]==owners[ownersOffset+index-4u]'
        );
      expect(getWGSLKernelNodeRecord(graph, 's2-polygon-covering-prepare-area')?.body).toContain(
        'area=dot(pole,cross(firstDelta,secondDelta))'
      );
      const kernels = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of kernels)
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('CORE-118 initializes the six root faces in parallel', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['serial', 'level-synchronous'] as const) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const contributor = new GPUS2PolygonCovering({
        positions: createTransientView(graph, 'positions', 'float32x3', 3),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
        maximumLevel: 1,
        maximumVisitedCells: 6,
        capacity: 12,
        refinement,
        output: {
          cells: createTransientView(graph, 'cells', 'uint32x2', 12),
          counts: createTransientView(graph, 'counts', 'uint32x2', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      const start = nodes.find(node => node.id === 's2-polygon-covering-initialize')!;
      expect(start.workload?.maximumInvocationCount).toBe(12);
      const initialization = getWGSLKernelNodeRecord(graph, start.id)!;
      const runInitialization = new Function(
        'index',
        'cells',
        'control',
        'totals',
        'roots',
        'roundParameters',
        'const cellsOffset=0,controlOffset=0,totalsOffset=0,rootsOffset=0,roundParametersOffset=0;' +
          initialization.body.replace(/\b(\d+)u\b/g, '$1')
      );
      const cells = new Uint32Array(24).fill(99);
      const control = new Uint32Array(8).fill(99);
      const totals = new Uint32Array(3).fill(99);
      const roots = new Uint32Array(12).fill(99);
      const roundParameters = new Uint32Array(128).fill(99);
      const initialize = (
        index: number,
        targetCells: Uint32Array,
        targetControl: Uint32Array,
        targetTotals: Uint32Array,
        targetRoots: Uint32Array
      ) =>
        runInitialization(
          index,
          targetCells,
          targetControl,
          targetTotals,
          targetRoots,
          roundParameters
        );
      for (let index = 0; index < 12; index++) initialize(index, cells, control, totals, roots);
      // Level-zero S2 IDs are the odd face markers 1,3,5,7,9,11 shifted by 60 bits.
      expect(Array.from(roots)).toEqual([
        0, 0x10000000, 0, 0x30000000, 0, 0x50000000, 0, 0x70000000, 0, 0x90000000, 0, 0xb0000000
      ]);
      expect(Array.from(cells)).toEqual(Array(24).fill(0));
      expect(Array.from(control)).toEqual(Array(8).fill(0));
      expect(Array.from(totals)).toEqual([6, 0, 0]);
      // Each 256-byte-aligned slot stores [currentLevel, maximumLevel].
      expect(Array.from(roundParameters)).toEqual([
        0,
        1,
        ...Array(62).fill(99),
        1,
        1,
        ...Array(62).fill(99)
      ]);
      const validateRootWrites = (implementation: typeof initialize) => {
        for (let index = 0; index < 12; index++) {
          const candidateRoots = new Uint32Array(12).fill(99);
          implementation(index, cells, control, totals, candidateRoots);
          const expected = Array<number>(12).fill(99);
          if (index < 6) {
            expected[index * 2] = 0;
            expected[index * 2 + 1] = (2 * index + 1) * 2 ** 28;
          }
          expect(Array.from(candidateRoots), `invocation ${index}`).toEqual(expected);
        }
      };
      validateRootWrites(initialize);
      // Mutation control: a serial initializer writing every root from invocation zero
      // must fail the parallelism oracle even though its final aggregate bytes are correct.
      const serialInitializer = (
        index: number,
        targetCells: Uint32Array,
        targetControl: Uint32Array,
        targetTotals: Uint32Array,
        targetRoots: Uint32Array
      ) => {
        initialize(index, targetCells, targetControl, targetTotals, targetRoots);
        if (index === 0) targetRoots.set(roots);
      };
      expect(() => validateRootWrites(serialInitializer)).toThrow();
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
