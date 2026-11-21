// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getFacilityCompactionRoundInputs} from './facility-compaction-fixtures';
import {
  createFacilitySelectedCompaction,
  getFacilityCompactionLevelCounts
} from '../../../src/gpu-spatial-analysis/location-allocation/facility-selected-compaction';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('bounds compaction levels through the supported default-device facility capacity', () => {
  // Each level groups 256 inputs; ceil division preserves partial groups.
  for (const [count, expected] of [
    [1, [1]],
    [256, [256]],
    [257, [257, 2]],
    [65536, [65536, 256]],
    [65537, [65537, 257, 2]],
    [16777216, [16777216, 65536, 256]]
  ] as const)
    expect(getFacilityCompactionLevelCounts(count)).toEqual(expected);
});

it('parses every compaction level with bounded bindings and the exact dispatch budget', () => {
  for (const facilityCount of [1, 257, 65537, 16777216]) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const view = (name: string, length: number) =>
        createTransientView(graph, name, 'uint32', length);
      const nodes = createFacilitySelectedCompaction(graph, {
        id: 'compact',
        facilityCount,
        slots: 1,
        selected: view('selected', facilityCount),
        status: view('status', 7),
        swapState: view('state', 3),
        swapDispatch: view('dispatch', 15),
        selectedList: view('list', 1),
        slotOfFacility: view('inverse', facilityCount)
      })(0);
      // One dispatch, L scans, L-1 offset passes and one scatter.
      expect(nodes.length).toBe(2 * getFacilityCompactionLevelCounts(facilityCount).length + 1);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(source)).not.toThrow();
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(6);
      }
    } finally {
      device.destroy();
    }
  }
});

it('the disarmed fixture detects a publisher that ignores the outer gate (B3)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const forceArmed of [false, true]) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const memory = createCPUGraphMemory();
      const view = (name: string, data: Uint32Array) => {
        const output = createTransientView(graph, name, 'uint32', data.length);
        memory.write(output, data);
        return output;
      };
      const inputs = getFacilityCompactionRoundInputs('disarmed');
      const list = view('list', Uint32Array.of(0));
      const inverse = view('inverse', Uint32Array.of(0));
      const nodes = createFacilitySelectedCompaction(graph, {
        id: 'compact',
        facilityCount: 1,
        slots: 1,
        selected: view('selected', Uint32Array.of(1)),
        status: view('status', inputs.status),
        swapState: view('state', inputs.state),
        swapDispatch: view('outer', forceArmed ? Uint32Array.of(1, 1, 1) : inputs.dispatch),
        selectedList: list,
        slotOfFacility: inverse
      })(0);
      // Seed the hand-derived rank, then execute only publisher + scatter (no cooperative scan).
      const scatter = getWGSLKernelNodeRecord(graph, 'compact-swap-0-list-scatter')!;
      memory.write(
        scatter.bindings.find(binding => binding.name === 'prefix')!.view,
        Uint32Array.of(1)
      );
      executeCommandNodesOnCPU([nodes[0], nodes[nodes.length - 1]], memory);
      const publisher = getWGSLKernelNodeRecord(graph, 'compact-swap-0-list-dispatch')!;
      const dispatchView = publisher.bindings.find(binding => binding.name === 'dispatch')!.view;
      expect(
        Array.from(memory.readUint32(dispatchView)),
        `inner dispatch forceArmed=${forceArmed}`
      ).toEqual([forceArmed ? 1 : 0, forceArmed ? 1 : 0, forceArmed ? 1 : 0]);
      expect(
        memory.workgroupCounts.get('compact-swap-0-list-scatter'),
        `scatter forceArmed=${forceArmed}`
      ).toBe(forceArmed ? 1 : 0);
      // Without the scan, list clearing is not interpreted. The inverse map alone discriminates
      // the forced-arm control, even for a one-facility input.
      expect(Array.from(memory.readUint32(list))).toEqual([0]);
      expect(Array.from(memory.readUint32(inverse))).toEqual([forceArmed ? 0xffffffff : 0]);
    }
  } finally {
    device.destroy();
  }
});
