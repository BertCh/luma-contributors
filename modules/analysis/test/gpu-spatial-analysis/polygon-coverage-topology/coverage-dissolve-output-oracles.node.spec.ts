// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createDissolveCapacityOracle,
  createDissolveTouchOracle
} from './coverage-dissolve-output-oracles';

useCPUExecutorTestHooks();
it('publishes two written rings but three required rings', () => {
  const device = createNullWebGPUDevice();
  const fixture = createDissolveCapacityOracle(device);
  try {
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    const end = fixture.nodes.findIndex(node => node.id.endsWith('-ring-stats'));
    executeCommandNodesOnCPU(fixture.nodes.slice(0, end), memory);
    executeCommandNodesOnCPU(
      fixture.nodes.filter(node => node.id.endsWith('-assembly-publish')),
      memory
    );
    for (const [name, expected] of [
      ['count', 2],
      ['requiredCount', 3],
      ['overflow', 1]
    ] as const)
      expect(
        Array.from(memory.readUint32(fixture.columns.find(entry => entry.name === name)!.view))
      ).toEqual([expected]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('publishes nonzero touching diagnostics with each view alone and both absent', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const optional of ['both', 'segmentFlags', 'touchingSegmentCount', undefined] as const) {
      const fixture = createDissolveTouchOracle(device, optional);
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        const end = fixture.nodes.findIndex(node => node.id.endsWith('-ring-stats'));
        executeCommandNodesOnCPU(fixture.nodes.slice(0, end), memory);
        executeCommandNodesOnCPU(
          fixture.nodes.filter(
            node => node.id.endsWith('-assembly-publish') || node.id.endsWith('-publish-counters')
          ),
          memory
        );
        const read = (name: string) =>
          Array.from(memory.readUint32(fixture.columns.find(entry => entry.name === name)!.view));
        expect(read('count')).toEqual([2]);
        expect(read('overflow')).toEqual([0]);
        expect(read('offsets')).toEqual([0, 5, 10]);
        if (optional === 'both' || optional === 'segmentFlags')
          expect(read('segmentFlags')).toEqual([0, 1, 0, 0, 0, 0, 0, 1]);
        if (optional === 'both' || optional === 'touchingSegmentCount')
          expect(read('touchingSegmentCount')).toEqual([2]);
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('CPL-14 republishes label completion provenance on reuse without discarding the incumbent', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  try {
    const inputComplete = column('label-completion', 'uint32', Uint32Array.of(0));
    const complete = output('completion', 'uint32', 2);
    const contributor = new GPUCoverageDissolve({
      positions: column('positions', 'float32x2', Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1)),
      ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 4)),
      polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 1)),
      featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1)),
      labels: column('labels', 'uint32', Uint32Array.of(3)),
      labelsComplete: inputComplete,
      output: {
        positions: output('result', 'float32x2', 5),
        ringOffsets: output('offsets', 'uint32', 2),
        status: {
          count: output('count', 'uint32', 1),
          overflow: output('overflow', 'uint32', 1),
          labelsComplete: complete
        }
      }
    });
    const legacy = {...contributor.props, snapTolerance: 2};
    expect(() => new GPUCoverageDissolve(legacy)).toThrow(/snapTolerance.*snapGridSize/);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.id.endsWith('-assembly-publish'))).toBe(true);
    const memory = createCPUGraphMemory();
    memory.write(complete, Uint32Array.of(99, 0x12345678));
    for (const value of [0, 1, 0]) {
      memory.write(inputComplete, Uint32Array.of(value));
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.endsWith('-labels-complete')),
        memory
      );
      expect(Array.from(memory.readUint32(complete))).toEqual([value, 0x12345678]);
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
