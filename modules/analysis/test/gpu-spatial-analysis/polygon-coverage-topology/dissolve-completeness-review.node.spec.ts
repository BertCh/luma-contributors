// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
useCPUExecutorTestHooks();

it('B5 refuses solver labels when no readable label provenance destination is supplied', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOutputOracleFixture(device);
  const {column, output} = fixture;
  try {
    expect(
      () =>
        new GPUCoverageDissolve({
          id: 'provisional',
          positions: column('input', 'float32x2', Float32Array.of(0, 0, 2, 0, 0, 2)),
          ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 3)),
          polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 1)),
          featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1)),
          labels: column('labels', 'uint32', Uint32Array.of(1)),
          labelsComplete: column('solver', 'uint32', Uint32Array.of(0)),
          output: {
            positions: output('result', 'float32x2', 4),
            ringOffsets: output('offsets', 'uint32', 2),
            status: {
              count: output('count', 'uint32', 1),
              overflow: output('overflow', 'uint32', 1),
              complete: output('complete', 'uint32', 1)
            }
          }
        })
    ).toThrow(/provisional.*labelsComplete.*output.status.labelsComplete/);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

for (const diagnostics of [false, true]) {
  it(`B6 includes lost snap keys in dissolve completeness, diagnostics=${diagnostics}`, () => {
    const device = createNullWebGPUDevice();
    const fixture = createOutputOracleFixture(device);
    const {column, output, graph} = fixture;
    try {
      const complete = output('complete', 'uint32', 2);
      const uncertain = diagnostics ? output('uncertain', 'uint32', 1) : undefined;
      const producer = new GPUCoverageDissolve({
        positions: column(
          'input',
          'float32x2',
          Float32Array.of(0, 0, 2, 0, 0, 2, 300000, 300000, 300002, 300000, 300000, 300002)
        ),
        ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 3, 6)),
        polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 1, 2)),
        featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1, 2)),
        labels: column('labels', 'uint32', Uint32Array.of(1, 2)),
        snapGridSize: 0.0001,
        output: {
          positions: output('result', 'float32x2', 12),
          ringOffsets: output('offsets', 'uint32', 4),
          status: {
            count: output('count', 'uint32', 1),
            overflow: output('overflow', 'uint32', 1),
            complete,
            uncertainCount: uncertain
          }
        }
      });
      const memory = createCPUGraphMemory();
      for (const entry of fixture.columns) memory.write(entry.view, entry.values);
      // Ring area uses cooperative barriers; scalar completeness does not consume those areas.
      executeCommandNodesOnCPU(
        producer.getCommandNodes(graph).filter(node => !node.id.endsWith('-ring-stats')),
        memory
      );
      // A complete result needs both disjoint triangles. Three second-component keys exceed
      // 2^24 steps and clamp together; assembly alone cannot certify the missing triangle.
      expect(Array.from(memory.readUint32(complete))).toEqual([0, 777]);
      if (uncertain) expect(memory.readUint32(uncertain)[0]).toBe(3);
    } finally {
      fixture.destroy();
      device.destroy();
    }
  });
}

it('S1 dissolve rejects the inherited completeness read/write alias during construction', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOutputOracleFixture(device);
  const {column, output, graph} = fixture;
  try {
    const backing = output('shared', 'uint32', 192);
    const scalar = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'uint32', byteOffset, length: 1});
    expect(
      () =>
        new GPUCoverageDissolve({
          positions: column('input', 'float32x2', Float32Array.of(0, 0, 2, 0, 0, 2)),
          ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 3)),
          polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 1)),
          featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1)),
          labels: column('labels', 'uint32', Uint32Array.of(1)),
          output: {
            positions: output('result', 'float32x2', 4),
            ringOffsets: output('offsets', 'uint32', 2),
            status: {
              count: output('count', 'uint32', 1),
              overflow: scalar(0),
              candidateOverflow: scalar(256),
              complete: scalar(512)
            }
          }
        })
    ).toThrow(/buffer/);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
