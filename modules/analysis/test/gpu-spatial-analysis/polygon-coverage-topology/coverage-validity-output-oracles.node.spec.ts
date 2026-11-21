// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCoverageValidity} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
it('counts invalid triangle edges on feature two rather than polygon one', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  try {
    const flags = output('flags', 'uint32', 11);
    const counts = output('counts', 'uint32', 3);
    const contributor = new GPUCoverageValidity({
      positions: column(
        'input',
        'float32x2',
        new Float32Array([0, 0, 8, 0, 8, 8, 0, 8, 2, 2, 4, 2, 2, 4, 10, 0, 12, 0, 12, 2, 10, 2])
      ),
      ringOffsets: column('rings', 'uint32', new Uint32Array([0, 4, 7, 11])),
      polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 1, 2, 3])),
      featureOffsets: column('features', 'uint32', new Uint32Array([0, 0, 1, 3])),
      parameters: column('parameters', 'float32', new Float32Array(4)),
      output: {segmentFlags: flags, featureInvalidCounts: counts}
    });
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    // Triangle (2,2)-(4,2)-(2,4) is strictly inside [0,8]^2: exactly three overlap edges.
    expect(Array.from(memory.readUint32(flags))).toEqual([0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0]);
    expect(Array.from(memory.readUint32(counts))).toEqual([0, 0, 3]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('WRT-08 clears only the advertised invalidCount word', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  try {
    const total = output('total', 'uint32', 2);
    const contributor = new GPUCoverageValidity({
      positions: column('input', 'float32x2', Float32Array.of(0, 0, 1, 0, 0, 1)),
      ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 3)),
      polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 1)),
      featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1)),
      parameters: column('parameters', 'float32', new Float32Array(4)),
      output: {segmentFlags: output('flags', 'uint32', 3), status: {invalidCount: total}}
    });
    const memory = createCPUGraphMemory();
    memory.write(total, Uint32Array.of(99, 0x12345678));
    executeCommandNodesOnCPU(
      contributor
        .getCommandNodes(graph)
        .filter(node => node.id === 'coverage-validity-clear-total'),
      memory
    );
    expect(Array.from(memory.readUint32(total))).toEqual([0, 0x12345678]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
