// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';

useCPUExecutorTestHooks();

for (const recordCapacity of [2, 3]) {
  it(`density reserves three records at a corner with capacity ${recordCapacity}`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `corner-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const requiredCount = view('uint32', 1);
    const overflow = view('uint32', 1);
    const contributor = new GPULineDensity({
      positions: view('float32x2', 2, [0.5, 0.5, 1.5, 1.5]),
      lineOffsets: view('uint32', 2, [0, 2]),
      parameters: view('float32', 4, [0, 0, 1, 1]),
      columns: 2,
      rows: 2,
      recordCapacity,
      output: {lengths: view('float32', 4), status: {overflow, requiredCount}}
    });
    const nodes = contributor.getCommandNodes(graph);
    // Execute real walk/count/emit/publication shaders; the workgroup cell-sum
    // reduction is covered headlessly, since the serial executor has no barriers.
    const publish = nodes.findIndex(node => node.id === 'line-density-publish');
    expect(publish).toBeGreaterThan(0);
    executeCommandNodesOnCPU(nodes.slice(0, publish + 1), memory);
    // Corner (1,1) advances both axes. One initial record + two axis advances = 3,
    // even though only the first and last records have positive length sqrt(1/2).
    expect(Array.from(memory.readUint32(requiredCount))).toEqual([3]);
    expect(Array.from(memory.readUint32(overflow))).toEqual([recordCapacity === 2 ? 1 : 0]);
  });
}
