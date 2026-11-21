// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createRasterGateSlotTable,
  createRasterIterationFinalizeNode,
  createRasterIterationGroupGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  createRasterTiledRelaxation,
  getGPURasterRelaxationContinuationStateLength
} from '../../../src/raster-analysis/cost-distance/raster-relaxation';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('executed group gates count the first unchanged round, including short groups and bit 31', () => {
  // Hand-derived histories: a set bit means that round improved a value. An all-changing
  // group cannot certify convergence; a prefix of k changing rounds certifies it at k + 1.
  // The upper word catches signed-shift mistakes, while the short groups catch counting the
  // four-round default instead of the actual remaining per-submission budget.
  const cases = [
    {size: 1, bits: 0, count: 9, converged: 1},
    {size: 1, bits: 1, count: 9, converged: 0},
    {size: 3, bits: 3, count: 11, converged: 1},
    {size: 3, bits: 7, count: 11, converged: 0},
    {size: 4, bits: 0, count: 9, converged: 1},
    {size: 4, bits: 1, count: 10, converged: 1},
    {size: 4, bits: 3, count: 11, converged: 1},
    {size: 4, bits: 7, count: 12, converged: 1},
    {size: 4, bits: 15, count: 12, converged: 0},
    {size: 32, bits: 0x7fffffff, count: 40, converged: 1},
    {size: 32, bits: 0xffffffff, count: 40, converged: 0}
  ];
  const device = createNullWebGPUDevice();
  try {
    for (const testCase of cases) {
      const graph = new GPUCommandGraph(device);
      const state = createRasterIterationState(graph, 'loop', 'GPUCostDistance', 256);
      const slots = createRasterGateSlotTable(graph, {
        id: 'slots',
        operation: 'GPUCostDistance',
        groupSize: testCase.size
      });
      // wgsl_reflect ignores the byteOffset of a bound TypedArray. Check the real slot bytes
      // separately, then supply the same word in a buffer starting at zero to execute the gate.
      // GPU slot-offset binding behavior remains covered by the headless diagnostic.
      const groupSize = createTransientView(graph, 'size', 'uint32', 1);
      const gate = createRasterIterationGroupGateNode(graph, {
        id: 'group',
        operation: 'GPUCostDistance',
        state,
        maxIterations: 100,
        groupSize
      });
      const memory = createCPUGraphMemory();
      executeCommandNodesOnCPU(
        [
          slots.writeNode,
          createRasterIterationResetNode(graph, {id: 'reset', operation: 'GPUCostDistance', state})
        ],
        memory
      );
      expect(Array.from(memory.readUint32(slots.getSlot(testCase.size)))).toEqual([testCase.size]);
      memory.write(groupSize, Uint32Array.of(testCase.size));
      expect(Array.from(memory.readUint32(state.dispatch))).toEqual([1, 1, 1]);
      memory.write(state.status, Uint32Array.of(testCase.bits, 1, 8, 0));
      executeCommandNodesOnCPU([gate], memory);
      expect(Array.from(memory.readUint32(state.status)), JSON.stringify(testCase)).toEqual([
        0,
        1 - testCase.converged,
        testCase.count,
        testCase.converged
      ]);
      expect(Array.from(memory.readUint32(state.dispatch))).toEqual([1 - testCase.converged, 1, 1]);
      const converged = createTransientView(graph, 'converged', 'uint32', 1);
      const count = createTransientView(graph, 'count', 'uint32', 1);
      const limited = createTransientView(graph, 'limited', 'uint32', 1);
      executeCommandNodesOnCPU(
        [
          createRasterIterationFinalizeNode(graph, {
            id: 'publish',
            operation: 'GPUCostDistance',
            state,
            converged,
            iterationCount: count,
            iterationLimitReached: limited
          })
        ],
        memory
      );
      expect(Array.from(memory.readUint32(converged))).toEqual([testCase.converged]);
      expect(Array.from(memory.readUint32(count))).toEqual([testCase.count]);
      expect(Array.from(memory.readUint32(limited))).toEqual([1 - testCase.converged]);
    }
  } finally {
    device.destroy();
  }
});

it('executed continuation gates retain completed rounds, stop at the execution budget and restart cleanly', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const persistent = createTransientView(
      graph,
      'persistent',
      'uint32',
      getGPURasterRelaxationContinuationStateLength({width: 48, height: 1})
    );
    const {relaxation, resetNodes} = createRasterTiledRelaxation(
      graph,
      {
        id: 'continued',
        operation: 'GPUCostDistance',
        width: 48,
        height: 1,
        cellSizeMode: 'uniform',
        maxIterations: 3,
        continuation: {state: persistent}
      },
      false
    );
    const slots = createRasterGateSlotTable(graph, {
      id: 'slots',
      operation: 'GPUCostDistance',
      groupSize: 3
    });
    const groupSize = createTransientView(graph, 'size', 'uint32', 1);
    const gate = createRasterIterationGroupGateNode(graph, {
      id: 'gate',
      operation: 'GPUCostDistance',
      state: relaxation.state,
      maxIterations: 3,
      groupSize
    });
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([slots.writeNode], memory);
    expect(Array.from(memory.readUint32(slots.getSlot(3)))).toEqual([3]);
    memory.write(groupSize, Uint32Array.of(3));
    // As above, the interpreter cannot execute storage bindings with nonzero offsets. Execute
    // the two control kernels and inspect the tile reset's indirect triple; actual reset writes
    // into the continuation's stamp sub-view are exercised by the GPU continuation spec.
    const controlNodes = resetNodes.filter(node => !node.id.endsWith('-tile-reset'));
    const tileReset = resetNodes.find(node => node.id.endsWith('-tile-reset'))!;
    const resetCondition = tileReset.condition!;
    if (resetCondition.source !== 'gpu') throw new Error('Expected GPU tile reset gate');
    const resetDispatch = graph.createDataView(resetCondition.buffer, {
      format: 'uint32',
      byteOffset: resetCondition.byteOffset,
      length: 3
    });
    memory.write(persistent, Uint32Array.of(0, 0, 64, 0, 64, 0));
    memory.write(relaxation.tileStamps, Uint32Array.of(65, 0, 65));
    executeCommandNodesOnCPU(controlNodes, memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 1, 64, 0, 67, 0]);
    expect(Array.from(memory.readUint32(relaxation.tileStamps))).toEqual([65, 0, 65]);
    expect(Array.from(memory.readUint32(resetDispatch))).toEqual([0, 0, 0]);

    // Three improving rounds exhaust this execution; exhaustion is not convergence.
    memory.write(persistent, Uint32Array.of(7, 1, 64, 0, 67, 0));
    executeCommandNodesOnCPU([gate], memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 0, 67, 0, 67, 0]);
    expect(Array.from(memory.readUint32(relaxation.state.dispatch))).toEqual([0, 1, 1]);
    // A disabled later group cannot count rounds that never ran.
    executeCommandNodesOnCPU([gate], memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 0, 67, 0, 67, 0]);

    executeCommandNodesOnCPU(controlNodes, memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 1, 67, 0, 70, 0]);
    // The first unchanged round of the next execution certifies cumulative round 68.
    executeCommandNodesOnCPU([gate], memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 0, 68, 1, 70, 0]);
    // Completed runs automatically reset count and tile activity at the next execution.
    executeCommandNodesOnCPU(controlNodes, memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 1, 0, 0, 3, 0]);
    expect(Array.from(memory.readUint32(resetDispatch))).toEqual([1, 1, 1]);
    // An explicit restart also consumes the request and discards prior progress.
    memory.write(persistent, Uint32Array.of(0, 0, 64, 0, 64, 1));
    memory.write(relaxation.tileStamps, Uint32Array.of(65, 65, 65));
    executeCommandNodesOnCPU(controlNodes, memory);
    expect(Array.from(memory.readUint32(persistent, 6))).toEqual([0, 1, 0, 0, 3, 0]);
    expect(Array.from(memory.readUint32(resetDispatch))).toEqual([1, 1, 1]);
  } finally {
    device.destroy();
  }
});
