// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createMergeOracleFixture,
  createMergeReuseFixture,
  MERGE_OPTIONALS,
  type OracleView
} from './line-merge-output-oracle-fixture';

useCPUExecutorTestHooks();

for (const optional of MERGE_OPTIONALS) {
  it(`merge executes hand-derived chain outputs with only ${optional}`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `merge-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createMergeOracleFixture(view, optional);
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    const read = (output: typeof fixture.count) => Array.from(memory.readUint32(output));
    expect(read(fixture.count)).toEqual([2]);
    expect(read(fixture.chainOffsets)).toEqual([0, 3, 5, 5, 5, 5]);
    expect(Array.from(memory.readFloat32(fixture.positions, 10))).toEqual([
      0, 0, 1, 0, 2, 0, 7, 7, 8, 8
    ]);
    if (fixture.scalar)
      expect(read(fixture.scalar)).toEqual([optional === 'requiredCount' ? 2 : 5]);
    if (fixture.sourceIds) expect(read(fixture.sourceIds)).toEqual([0, 3, 0xffffffff, 0xffffffff]);
    if (fixture.lineOrders) expect(read(fixture.lineOrders)).toEqual([0, 1, 0xffffffff, 0]);
    if (fixture.lineReversed) expect(read(fixture.lineReversed)).toEqual([0, 1, 0, 0]);
  });
}

it('merge clears reversal when a reversed line becomes a singleton on resubmission', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: OracleView = (format, length, values) => {
    const result = createTransientView(graph, `reuse-${serial++}`, format, length);
    const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  const fixture = createMergeReuseFixture(view);
  const nodes = fixture.contributor.getCommandNodes(graph);
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(fixture.lineReversed))).toEqual([0, 1]);
  memory.write(fixture.inputOffsets, Uint32Array.of(0, 2, 3));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(fixture.lineReversed))).toEqual([0, 0]);
});

it('merge captures reversal destination before caller rebinds it to source offsets', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: OracleView = (format, length, values) => {
    const result = createTransientView(graph, `snapshot-${serial++}`, format, length);
    const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  const fixture = createMergeReuseFixture(view);
  fixture.props.output.lineReversed = graph.createDataView(fixture.inputOffsets.buffer, {
    format: 'uint32',
    length: 2
  });
  executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(fixture.inputOffsets))).toEqual([0, 2, 4]);
  expect(Array.from(memory.readUint32(fixture.lineReversed))).toEqual([0, 1]);
});
