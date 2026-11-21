// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {addStopValueCubeRecipe} from '../../../src/gpu-spatial-analysis/index';
import {getGPUAnalysisOperator} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createStopValueCubeFixture} from './stop-value-cube-fixture';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/index';
import {GPUPeriodIndex} from '../../../src/dataframe-analysis/period-index/index';
import {GPUValueCube} from '../../../src/dataframe-analysis/value-cube/index';

it('CCK-01 registers the exact public stop-cube identity and its producer contract', () => {
  expect(addStopValueCubeRecipe).toBeTypeOf('function');
  const operator = getGPUAnalysisOperator({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'addStopValueCubeRecipe'
  });
  expect(operator).toBeDefined();
  expect(operator?.package).toBe('@luma-contributors/analysis/gpu-spatial-analysis');
  expect(operator?.exportName).toBe('addStopValueCubeRecipe');
  expect(operator?.status).toEqual(
    expect.arrayContaining(['overflow', 'candidateOverflow', 'invalidCount'])
  );
  expect(operator?.evidence.contract).toBe('GPUStopValueCubeRecipeProps');
  const description = `${operator?.summary} ${operator?.limitation}`;
  for (const term of [
    /compact|live stop prefix/i,
    /signed.*Int64/i,
    /fixed.*period/i,
    /dictionary/i,
    /rows.*weights|weights.*row/i,
    /cell.major/i,
    /suppress/i
  ])
    expect(description).toMatch(term);
});

it('CCK-01 producer declares signed times and its glue propagates compact rows and live loss', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const props = createStopValueCubeFixture((format, values, length) => {
    const view = createTransientView(graph, `input-${serial++}`, format, length);
    memory.write(view, values);
    return view;
  });
  try {
    const result = addStopValueCubeRecipe(graph, props);
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
    const join = result.contributors.find(
      contributor => contributor instanceof GPUKeyLookup
    ) as GPUKeyLookup;
    expect(join.props.rightKeys).toBe(props.dictionary);
    expect(join.props.output?.rightRows).toBe(result.intermediates.cellRows);
    const period = result.contributors.find(
      contributor => contributor instanceof GPUPeriodIndex
    ) as GPUPeriodIndex;
    const cube = result.contributors.find(
      contributor => contributor instanceof GPUValueCube
    ) as GPUValueCube;
    expect(period.props.periods).toEqual(props.periods);
    expect(period.props.temporalContext).toEqual(props.temporalContext);
    expect(period.props.timestamps).toBe(result.intermediates.times);
    expect(cube.props.cells).toBe(result.intermediates.cellRows);
    expect(cube.props.periods).toBe(result.intermediates.periods);
    expect(cube.props.cellCount).toBe(2);
    expect(cube.props.periodCount).toBe(3);
    // WgslExec does not preserve the high unsigned words in this vector assignment. The
    // device case pins the bin for -5 seconds; use +5 as the interpreter's copy control.
    memory.write(props.timestamps, Uint32Array.of(5, 0, 5, 0, 15, 0));
    // Cooperative primitives and chunk-group dispatch need a real device. Execute only the
    // recipe-owned time/loss/suppression glue here; the device spec runs the complete chain.
    const run = () => {
      memory.write(result.outputs.counts, Uint32Array.of(0, 0, 1, 1, 0, 0));
      executeCommandNodesOnCPU(
        nodes.filter(node =>
          /stop-value-cube-(stop-time$|overflow-|candidateOverflow-|suppress$)/.test(node.id)
        ),
        memory
      );
    };
    run();
    expect(Array.from(memory.readUint32(result.intermediates.times))).toEqual([5, 0, 15, 0, 0, 0]);
    expect(Array.from(memory.readUint32(result.intermediates.live))).toEqual([1, 1, 0]);
    // SE dictionary row 0 has stop at t=15 (period 2); NW row 1 has t=-5 (period 0).
    expect(Array.from(memory.readUint32(result.outputs.counts))).toEqual([0, 0, 1, 1, 0, 0]);
    expect(result.weights.offsets).toBe(props.weights.offsets);
    for (const [input, output] of [
      [props.stops.status.overflow!, result.status.overflow],
      [props.stops.status.candidateOverflow!, result.status.candidateOverflow],
      [props.sliceStatuses![0].overflow!, result.status.overflow],
      [props.sliceStatuses![0].candidateOverflow!, result.status.candidateOverflow],
      [props.weights.status!.overflow!, result.status.overflow],
      [props.weights.status!.candidateOverflow!, result.status.candidateOverflow]
    ] as const) {
      memory.write(input, Uint32Array.of(1));
      run();
      expect(Array.from(memory.readUint32(output)), input.buffer.id).toEqual([1]);
      expect(Array.from(memory.readUint32(result.outputs.counts))).toEqual([0, 0, 0, 0, 0, 0]);
      memory.write(input, Uint32Array.of(0));
      run();
      expect(Array.from(memory.readUint32(result.outputs.counts))).toEqual([0, 0, 1, 1, 0, 0]);
    }
  } finally {
    device.destroy();
  }
});

it('W5-19 rejects dictionary/weights row mismatch before mutation and retries the same ID', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const props = createStopValueCubeFixture((format, _values, length) =>
    createTransientView(graph, `input-${serial++}`, format, length)
  );
  try {
    const add = vi.spyOn(graph, 'add');
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() =>
      addStopValueCubeRecipe(graph, {...props, id: 'retry', dictionary: props.timestamps})
    ).toThrow(/weights rows must equal dictionary rows/);
    expect(add).not.toHaveBeenCalled();
    expect(allocate).not.toHaveBeenCalled();
    expect(() => addStopValueCubeRecipe(graph, {...props, id: 'retry'})).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B6 cardinality-only weights loss suppresses the cube and recovers at the retained count', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const props = createStopValueCubeFixture((format, values, length) => {
    const view = createTransientView(graph, `input-${serial++}`, format, length);
    memory.write(view, values);
    return view;
  });
  const requiredCount = createTransientView(graph, 'required', 'uint32', 1);
  props.weights.status = {requiredCount};
  try {
    const result = addStopValueCubeRecipe(graph, props);
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes.filter(node =>
      /stop-value-cube-(overflow-|candidateOverflow-|cardinality|suppress$)/.test(node.id)
    );
    for (const required of [2, 3, 2]) {
      memory.write(requiredCount, Uint32Array.of(required));
      memory.write(result.outputs.counts, Uint32Array.of(0, 0, 1, 1, 0, 0));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(result.status.overflow))).toEqual([
        required === 3 ? 1 : 0
      ]);
      expect(Array.from(memory.readUint32(result.status.candidateOverflow))).toEqual([0]);
      expect(Array.from(memory.readUint32(result.outputs.counts))).toEqual(
        required === 3 ? [0, 0, 0, 0, 0, 0] : [0, 0, 1, 1, 0, 0]
      );
    }
  } finally {
    device.destroy();
  }
});
