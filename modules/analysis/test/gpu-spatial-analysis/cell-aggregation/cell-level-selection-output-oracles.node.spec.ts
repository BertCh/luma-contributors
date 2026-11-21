// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellLevelSelection} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-level-selection';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('validates the optional requiredCount before wiring its writable binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const levelCounts = createTransientView(graph, 'counts', 'uint32', 3);
  const activeLevel = createTransientView(graph, 'active', 'uint32', 1);
  const count = createTransientView(graph, 'count', 'uint32', 1);
  try {
    for (const requiredCount of [
      count,
      activeLevel,
      createTransientView(graph, 'empty', 'uint32', 0)
    ]) {
      expect(
        () =>
          new GPUCellLevelSelection({
            levelCounts,
            activeLevel,
            output: {status: {count, requiredCount}}
          })
      ).toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('publishes the selected literal count into requiredCount and resets it on replay', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const levelCounts = createTransientView(graph, 'counts', 'uint32', 3);
  const activeLevel = createTransientView(graph, 'active', 'uint32', 1);
  const count = createTransientView(graph, 'count', 'uint32', 1);
  const requiredCount = createTransientView(graph, 'required', 'uint32', 1);
  const memory = createCPUGraphMemory();
  memory.write(levelCounts, new Uint32Array([5, 17, 0]));
  memory.write(count, new Uint32Array([0xa5a5a5a5]));
  memory.write(requiredCount, new Uint32Array([0xa5a5a5a5]));
  const nodes = new GPUCellLevelSelection({
    levelCounts,
    activeLevel,
    output: {status: {count, requiredCount}}
  }).getCommandNodes(graph);
  try {
    for (const [level, expected] of [
      [1, 17],
      [2, 0],
      [0, 5]
    ]) {
      memory.write(activeLevel, new Uint32Array([level]));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(count))).toEqual([expected]);
      expect(Array.from(memory.readUint32(requiredCount))).toEqual([expected]);
    }
  } finally {
    device.destroy();
  }
});

it('captures the declaration and first rows while level and count contents stay live (DGGS-115)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = (length = 1) =>
    createTransientView(graph, `selection-${serial++}`, 'uint32', length);
  const levelCounts = view(3);
  const activeLevel = view();
  const levelFirstRows = [11, 40, 100];
  const output = {
    status: {count: view(), requiredCount: view()},
    firstRow: view(),
    drawArguments: view(4)
  };
  const original = {...output, status: {...output.status}};
  const props = {levelCounts, activeLevel, levelFirstRows, output};
  const selection = new GPUCellLevelSelection(props);
  // Replace each layer of the caller's declaration, including fields within
  // the original records. Only the live buffer contents may affect replay.
  levelFirstRows.splice(0, 3, 999, 998, 997);
  output.status.count = view();
  output.status.requiredCount = view();
  output.firstRow = view();
  output.drawArguments = view(4);
  output.status = {count: view(), requiredCount: view()};
  props.output = {
    status: {count: view(), requiredCount: view()},
    firstRow: view(),
    drawArguments: view(4)
  };
  props.levelCounts = view(3);
  props.activeLevel = view();
  props.levelFirstRows = [1, 2, 3];
  expect(selection.props.levelCounts).toBe(levelCounts);
  expect(selection.props.activeLevel).toBe(activeLevel);
  expect(selection.props.output.status.count).toBe(original.status.count);
  expect(selection.props.output.status.requiredCount).toBe(original.status.requiredCount);
  expect(selection.props.output.firstRow).toBe(original.firstRow);
  expect(selection.props.output.drawArguments).toBe(original.drawArguments);
  expect(selection.props.levelFirstRows).toEqual([11, 40, 100]);
  const nodes = selection.getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  try {
    // Literal lookup oracle: the selected count and first row are independent
    // columns. Indirect vertexCount and firstVertex are caller-owned words.
    for (const [level, counts, expectedCount, expectedFirst] of [
      [1, [5, 17, 0], 17, 40],
      [2, [7, 8, 9], 9, 100],
      [0, [23, 3, 1], 23, 11]
    ] as const) {
      memory.write(levelCounts, Uint32Array.from(counts));
      memory.write(activeLevel, Uint32Array.of(level));
      memory.write(original.drawArguments, Uint32Array.of(6, 999, 2, 999));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(original.status.count))).toEqual([expectedCount]);
      expect(Array.from(memory.readUint32(original.status.requiredCount))).toEqual([expectedCount]);
      expect(Array.from(memory.readUint32(original.firstRow))).toEqual([expectedFirst]);
      expect(Array.from(memory.readUint32(original.drawArguments))).toEqual([
        6,
        expectedCount,
        2,
        expectedFirst
      ]);
    }
  } finally {
    device.destroy();
  }
});

it('R1 captures inherited declared fields at every map layer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const levelCounts = createTransientView(graph, 'inherited-counts', 'uint32', 2);
  const activeLevel = createTransientView(graph, 'inherited-level', 'uint32', 1);
  const count = createTransientView(graph, 'inherited-output', 'uint32', 1);
  const firstRow = createTransientView(graph, 'inherited-first-row', 'uint32', 1);
  const levelFirstRows = [11, 40];
  const status = Object.create({count});
  const output = Object.create({status, firstRow});
  const declaration = {levelCounts, activeLevel, levelFirstRows, output};
  const selection = new GPUCellLevelSelection(Object.create(declaration));
  declaration.levelCounts = createTransientView(graph, 'replacement-inherited-counts', 'uint32', 2);
  levelFirstRows[1] = 999;
  status.count = createTransientView(graph, 'replacement-inherited-output', 'uint32', 1);
  try {
    const nodes = selection.getCommandNodes(graph);
    expect(selection.props.levelCounts).toBe(levelCounts);
    expect(selection.props.output.status.count).toBe(count);
    expect(selection.props.levelFirstRows).toEqual([11, 40]);
    const memory = createCPUGraphMemory();
    for (const [level, counts, expectedCount, expectedFirst] of [
      [1, [5, 17], 17, 40],
      [0, [23, 9], 23, 11]
    ] as const) {
      memory.write(levelCounts, Uint32Array.from(counts));
      memory.write(activeLevel, Uint32Array.of(level));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(count))).toEqual([expectedCount]);
      expect(Array.from(memory.readUint32(firstRow))).toEqual([expectedFirst]);
    }
  } finally {
    device.destroy();
  }
});
