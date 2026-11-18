// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GPUGather, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {GPUTimeWindowFilter} from '../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {getGPUTimeWindowFilterParameterValues} from '../../src/dataframe-analysis/time-window-filter/time-window-parameters';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

test('final1b F9 reversed identities do not change sorted rows or gathered attributes', () => {
  const sorted = true;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `identity-${serial++}`, format, length);
  try {
    const times = view('float32', 2),
      sourceIds = view('uint32', 2),
      parameters = view('float32', 8);
    const rows = view('uint32', 1),
      ids = view('uint32', 1),
      count = view('uint32', 1),
      overflow = view('uint32', 1);
    const attributes = view('float32', 2),
      gathered = view('float32', 1);
    const contributor = new GPUTimeWindowFilter({
      timestamps: times,
      sourceIds,
      window: parameters,
      sorted,
      output: {rows, ids, status: {count, overflow}}
    });
    const nodes = [
      ...contributor.getCommandNodes(graph),
      ...new GPUGather({source: attributes, indices: rows, output: gathered}).getCommandNodes(graph)
    ];
    const memory = createCPUGraphMemory();
    memory.write(times, new Float32Array([0, 10]));
    memory.write(sourceIds, new Uint32Array([1, 0]));
    memory.write(parameters, getGPUTimeWindowFilterParameterValues({start: 0, end: 1}));
    memory.write(attributes, new Float32Array([10, 20]));
    memory.write(rows, new Uint32Array([37]));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(count))).toEqual([1]);
    expect(Array.from(memory.readUint32(rows))).toEqual([0]);
    expect(Array.from(memory.readUint32(ids))).toEqual([1]);
    expect(Array.from(memory.readFloat32(gathered))).toEqual([10]);
  } finally {
    device.destroy();
  }
});
