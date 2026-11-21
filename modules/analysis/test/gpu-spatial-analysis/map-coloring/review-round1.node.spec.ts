// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMapColoring} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {GPUMapColoringOperation} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {CPUMapColoringOperation} from '../../../src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();

it('B3 carries both operation loss names and recovers after loss clears', async () => {
  const props = {
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'loss', candidateOverflow: 'candidate'}
    },
    rowCount: 3,
    parameters: 'parameters',
    output: {
      colors: 'colors',
      colorCount: 'colorCount',
      conflictCount: 'conflictCount',
      status: {converged: 'done', iterationCount: 'rounds'}
    }
  };
  const operation = new CPUMapColoringOperation(props);
  expect(operation.inputs['overflow']).toBe('loss');
  const gpu = new GPUMapColoringOperation(props);
  expect(gpu.inputs['candidateOverflow']).toBe('candidate');
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = gpu.createContributor(
    createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 4},
      neighbors: {format: 'uint32', length: 2},
      weights: {format: 'float32', length: 2},
      parameters: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidateOverflow: {format: 'uint32', length: 1},
      colors: {format: 'uint32', length: 3},
      colorCount: {format: 'uint32', length: 1},
      conflictCount: {format: 'uint32', length: 1},
      converged: {format: 'uint32', length: 1},
      iterationCount: {format: 'uint32', length: 1}
    })
  );
  if (!(contributor instanceof GPUMapColoring))
    throw new Error('Expected a map coloring contributor');
  expect(
    contributor.getCommandNodes(graph).filter(node => node.id.includes('-incomplete-'))
  ).toHaveLength(8);
  const vector = (values: Uint32Array | Float32Array): GPUCPUVector => ({
    format: values instanceof Float32Array ? 'float32' : 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  for (const [overflow, candidateOverflow] of [
    [0, 0],
    [1, 0],
    [0, 1],
    [0, 0]
  ]) {
    const result = await operation.execute({
      inputs: {
        offsets: vector(Uint32Array.of(0, 1, 2, 2)),
        neighbors: vector(Uint32Array.of(1, 0)),
        weights: vector(Float32Array.of(1, 1)),
        parameters: vector(Uint32Array.of(0)),
        overflow: vector(Uint32Array.of(overflow)),
        candidateOverflow: vector(Uint32Array.of(candidateOverflow))
      }
    });
    const loss = overflow || candidateOverflow;
    expect(Array.from(result['colors'] as Uint32Array).sort((left, right) => left - right)).toEqual(
      loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 1]
    );
    expect(Array.from(result['colorCount'] as Uint32Array)).toEqual([loss ? 0 : 2]);
    expect(Array.from(result['conflictCount'] as Uint32Array)).toEqual([0]);
    expect(Array.from(result['done'] as Uint32Array)).toEqual([loss ? 0 : 1]);
    expect(Array.from(result['rounds'] as Uint32Array)).toEqual([1]);
  }
});

it('S3 clears each packed scalar from a distinct nonzero word with an omission control', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `packed-review-${serial++}`, format, length);
  const packed = view('uint32', 2);
  const colorCount = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const conflictCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const loss = view('uint32', 1);
  const nodes = new GPUMapColoring({
    weights: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2),
      status: {overflow: loss}
    },
    parameters: view('uint32', 1),
    output: {colors: view('uint32', 2), colorCount, conflictCount}
  })
    .getCommandNodes(graph)
    .filter(node => node.id.includes('-incomplete-'));
  for (const omitConflictWriter of [true, false]) {
    const memory = createCPUGraphMemory();
    memory.write(packed, Uint32Array.of(17, 23));
    memory.write(loss, Uint32Array.of(1));
    executeGeometryNodesOnCPU(
      nodes.filter(node => !omitConflictWriter || !node.id.endsWith('-conflictCount')),
      memory
    );
    expect(Array.from(memory.readUint32(packed))).toEqual(omitConflictWriter ? [0, 23] : [0, 0]);
  }
});
