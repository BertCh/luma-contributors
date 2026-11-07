// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

async function setup() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  function output<Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) {
    const components = format.endsWith('x2') ? 2 : format.endsWith('x3') ? 3 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length)
    };
  }
  return {
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}
function grid(width: number, height: number): GPURasterMetadata {
  return {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'};
}

it('flow distance rejects malformed directions and unresolved stream masks throughout a path', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const [caseIndex, testCase] of [
    {directions: [1, 3, 0], streams: [0, 0, 0]},
    {directions: [1, 64, 0], streams: [0, 0, 0]},
    {directions: [1, 1, 0], streams: [0, 0xfffffffe, 0]}
  ].entries()) {
    const distance = output(3, 'float32');
    const validity = output(3, 'uint32');
    const terminals = output(3, 'uint32');
    results.push({distance, validity, terminals});
    graph.add(
      new GPUTerrainFlowDistance({
        id: `invalid-distance-${caseIndex}`,
        grid: grid(3, 1),
        flowDirections: input(testCase.directions, 'uint32'),
        streams: input(testCase.streams, 'uint32'),
        output: {distance: distance.view, validity: validity.view, terminalCells: terminals.view}
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) {
      expect(Array.from(await readUint32(result.validity.buffer, 3))).toEqual([0, 0, 1]);
      expect(Array.from(await readUint32(result.terminals.buffer, 3))).toEqual([
        0xfffffffe, 0xfffffffe, 2
      ]);
      const values = Array.from(await readFloat32(result.distance.buffer, 3));
      expect(values[0]).toBeNaN();
      expect(values[1]).toBeNaN();
      expect(values[2]).toBe(0);
    }
  } finally {
    destroy();
  }
});

it('stream network rejects malformed directions and unresolved stream masks without publishing reaches', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const [caseIndex, testCase] of [
    {directions: [1, 3, 0], streams: [1, 1, 1]},
    {directions: [1, 64, 0], streams: [1, 1, 1]},
    {directions: [1, 0xffffffff, 0], streams: [1, 1, 1]},
    {directions: [1, 1, 0], streams: [1, 0xffffffff, 1]},
    {directions: [1, 1, 0], streams: [1, 0xfffffffe, 1]}
  ].entries()) {
    const reachCount = output(1, 'uint32');
    const unresolvedCount = output(1, 'uint32');
    results.push({reachCount, unresolvedCount});
    graph.add(
      new GPUTerrainStreamNetwork({
        id: `invalid-network-${caseIndex}`,
        grid: grid(3, 1),
        flowDirections: input(testCase.directions, 'uint32'),
        streams: input(testCase.streams, 'uint32'),
        output: {
          sourceCells: output(3, 'uint32').view,
          targetCells: output(3, 'uint32').view,
          vertexOffsets: output(4, 'uint32').view,
          positions: output(6, 'float32x2').view,
          lengths: output(3, 'float32').view,
          reachCount: reachCount.view,
          vertexCount: output(1, 'uint32').view,
          status: {overflow: output(1, 'uint32').view},
          unresolvedCount: unresolvedCount.view
        }
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) {
      expect(Array.from(await readUint32(result.reachCount.buffer, 1))).toEqual([0]);
      expect(Array.from(await readUint32(result.unresolvedCount.buffer, 1))).toEqual([2]);
    }
  } finally {
    destroy();
  }
});

it('stream reach length is independent of a large affine translation', async () => {
  const {graph, input, output, run} = await setup();
  const lengths = output(1, 'float32');
  const reachCount = output(1, 'uint32');
  graph.add(
    new GPUTerrainStreamNetwork({
      grid: {...grid(3, 1), affine: [0.25, 0, 1e8, 0, 0.25, 1e8]},
      flowDirections: input([1, 1, 0], 'uint32'),
      streams: input([1, 1, 1], 'uint32'),
      output: {
        sourceCells: output(1, 'uint32').view,
        targetCells: output(1, 'uint32').view,
        vertexOffsets: output(2, 'uint32').view,
        positions: output(3, 'float32x2').view,
        lengths: lengths.view,
        reachCount: reachCount.view,
        vertexCount: output(1, 'uint32').view,
        status: {overflow: output(1, 'uint32').view},
        unresolvedCount: output(1, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    expect(Array.from(await readUint32(reachCount.buffer, 1))).toEqual([1]);
    expect(Array.from(await readFloat32(lengths.buffer, 1))).toEqual([0.5]);
  } finally {
    destroy();
  }
});
