// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';
import {createSplitOracleFixture} from './line-split-output-oracle-fixture';

useCPUExecutorTestHooks();

it('split refuses top-level uncertainCount aliased to input line offsets', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view: OracleView = (format, length) =>
    createTransientView(graph, `alias-${serial++}`, format, length);
  const lineOffsets = view('uint32', 3);
  const uncertainCount = graph.createDataView(lineOffsets.buffer, {format: 'uint32', length: 1});
  expect(
    () =>
      new GPULineSplit({
        lines: {kind: 'lines', positions: view('float32x2', 5), lineOffsets},
        intersectionCapacity: 8,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: view('float32x2', 8),
              lineOffsets: view('uint32', 5)
            },
            sourceIds: view('uint32', 4),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          },
          status: {uncertainCount}
        }
      })
  ).toThrow(/^line-split outputs must not share buffers with inputs/);
});

for (const port of [
  'geometry.sourceIds',
  'uncertainCount',
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation',
  'ringCount',
  'requiredRingCount'
]) {
  it(`split refuses unwritten pieces.${port} instead of accepting a poisoned output`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
      createTransientView(graph, `split-${serial++}`, format, length);
    const unsupported = view('uint32', 1);
    expect(
      () =>
        new GPULineSplit({
          lines: {kind: 'lines', positions: view('float32x2', 2), lineOffsets: view('uint32', 2)},
          intersectionCapacity: 8,
          output: {
            pieces: {
              geometry: {
                kind: 'lines',
                positions: view('float32x2', 4),
                lineOffsets: view('uint32', 2),
                ...(port === 'geometry.sourceIds' ? {sourceIds: unsupported} : {})
              },
              sourceIds: view('uint32', 1),
              status: {
                count: view('uint32', 1),
                overflow: view('uint32', 1),
                vertexCount: view('uint32', 1),
                ...(port !== 'geometry.sourceIds' ? {[port]: unsupported} : {})
              }
            }
          }
        })
    ).toThrow(`line-split output.pieces.${port.startsWith('geometry.') ? port : `status.${port}`}`);
  });
}

for (const selfContact of [false, true]) {
  it(`split executes outgoing edge and self-contact literals (selfContact=${selfContact})`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `edge-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createSplitOracleFixture(view, selfContact, 'edges');
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.count))).toEqual([selfContact ? 3 : 4]);
    expect(Array.from(memory.readUint32(fixture.lineOffsets))).toEqual(
      selfContact ? [0, 2, 6, 8, 8] : [0, 2, 4, 6, 8]
    );
    const edges = memory.readUint32(fixture.vertexSourceEdges!);
    expect((selfContact ? [0, 2, 3, 4, 6] : [0, 2, 4, 6]).map(row => edges[row])).toEqual(
      selfContact ? [2147483648, 2147483648, 1, 2147483650, 2147483650] : [0, 0, 2, 2]
    );
  });
}
