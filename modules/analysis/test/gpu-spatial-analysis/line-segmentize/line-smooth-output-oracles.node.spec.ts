// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULineChunk, GPULineSmooth} from '../../../src/gpu-spatial-analysis/line-segmentize/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';
import {createSmoothOracleFixture} from './line-smooth-output-oracle-fixture';

useCPUExecutorTestHooks();

for (const Contributor of [GPULineChunk, GPULineSmooth]) {
  it(`${Contributor.name} explicitly refuses the shared approximation port`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view: OracleView = (format, length) =>
      createTransientView(graph, `refuse-${serial++}`, format, length);
    expect(
      () =>
        new Contributor({
          positions: view('float32x2', 2),
          lineOffsets: view('uint32', 2),
          parameters: view('float32', 4),
          iterationCount: 1,
          mode: 'chunk',
          output: {
            positions: view('float32x2', 4),
            lineOffsets: view('uint32', 2),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              approximation: view('uint32', 1)
            }
          }
        })
    ).toThrow(
      'output.status.approximation is written only by GPULineSegmentize and GPUGreatCircleArcs'
    );
  });
}

for (const optional of ['required', 'sources', 'absent'] as const) {
  it(`smooth executes the two-line hand oracle with only ${optional}`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `smooth-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createSmoothOracleFixture(view, optional);
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    const read = (output: typeof fixture.count) => Array.from(memory.readUint32(output));
    expect(read(fixture.count)).toEqual([2]);
    expect(read(fixture.overflow)).toEqual([0]);
    expect(read(fixture.lineOffsets)).toEqual([0, 4, 8]);
    expect(Array.from(memory.readFloat32(fixture.positions))).toEqual([
      0, 0, 1, 0, 3, 0, 4, 0, 10, 2, 11, 2, 13, 2, 14, 2
    ]);
    if (fixture.requiredCount) expect(read(fixture.requiredCount)).toEqual([2]);
    if (fixture.sourceLineIndices) expect(read(fixture.sourceLineIndices)).toEqual([0, 1]);
  });
}
