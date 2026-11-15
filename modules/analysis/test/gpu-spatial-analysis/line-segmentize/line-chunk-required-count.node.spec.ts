// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import * as prefixKernels from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {
  GPULineChunk,
  getGPULineChunkParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

// Execute the contributor's count, range and publish WGSL with upstream scan emulation.
// The cooperative prefix needs barriers, so supply exact measures of horizontal two-point lines.
function createChunkCountFixture(lengths: number[], lineCapacity: number, withRequirement = true) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `chunk-count-${sequence++}`, format, length);
  const positions = view('float32x2', lengths.length * 2);
  const lineOffsets = view('uint32', lengths.length + 1);
  const parameters = view('float32', 4);
  const output = {
    positions: view('float32x2', lineCapacity * 2),
    lineOffsets: view('uint32', lineCapacity + 1),
    sourceLineIndices: view('uint32', lineCapacity),
    status: {
      count: view('uint32', 1),
      requiredCount: withRequirement ? view('uint32', 1) : undefined,
      overflow: view('uint32', 1)
    }
  };
  const prefix = vi
    .spyOn(prefixKernels, 'createLinePrefixNode')
    .mockImplementation((targetGraph, props) =>
      createWGSLKernelNode(targetGraph, {
        id: props.id,
        operation: 'GPULineChunk',
        variant: 'exact-test-measures',
        invocationCount: lengths.length * 2,
        bindings: [{name: 'measures', view: props.rowMeasures!, type: 'f32', access: 'read_write'}],
        declarations: `const MEASURES = array<f32, ${lengths.length * 2}>(${lengths.flatMap(length => ['0.0', `${length}.0`]).join(', ')});`,
        body: 'measures[measuresOffset + index] = MEASURES[index];'
      })
    );
  let nodes;
  try {
    nodes = new GPULineChunk({
      positions,
      lineOffsets,
      parameters,
      mode: 'chunk',
      output
    }).getCommandNodes(graph);
  } finally {
    prefix.mockRestore();
  }
  const memory = createCPUGraphMemory();
  memory.write(positions, new Float32Array(lengths.flatMap(length => [0, 0, length, 0])));
  memory.write(
    lineOffsets,
    new Uint32Array(Array.from({length: lengths.length + 1}, (_, index) => index * 2))
  );
  return {
    run(chunkLength: number) {
      memory.write(parameters, getGPULineChunkParameterValues({chunkLength}));
      // Balanced vertex emission also needs workgroup barriers; headless specs cover positions.
      executeCommandNodesOnCPU(
        nodes.filter(node => !node.id.endsWith('-emit')),
        memory
      );
      return {
        count: memory.readUint32(output.status.count)[0],
        requiredCount: output.status.requiredCount
          ? memory.readUint32(output.status.requiredCount)[0]
          : undefined,
        overflow: memory.readUint32(output.status.overflow)[0],
        offsets: Array.from(memory.readUint32(output.lineOffsets)),
        sources: Array.from(memory.readUint32(output.sourceLineIndices))
      };
    },
    destroy: () => device.destroy()
  };
}

it('reports the full requirement of lines that individually exceed output capacity', () => {
  const fixture = createChunkCountFixture([100, 8], 2);
  try {
    expect(fixture.run(1)).toEqual({
      count: 2,
      requiredCount: 108,
      overflow: 1,
      offsets: [0, 2, 4],
      sources: [0, 0]
    });
    // Reusing the same nodes rewrites all statuses for the next frame.
    const complete = fixture.run(100);
    expect(complete).toMatchObject({count: 2, requiredCount: 2, overflow: 0, sources: [0, 1]});
  } finally {
    fixture.destroy();
  }
});

it.each([
  [[100], 1e-30], // One line's requirement exceeds uint32.
  [[2 ** 31, 2 ** 31], 1], // Each count fits, but the sum wraps exactly to zero.
  [[2 ** 31, 2 ** 31, 8], 1] // The wrapped sum is small and nonzero.
])('saturates excessive requirements for lengths %j', (lengths, chunkLength) => {
  const fixture = createChunkCountFixture(lengths as number[], 2);
  try {
    expect(fixture.run(chunkLength as number)).toMatchObject({
      count: 2,
      requiredCount: 0xffffffff,
      overflow: 1
    });
  } finally {
    fixture.destroy();
  }
});

it('retains bounded geometry when the optional requirement is omitted', () => {
  const fixture = createChunkCountFixture([100], 2, false);
  try {
    expect(fixture.run(1)).toMatchObject({
      count: 2,
      requiredCount: undefined,
      overflow: 1,
      offsets: [0, 2, 4]
    });
  } finally {
    fixture.destroy();
  }
});
