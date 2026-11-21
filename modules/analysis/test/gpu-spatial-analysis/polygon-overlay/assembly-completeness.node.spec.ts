// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createOverlayAssemblyFixture} from './overlay-assembly-fixture';
import {createOverlayAssemblyCompletenessNode} from '../../../src/gpu-spatial-analysis/polygon-overlay/overlay-assembly-completeness';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('GEO-125 propagates open segments and conflicts with spare final capacity and clears on clean publication', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const segmentFlags = view('flags', 4),
    segmentCount = view('count'),
    openSegmentCount = view('open'),
    candidateOverflow = view('candidate');
  const node = createOverlayAssemblyCompletenessNode(graph, {
    id: 'probe',
    segmentFlags,
    segmentCount,
    openSegmentCount,
    candidateOverflow
  });
  const memory = createCPUGraphMemory();
  memory.write(segmentCount, Uint32Array.of(4));
  for (const [open, flags, expected] of [
    [3, [0, 0, 0, 0], 1],
    [0, [0, 4, 0, 0], 1],
    [0, [0, 0, 0, 0], 0]
  ] as const) {
    // The overlay publisher resets this word before assembly loss is folded in on every submit.
    memory.write(candidateOverflow, Uint32Array.of(0));
    memory.write(openSegmentCount, Uint32Array.of(open));
    memory.write(segmentFlags, Uint32Array.from(flags));
    executeCommandNodesOnCPU([node], memory);
    expect(memory.readUint32(candidateOverflow)[0]).toBe(expected);
  }
  device.destroy();
});

it('S2 executes the production overlay publisher and loss destination on open, conflict and clean submissions', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOverlayAssemblyFixture(device);
  try {
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    const publication = fixture.nodes.filter(node =>
      ['overlay-review-publish', 'overlay-review-assembly-completeness'].includes(node.id)
    );
    memory.write(fixture.internal('ring-overflow'), Uint32Array.of(0));
    memory.write(fixture.geometry.featureOffsets, Uint32Array.of(0, 1, 1, 1, 1, 1, 1, 1, 1));
    for (const [open, flags, expected] of [
      [3, [0, 0, 0, 0, 0], 1],
      [0, [0, 0, 0, 0, 4], 1],
      [0, [0, 0, 0, 0, 0], 0]
    ] as const) {
      memory.write(fixture.boundaryStatus.count, Uint32Array.of(5));
      memory.write(fixture.internal('assembly-open-count'), Uint32Array.of(open));
      memory.write(fixture.internal('assembly-flags'), Uint32Array.from(flags));
      // No manual reset of the top-level word: the real overlay publisher must reset it.
      executeCommandNodesOnCPU(publication, memory);
      expect(memory.readUint32(fixture.status.candidateOverflow)[0]).toBe(expected);
      expect(memory.readUint32(fixture.boundaryStatus.candidateOverflow)[0]).toBe(0);
      expect(memory.readUint32(fixture.status.overflow)[0]).toBe(0);
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
