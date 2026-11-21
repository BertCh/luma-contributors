// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {createOverlayAssemblyFixture} from './overlay-assembly-fixture';

it('GEO-125 reports dropped open edges, a continuation conflict and a subsequent clean ring', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOverlayAssemblyFixture(device);
  try {
    // Inject a controlled noded boundary into the actual overlay's assembler, then execute its
    // production publication/aggregation nodes. This does not manually wire/reset either status.
    const source = fixture.column(
      'controlled-boundary',
      'float32x4',
      Float32Array.of(0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0)
    );
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'controlled-assembly-handoff',
        operation: 'overlay-test',
        variant: 'handoff',
        invocationCount: 5,
        bindings: [
          {name: 'source', view: source, type: 'f32', access: 'read'},
          {
            name: 'endpoints',
            view: fixture.internal('lattice-boundary'),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'groups',
            view: fixture.internal('boundary-groups'),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `let row = index * 4u;
  endpoints[endpointsOffset + row] = source[sourceOffset + row];
  endpoints[endpointsOffset + row + 1u] = source[sourceOffset + row + 1u];
  endpoints[endpointsOffset + row + 2u] = source[sourceOffset + row + 2u];
  endpoints[endpointsOffset + row + 3u] = source[sourceOffset + row + 3u];
  groups[groupsOffset + index] = 0u;`
      })
    );
    for (const node of fixture.nodes)
      if (
        node.id.startsWith('overlay-review-assemble-') ||
        ['overlay-review-publish', 'overlay-review-assembly-completeness'].includes(node.id)
      )
        fixture.graph.add(node);
    const countInput = fixture.columns.find(entry => entry.name === 'boundary-count')!.buffer;
    const boundaryCandidate = fixture.columns.find(
      entry => entry.name === 'boundary-candidate'
    )!.buffer;
    for (const [liveCount, candidate] of [
      [3, 1],
      [5, 1],
      [4, 0]
    ]) {
      countInput.write(Uint32Array.of(liveCount));
      const result = await fixture.runGPU();
      expect(result['overflow']).toEqual([0]);
      expect(result['candidate']).toEqual([candidate]);
      expect(await readUint32(boundaryCandidate, 1)).toEqual([0]);
      if (liveCount === 3) expect(result['count']).toEqual([0]);
      if (liveCount === 4) expect(result['count']).toEqual([1]);
    }
  } finally {
    fixture.destroy();
  }
});
