// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLKernelNodeRecord} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createOverlayInputSnapshotFixture} from './overlay-input-snapshot-fixture';

useCPUExecutorTestHooks();
for (const packed of [false]) {
  for (const ids of [false, true]) {
    it(`captures operand maps and caller ID views before deferred lowering (packed=${packed}, ids=${ids})`, () => {
      const device = createNullWebGPUDevice();
      const fixture = createOverlayInputSnapshotFixture(device, {mutate: true, packed, ids});
      const memory = createCPUGraphMemory();
      for (const entry of fixture.columns) memory.write(entry.view, entry.values);
      try {
        // Seed a hand-worked assembled triangle, then execute the actual producer's
        // representative-row reduction and final row-to-ID publishers. Cooperative ring
        // kernels are exercised by the companion headless case.
        const sourceStart = fixture.nodes.findIndex(node => node.id.endsWith('-source-shells'));
        const sourceEnd = fixture.nodes.findIndex(node =>
          node.id.endsWith('-publish-feature-sources')
        );
        const sourceNodes = fixture.nodes.slice(sourceStart, sourceEnd + 1);
        const view = (nodeSuffix: string, bindingName: string) => {
          const node = fixture.nodes.find(node => node.id.endsWith(nodeSuffix))!;
          return getWGSLKernelNodeRecord(fixture.graph, node.id)!.bindings.find(
            binding => binding.name === bindingName
          )!.view;
        };
        memory.write(view('-source-shells', 'shells'), Uint32Array.of(0, 0xffffffff));
        memory.write(view('-source-shells', 'areas'), Float32Array.of(0.5, 0));
        memory.write(view('-source-shells', 'ringCount'), Uint32Array.of(1));
        memory.write(
          view('-reduce-feature-sources', 'segmentRings'),
          Uint32Array.of(0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff)
        );
        memory.write(fixture.boundary.status.count, Uint32Array.of(3));
        memory.write(fixture.boundary.sourceFeatureIds, Uint32Array.of(1, 1, 1));
        memory.write(fixture.boundary.operandIds, Uint32Array.of(0, 1, 0));
        executeCommandNodesOnCPU(
          [
            ...sourceNodes,
            ...fixture.nodes.filter(
              node =>
                node.id.endsWith('-source-feature-ids') ||
                node.id.endsWith('-publish-source-feature-ids')
            )
          ],
          memory
        );
        expect(Array.from(memory.readUint32(fixture.boundary.status.count))).toEqual([3]);
        expect(
          Array.from(memory.readUint32(fixture.boundary.sourceFeatureIds)).slice(0, 3)
        ).toEqual(ids ? [101, 202, 101] : [1, 1, 1]);
        // x>=1, y>=0, x+y<=2: bottom and diagonal belong to left row 1,
        // vertical edge belongs to right row 1. Representative rows are never caller IDs.
        expect(memory.readUint32(fixture.featureSources.sourceRows)[0]).toBe(1);
        expect(fixture.producer.props.left.positions === fixture.originalLeftPositions).toBe(true);
        expect(fixture.producer.props.right?.featureOffsets === fixture.originalRightFeatures).toBe(
          true
        );
        expect(fixture.readsAtConstruction).toBe(ids ? 2 : 0);
        expect(fixture.identityReads).toBe(fixture.readsAtConstruction);
      } finally {
        fixture.destroy();
        device.destroy();
      }
    });
  }
}

it('captures packed identity columns at byte offsets zero and 256 without rereading accessors', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOverlayInputSnapshotFixture(device, {
    mutate: true,
    packed: true,
    ids: true
  });
  try {
    const node = fixture.nodes.find(node => node.id.endsWith('-source-feature-ids'))!;
    const bindings = getWGSLKernelNodeRecord(fixture.graph, node.id)!.bindings;
    const left = bindings.find(binding => binding.name === 'leftIds')!.view;
    const right = bindings.find(binding => binding.name === 'rightIds')!.view;
    expect(left.buffer === right.buffer).toBe(true);
    expect([left.byteOffset, right.byteOffset]).toEqual([0, 256]);
    expect(fixture.readsAtConstruction).toBe(2);
    expect(fixture.identityReads).toBe(2);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
