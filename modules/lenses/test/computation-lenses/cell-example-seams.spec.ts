// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {createGlobalCoordinateFixture} from '../../../analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinate-fixture';
import {readUint32, submitGraph} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {addExampleCellConsumers} from './cell-example-consumers-fixture';

for (const example of ['showcase', 'explorer'] as const) {
  test(`B3 ${example} rendered ring counts are current on first success and clean-loss-clean`, {
    timeout: 300000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGlobalCoordinateFixture(device);
    const consumers = addExampleCellConsumers(fixture, example, 16, true);
    consumers.sourceCells.buffer.write(Uint32Array.of(0xffffffff, 0x4820ffff, 0, 0, 0, 0, 0, 0));
    consumers.sourceCount.buffer.write(Uint32Array.of(1));
    consumers.sourceCandidateOverflow.buffer.write(Uint32Array.of(0));
    const compiled = fixture.graph.compile();
    try {
      for (const loss of [0, 1, 0]) {
        consumers.sourceOverflow.buffer.write(Uint32Array.of(loss));
        submitGraph(device, compiled, undefined);
        // One closed square has five vertices, including its repeated first vertex.
        expect(await readUint32(consumers.ringCount.buffer, 1)).toEqual([loss ? 0 : 1]);
        expect(await readUint32(consumers.ringVertexCounts.buffer, 8)).toEqual(
          loss ? new Array(8).fill(0) : [5, 0, 0, 0, 0, 0, 0, 0]
        );
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
  test(`CPL-12 ${example} actual cell consumers propagate candidate/final loss and recover`, {
    timeout: 300000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGlobalCoordinateFixture(device);
    const consumers = addExampleCellConsumers(fixture, example);
    // Quadbin z=2,x=0,y=0: one square has four outline edges. Compact/uncompact
    // retain this cell because its siblings are absent (literal Quadbin bit layout).
    consumers.sourceCells.buffer.write(Uint32Array.of(0xffffffff, 0x4820ffff, 0, 0, 0, 0, 0, 0));
    consumers.sourceCount.buffer.write(Uint32Array.of(1));
    const compiled = fixture.graph.compile();
    try {
      for (const [overflow, candidateOverflow] of [
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 0]
      ]) {
        consumers.sourceOverflow.buffer.write(Uint32Array.of(overflow));
        consumers.sourceCandidateOverflow.buffer.write(Uint32Array.of(candidateOverflow));
        submitGraph(device, compiled, undefined);
        const lost = overflow || candidateOverflow;
        expect(await readUint32(consumers.compactOverflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(consumers.compactCandidateOverflow.buffer, 1)).toEqual([lost]);
        // Aggregation has final overflow only, whereas the cover can lose candidates too.
        expect(await readUint32(consumers.outputs.get('outline-count')!.buffer, 1)).toEqual([
          overflow ? 0 : 4
        ]);
        if (example === 'showcase') {
          expect(await readUint32(consumers.outputs.get('uncompact-count')!.buffer, 1)).toEqual([
            lost ? 0 : 1
          ]);
          expect(await readUint32(consumers.outputs.get('uncompact-overflow')!.buffer, 1)).toEqual([
            0
          ]);
        }
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });

  test(`CPL-12 ${example} actual compaction final overflow reaches the next consumer`, {
    timeout: 300000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGlobalCoordinateFixture(device);
    const consumers = addExampleCellConsumers(fixture, example, 1);
    // Two nonsibling z=2 cells cannot collapse to one parent: output capacity one loses a row.
    consumers.sourceCells.buffer.write(
      Uint32Array.of(0xffffffff, 0x4820ffff, 0xffffffff, 0x4824ffff, 0, 0, 0, 0)
    );
    consumers.sourceCount.buffer.write(Uint32Array.of(2));
    consumers.sourceOverflow.buffer.write(Uint32Array.of(0));
    consumers.sourceCandidateOverflow.buffer.write(Uint32Array.of(0));
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(consumers.compactOverflow.buffer, 1)).toEqual([1]);
      expect(await readUint32(consumers.compactCandidateOverflow.buffer, 1)).toEqual([0]);
      if (example === 'showcase')
        expect(await readUint32(consumers.outputs.get('uncompact-count')!.buffer, 1)).toEqual([0]);
      consumers.sourceCount.buffer.write(Uint32Array.of(1));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(consumers.compactOverflow.buffer, 1)).toEqual([0]);
      if (example === 'showcase')
        expect(await readUint32(consumers.outputs.get('uncompact-count')!.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
