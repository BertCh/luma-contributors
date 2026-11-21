// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  expectTruncatedPairPrefix,
  withTestResourceCleanup
} from '../geometry-refinement/geometry-output-review-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

async function runBoundaryJoin(
  scene: 'nine-matches' | 'degenerate-within',
  candidateCapacity?: number,
  containment: 'within' | 'coveredBy' = 'within',
  multipart = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources: {destroy(): void}[] = [];
  return withTestResourceCleanup(resources, async () => {
    const graph = new GPUCommandGraph(device);
    const input = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      resources.push(buffer);
      return importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      );
    };
    const output = (name: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      resources.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const nineMatches = scene === 'nine-matches';
    const capacity = nineMatches ? 9 : 2;
    const left: GPUSpatialJoinGeometry = nineMatches
      ? {
          kind: 'polygons',
          positions: input('left', 'float32x2', Float32Array.of(0, 0, 100, 0, 100, 10, 0, 10)),
          featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
          polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
          ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4))
        }
      : {
          kind: 'lines',
          positions: input('left', 'float32x2', Float32Array.of(0, 0, 0, 0, 0.5, 0, 0.5, 0)),
          lineOffsets: input('left-offsets', 'uint32', Uint32Array.of(0, 2, 4)),
          ...(multipart
            ? {featureOffsets: input('left-features', 'uint32', Uint32Array.of(0, 1, 2))}
            : {})
        };
    const right: GPUSpatialJoinGeometry = nineMatches
      ? {
          kind: 'points',
          positions: input(
            'right',
            'float32x2',
            Float32Array.from([5, 5, 15, 5, 25, 5, 35, 5, 45, 5, 55, 5, 65, 5, 75, 5, 85, 5])
          )
        }
      : {
          kind: 'lines',
          positions: input('right', 'float32x2', Float32Array.of(0, 0, 1, 0)),
          lineOffsets: input('right-offsets', 'uint32', Uint32Array.of(0, 2)),
          ...(multipart
            ? {featureOffsets: input('right-features', 'uint32', Uint32Array.of(0, 1))}
            : {})
        };
    const leftRows = output('left-rows', capacity),
      rightRows = output('right-rows', capacity);
    const count = output('count', 1),
      required = output('required', 1),
      overflow = output('overflow', 1);
    const lossReadback = output('loss-readback', 1);
    const contributor = new GPUSpatialPredicateJoin({
      left,
      right,
      predicate: nineMatches ? 'intersects' : containment,
      engine: containment === 'coveredBy' && !multipart ? 'auto' : 'fast',
      candidateCapacity,
      // Deliberately omit both caller loss destinations: the producer must attach its own.
      output: {
        pairs: {
          leftRows: leftRows.view,
          rightRows: rightRows.view,
          status: {count: count.view, requiredCount: required.view, overflow: overflow.view}
        }
      }
    });
    graph.add(contributor);
    if (nineMatches) {
      const loss = contributor.outputPairs?.status.candidateOverflow;
      expect(
        loss,
        'outputPairs must expose candidate loss without a caller loss view'
      ).toBeDefined();
      if (!loss) throw new Error('missing public candidate-loss view');
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'read-owned-loss',
          operation: 'geometry-join-boundary-probe',
          invocationCount: 1,
          bindings: [
            {name: 'loss', view: loss, type: 'u32', access: 'read'},
            {name: 'copied', view: lossReadback.view, type: 'u32', access: 'read_write'}
          ],
          body: 'copied[copiedOffset + index] = loss[lossOffset];'
        })
      );
    }
    const compiled = graph.compile();
    resources.push(compiled);
    submitGraph(device, compiled, undefined);
    const written = (await readUint32(count.buffer, 1))[0];
    return {
      candidateCapacity: contributor.candidateCapacity,
      count: written,
      required: (await readUint32(required.buffer, 1))[0],
      overflow: (await readUint32(overflow.buffer, 1))[0],
      candidateOverflow: nineMatches ? (await readUint32(lossReadback.buffer, 1))[0] : 0,
      leftRows: await readUint32(leftRows.buffer, written),
      rightRows: await readUint32(rightRows.buffer, written)
    };
  });
}

it('pairs-only join exposes the ninth missing match at default candidate capacity eight', async () => {
  // All nine literal points lie strictly inside [0,100] x [0,10]. Final capacity is nine.
  const truncated = await runBoundaryJoin('nine-matches');
  expect(truncated.candidateCapacity).toBe(8);
  expect(truncated.count).toBe(8);
  expect(truncated.required).toBe(8);
  expect(truncated.overflow).toBe(0); // Final destination fits: this flag alone cannot certify completeness.
  expect(truncated.candidateOverflow).toBe(1);
  expectTruncatedPairPrefix(truncated);
  const complete = await runBoundaryJoin('nine-matches', 9);
  expect(complete.count).toBe(9);
  expect(complete.required).toBe(9);
  expect(complete.overflow).toBe(0);
  expect(complete.candidateOverflow).toBe(0);
  expect(complete.leftRows).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);
  expect(complete.rightRows).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
});

it('fast line within excludes a degenerate endpoint and includes a degenerate interior', async () => {
  const result = await runBoundaryJoin('degenerate-within');
  expect(result.count).toBe(1);
  expect(result.required).toBe(1);
  expect(result.overflow).toBe(0);
  expect(result.leftRows).toEqual([1]);
  expect(result.rightRows).toEqual([0]);
});

it('line coveredBy includes both degenerate endpoint and interior controls', async () => {
  const result = await runBoundaryJoin('degenerate-within', undefined, 'coveredBy');
  expect(result.count).toBe(2);
  expect(result.leftRows).toEqual([0, 1]);
  expect(result.rightRows).toEqual([0, 0]);
});

it('S1 fast multipart line coveredBy includes endpoint and interior degenerate controls', async () => {
  const result = await runBoundaryJoin('degenerate-within', undefined, 'coveredBy', true);
  expect(result.count).toBe(2);
  expect(result.leftRows).toEqual([0, 1]);
  expect(result.rightRows).toEqual([0, 0]);
});
