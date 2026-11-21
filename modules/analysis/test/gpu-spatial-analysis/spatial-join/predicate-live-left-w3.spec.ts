// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUPairGather, GPU_PAIR_GATHER_NO_ROW} from '../../../src/gpu-spatial-analysis/pair-gather';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {withTestResourceCleanup} from '../geometry-refinement/geometry-output-review-test-utils';

it('UGE-02 bounds anti rows by the live prefix on reuse and retains a live empty feature', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources: {destroy(): void}[] = [];
  await withTestResourceCleanup(resources, async () => {
    const graph = new GPUCommandGraph(device);
    const input = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      resources.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(
          graph,
          name,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        )
      };
    };
    const output = (name: string, length = 1) => {
      const buffer = createOutputBuffer(device, length);
      resources.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const live = input('live', 'uint32', Uint32Array.of(3));
    const upstreamLoss = input('upstream-loss', 'uint32', Uint32Array.of(0));
    const featureOffsets = input('features', 'uint32', Uint32Array.of(0, 1, 2, 3));
    const polygonOffsets = input('polygons', 'uint32', Uint32Array.of(0, 1, 2, 3));
    const ringOffsets = input('rings', 'uint32', Uint32Array.of(0, 4, 8, 12));
    const count = output('count'),
      overflow = output('overflow'),
      candidateOverflow = output('candidate-overflow'),
      ids = output('ids', 3);
    graph.add(
      new GPUSpatialPredicateJoin({
        left: {
          kind: 'polygons',
          positions: input(
            'squares',
            'float32x2',
            Float32Array.of(
              0,
              0,
              2,
              0,
              2,
              2,
              0,
              2,
              4,
              0,
              6,
              0,
              6,
              2,
              4,
              2,
              8,
              0,
              10,
              0,
              10,
              2,
              8,
              2
            )
          ).view,
          featureOffsets: featureOffsets.view,
          polygonOffsets: polygonOffsets.view,
          ringOffsets: ringOffsets.view
        },
        leftFeatureCount: live.view,
        leftStatus: {overflow: upstreamLoss.view},
        right: {kind: 'points', positions: input('point', 'float32x2', Float32Array.of(1, 1)).view},
        predicate: 'intersects',
        how: 'anti',
        output: {
          unmatched: {
            ids: ids.view,
            status: {
              count: count.view,
              overflow: overflow.view,
              candidateOverflow: candidateOverflow.view
            }
          }
        }
      })
    );
    const compiled = graph.compile();
    resources.push(compiled);
    const run = async (liveCount: number, loss = 0) => {
      live.buffer.write(Uint32Array.of(liveCount));
      upstreamLoss.buffer.write(Uint32Array.of(loss));
      submitGraph(device, compiled, undefined);
      const written = (await readUint32(count.buffer, 1))[0];
      return {
        count: written,
        ids: await readUint32(ids.buffer, written),
        candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0]
      };
    };
    expect(await run(3)).toEqual({count: 2, ids: [1, 2], candidateOverflow: 0});
    // Shrinking ring assembly repeats every terminal offset; old position tails remain resident.
    featureOffsets.buffer.write(Uint32Array.of(0, 1, 1, 1));
    polygonOffsets.buffer.write(Uint32Array.of(0, 1, 1, 1));
    ringOffsets.buffer.write(Uint32Array.of(0, 4, 4, 4));
    expect(await run(1)).toEqual({count: 0, ids: [], candidateOverflow: 0});
    expect(await run(2)).toEqual({count: 1, ids: [1], candidateOverflow: 0});
    expect(await run(2, 1)).toEqual({count: 0, ids: [], candidateOverflow: 1});
    expect(await run(2)).toEqual({count: 1, ids: [1], candidateOverflow: 0});
  });
});

for (const pairCapacity of [1, 2])
  it(`UGE-12 UGE-13 gathers interior, boundary and unmatched points from one inner join at pair capacity ${pairCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const resources: {destroy(): void}[] = [];
    await withTestResourceCleanup(resources, async () => {
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
      const output = (name: string, length = 1) => {
        const buffer = createOutputBuffer(device, length);
        resources.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
      };
      const pairCount = output('pair-count'),
        pairOverflow = output('pair-overflow');
      const unmatched = {
        ids: output('unmatched-ids', 3).view,
        status: {count: output('unmatched-count').view, overflow: output('unmatched-overflow').view}
      };
      const join = new GPUSpatialPredicateJoin({
        left: {
          kind: 'points',
          positions: input('points', 'float32x2', Float32Array.of(1, 1, 0, 1, 3, 1))
        },
        right: {
          kind: 'polygons',
          positions: input('square', 'float32x2', Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2)),
          featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
          polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
          ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4))
        },
        predicate: 'intersects',
        candidateCapacity: 3,
        output: {
          pairs: {
            leftRows: output('pair-left', pairCapacity).view,
            rightRows: output('pair-right', pairCapacity).view,
            status: {count: pairCount.view, overflow: pairOverflow.view}
          },
          unmatched
        }
      });
      graph.add(join);
      const count = output('gather-count'),
        left = output('gather-left', 3),
        right = output('gather-right', 3),
        values = output('gather-values', 3);
      graph.add(
        new GPUPairGather({
          how: 'left',
          pairs: join.outputPairs!,
          unmatchedLeft: unmatched,
          rightColumns: [
            {source: input('attribute', 'uint32', Uint32Array.of(7)), output: values.view}
          ],
          output: {
            leftRows: left.view,
            rightRows: right.view,
            status: {
              count: count.view,
              overflow: output('gather-overflow').view,
              candidateOverflow: output('gather-loss').view
            }
          }
        })
      );
      const compiled = graph.compile();
      resources.push(compiled);
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count.buffer, 1)).toEqual([pairCapacity + 1]);
      expect(await readUint32(left.buffer, pairCapacity + 1)).toEqual(
        pairCapacity === 2 ? [0, 1, 2] : [0, 2]
      );
      expect(await readUint32(right.buffer, pairCapacity + 1)).toEqual(
        pairCapacity === 2 ? [0, 0, GPU_PAIR_GATHER_NO_ROW] : [0, GPU_PAIR_GATHER_NO_ROW]
      );
      expect(await readUint32(values.buffer, pairCapacity + 1)).toEqual(
        pairCapacity === 2 ? [7, 7, GPU_PAIR_GATHER_NO_ROW] : [7, GPU_PAIR_GATHER_NO_ROW]
      );
    });
  });

for (const usePrepared of [false, true]) {
  for (const lossKind of ['bvh', 'candidate'])
    it(`GEO-103 exposes ${usePrepared ? 'prepared' : 'private'} ${lossKind} loss on reuse`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources: {destroy(): void}[] = [];
      await withTestResourceCleanup(resources, async () => {
        const graph = new GPUCommandGraph(device);
        const input = (name: string, values: Float32Array) => {
          const buffer = createInputBuffer(device, values);
          resources.push(buffer);
          return {
            buffer,
            view: importGraphBuffer(graph, name, buffer, 'float32x2', values.length / 2)
          };
        };
        const output = (name: string, length = 1) => {
          const buffer = createOutputBuffer(device, length);
          resources.push(buffer);
          return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
        };
        const left = input('left', Float32Array.of(0, 0));
        const right = {
          kind: 'points' as const,
          positions: input('right', Float32Array.of(0, 0, 0, 0)).view
        };
        const leafCapacity = lossKind === 'bvh' ? 1 : 2;
        const prepared = usePrepared
          ? new GPUSpatialJoinPrepared({geometry: right, leafCapacity, spatialSort: false})
          : undefined;
        if (prepared) {
          resources.push(prepared);
          graph.add(prepared);
        }
        const count = output('count'),
          overflow = output('overflow'),
          loss = output('loss'),
          aggregateLoss = output('aggregate-loss');
        const join = new GPUSpatialPredicateJoin({
          left: {kind: 'points', positions: left.view},
          right,
          prepared,
          leafCapacity,
          candidateCapacity: lossKind === 'candidate' ? 1 : 2,
          predicate: 'intersects',
          output: {
            status: {candidateOverflow: aggregateLoss.view},
            pairs: {
              leftRows: output('left-rows', 2).view,
              rightRows: output('right-rows', 2).view,
              status: {count: count.view, overflow: overflow.view}
            }
          }
        });
        graph.add(join);
        graph.add(
          createWGSLKernelNode(graph, {
            id: 'copy-loss',
            operation: 'join-loss-probe',
            invocationCount: 1,
            bindings: [
              {
                name: 'source',
                view: join.outputPairs!.status.candidateOverflow,
                type: 'u32',
                access: 'read'
              },
              {name: 'destination', view: loss.view, type: 'u32', access: 'read_write'}
            ],
            body: 'destination[destinationOffset] = source[sourceOffset];'
          })
        );
        const compiled = graph.compile();
        resources.push(compiled);
        submitGraph(device, compiled, undefined);
        expect(await readUint32(loss.buffer, 1)).toEqual([1]);
        expect(await readUint32(aggregateLoss.buffer, 1)).toEqual([1]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        // BVH loss is structural and must stay set even when no retained point matches on reuse.
        left.buffer.write(Float32Array.of(10, 10));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(count.buffer, 1)).toEqual([0]);
        expect(await readUint32(loss.buffer, 1)).toEqual([lossKind === 'bvh' ? 1 : 0]);
        expect(await readUint32(aggregateLoss.buffer, 1)).toEqual([lossKind === 'bvh' ? 1 : 0]);
      });
    });
}
