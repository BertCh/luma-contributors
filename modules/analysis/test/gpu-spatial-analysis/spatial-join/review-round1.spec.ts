// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {addSpatialPredicateUnmatchedNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-unmatched';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {withTestResourceCleanup} from '../geometry-refinement/geometry-output-review-test-utils';

it('B4 shrinking live polygons excludes padded loss rows and stale crossing rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources: {destroy(): void}[] = [];
  await withTestResourceCleanup(resources, async () => {
    const graph = new GPUCommandGraph(device);
    const input = <Format extends 'uint32' | 'float32x2'>(
      id: string,
      format: Format,
      values: Uint32Array | Float32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      resources.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(
          graph,
          id,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        )
      };
    };
    const output = (id: string, length = 1) => {
      const buffer = createOutputBuffer(device, length);
      resources.push(buffer);
      return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', length)};
    };
    const live = input('live', 'uint32', Uint32Array.of(3));
    const flags = output('truncated', 3),
      truncatedCount = output('truncated-count'),
      geographic = output('geographic');
    graph.add(
      new GPUSpatialPredicateJoin({
        left: {
          kind: 'polygons',
          positions: input(
            'positions',
            'float32x2',
            Float32Array.of(
              0,
              0,
              1,
              0,
              1,
              1,
              0,
              1,
              179,
              0,
              -179,
              0,
              -179,
              1,
              179,
              1,
              10,
              0,
              11,
              0,
              11,
              1,
              10,
              1
            )
          ).view,
          featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1, 2, 3)).view,
          polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1, 2, 3)).view,
          ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4, 8, 12)).view
        },
        leftFeatureCount: live.view,
        leftStatus: {overflow: input('input-loss', 'uint32', Uint32Array.of(1)).view},
        right: {
          kind: 'points',
          positions: input('point', 'float32x2', Float32Array.of(0.5, 0.5)).view
        },
        predicate: 'intersects',
        how: 'anti',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
        output: {
          candidateTruncated: flags.view,
          unmatched: {
            ids: output('unmatched', 3).view,
            status: {
              count: output('count').view,
              overflow: output('overflow').view,
              candidateOverflow: output('candidate-overflow').view
            }
          },
          status: {
            candidateTruncatedCount: truncatedCount.view,
            antimeridianUncertainCount: geographic.view
          }
        }
      })
    );
    const compiled = graph.compile();
    resources.push(compiled);
    for (const liveCount of [3, 1, 3]) {
      live.buffer.write(Uint32Array.of(liveCount));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(flags.buffer, 3)).toEqual(liveCount === 3 ? [1, 1, 1] : [1, 0, 0]);
      expect(await readUint32(truncatedCount.buffer, 1)).toEqual([liveCount]);
      expect(await readUint32(geographic.buffer, 1)).toEqual([liveCount === 3 ? 1 : 0]);
    }
  });
});

it('B7 retry consumes fresh graph producer writes through one canonical imported handle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources: {destroy(): void}[] = [];
  await withTestResourceCleanup(resources, async () => {
    const buffer = createInputBuffer(device, Float32Array.of(0, 0));
    resources.push(buffer);
    const original = new GPUCommandGraph(device);
    const prepared = new GPUSpatialJoinPrepared({
      id: 'retry',
      geometry: {
        kind: 'points',
        positions: importGraphBuffer(original, 'old-positions', buffer, 'float32x2', 1)
      }
    });
    resources.push(prepared);
    const allocate = device.createBuffer.bind(device);
    const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
      if (settings && 'id' in settings && settings.id === 'retry-node-minima')
        throw new Error('injected failure');
      return allocate(settings);
    });
    try {
      expect(() => original.add(prepared)).toThrow('injected failure');
    } finally {
      spy.mockRestore();
    }
    const graph = new GPUCommandGraph(device);
    const positions = importGraphBuffer(graph, 'produced-positions', buffer, 'float32x2', 1);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'producer',
        operation: 'RetryProbe',
        invocationCount: 1,
        bindings: [{name: 'positions', view: positions, type: 'f32', access: 'read_write'}],
        body: 'positions[positionsOffset] = 5.0; positions[positionsOffset + 1u] = 0.0;'
      })
    );
    graph.add(prepared);
    const left = createInputBuffer(device, Float32Array.of(5, 0));
    resources.push(left);
    const output = (id: string) => {
      const buffer = createOutputBuffer(device, 1);
      resources.push(buffer);
      return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', 1)};
    };
    const count = output('count');
    graph.add(
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: importGraphBuffer(graph, 'left', left, 'float32x2', 1)},
        right: {kind: 'points', positions},
        prepared,
        predicate: 'intersects',
        output: {
          pairs: {
            leftRows: output('left-rows').view,
            rightRows: output('right-rows').view,
            status: {count: count.view, overflow: output('overflow').view}
          }
        }
      })
    );
    const compiled = graph.compile();
    resources.push(compiled);
    submitGraph(device, compiled, undefined);
    expect(await readUint32(count.buffer, 1)).toEqual([1]);
  });
});

it('S2 unmatched publication withholds uncertified absence and clears on certified reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources: {destroy(): void}[] = [];
  await withTestResourceCleanup(resources, async () => {
    const graph = new GPUCommandGraph(device);
    const flagBuffer = createInputBuffer(device, Uint32Array.of(1));
    resources.push(flagBuffer);
    const uncertainty = importGraphBuffer(graph, 'uncertainty', flagBuffer, 'uint32', 1);
    const state = createTransientView(graph, 'state', 'uint32', 6),
      matches = createTransientView(graph, 'matches', 'uint32', 1);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'classification-state',
        operation: 'UncertaintyProbe',
        invocationCount: 6,
        bindings: [
          {name: 'uncertainty', view: uncertainty, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'matches', view: matches, type: 'u32', access: 'read_write'}
        ],
        body: 'state[stateOffset + index] = select(0u, uncertainty[uncertaintyOffset], index == 1u); if (index == 0u) { matches[matchesOffset] = 0u; }'
      })
    );
    const countBuffer = createOutputBuffer(device, 1);
    resources.push(countBuffer);
    const count = importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1);
    const nodes: GPUCommandNode[] = [];
    addSpatialPredicateUnmatchedNodes(graph, nodes, {
      id: 'uncertain',
      leftCount: 1,
      candidateCapacity: 1,
      finishOnly: true,
      state,
      leftMatchCounts: matches,
      flags: createTransientView(graph, 'flags', 'uint32', 1),
      candidatePairs: createTransientView(graph, 'candidates', 'uint32x2', 1),
      unmatched: {
        ids: createTransientView(graph, 'ids', 'uint32', 1),
        status: {count, overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    });
    for (const node of nodes) graph.add(node);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'read-count',
        operation: 'UncertaintyProbe',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read_write'}
        ],
        body: 'count[countOffset] = state[stateOffset + 3u];'
      })
    );
    const compiled = graph.compile();
    resources.push(compiled);
    submitGraph(device, compiled, undefined);
    expect(await readUint32(countBuffer, 1)).toEqual([0]);
    flagBuffer.write(Uint32Array.of(0));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(countBuffer, 1)).toEqual([1]);
  });
});
