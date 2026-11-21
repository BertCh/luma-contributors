// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialJoinCandidates} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {GPUSegmentIntersection} from '../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {GPUPairGather} from '../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

/** Real BVH loss, final pair loss, and clean reuse, including downstream status propagation. */
export async function checkCompactPairCauses(
  device: Device,
  kind: 'candidates' | 'segments',
  loss: 'leaves' | 'pairs',
  adjacentFlags = false
): Promise<void> {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
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
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const left = input(
    'left',
    'float32x2',
    Float32Array.from(kind === 'candidates' ? [0, 0] : [0, 0, 2, 0])
  );
  const right = input(
    'right',
    'float32x2',
    Float32Array.from(kind === 'candidates' ? [0, 0, 0, 0] : [1, -1, 1, 1, 1, -1, 1, 1])
  );
  const leftOffsets = input('left-offsets', 'uint32', Uint32Array.of(0, 2));
  const rightOffsets = input('right-offsets', 'uint32', Uint32Array.of(0, 2, 4));
  const capacity = loss === 'leaves' ? 4 : 1;
  const count = output('count');
  const overflow = output('overflow', adjacentFlags ? 2 : 1);
  const candidateOverflow = adjacentFlags
    ? {
        buffer: overflow.buffer,
        view: graph.createDataView(overflow.view.buffer, {
          format: 'uint32',
          byteOffset: 4,
          length: 1
        })
      }
    : output('candidate-overflow');
  const finalOverflow = adjacentFlags
    ? graph.createDataView(overflow.view.buffer, {format: 'uint32', length: 1})
    : overflow.view;
  const requiredCount = output('required-count');
  const pairs = {
    leftRows: output('left-rows', capacity).view,
    rightRows: output('right-rows', capacity).view,
    status: {
      count: count.view,
      overflow: finalOverflow,
      candidateOverflow: candidateOverflow.view,
      requiredCount: requiredCount.view
    }
  };
  const leafCapacity = loss === 'leaves' ? 1 : 4;
  const gatheredCount = output('gathered-count');
  const gatheredOverflow = output('gathered-overflow');
  const gatheredCandidateOverflow = output('gathered-candidate-overflow');
  const gatheredLeft = output('gathered-left', 4);
  const gatheredRight = output('gathered-right', 4);
  const compile = (targetGraph: GPUCommandGraph, targetLeafCapacity: number) => {
    const handles = new Map<GraphDataView['buffer'], GraphDataView['buffer']>();
    const bind = <Format extends 'uint32' | 'float32x2'>(view: GraphDataView<Format>) => {
      if (view.buffer.graph === targetGraph) return view;
      const buffer = view.buffer.defaultBuffer;
      if (!(buffer instanceof Buffer)) throw new Error('Fixture needs caller-owned buffers');
      let handle = handles.get(view.buffer);
      if (!handle) {
        handle = importGraphBuffer(
          targetGraph,
          view.buffer.id,
          buffer,
          'uint32',
          buffer.byteLength / 4
        ).buffer;
        handles.set(view.buffer, handle);
      }
      return targetGraph.createDataView(handle, {
        format: view.format,
        byteOffset: view.byteOffset,
        length: view.length
      });
    };
    const targetPairs = {
      leftRows: bind(pairs.leftRows),
      rightRows: bind(pairs.rightRows),
      status: {
        count: bind(pairs.status.count),
        overflow: bind(pairs.status.overflow),
        candidateOverflow: bind(pairs.status.candidateOverflow),
        requiredCount: bind(pairs.status.requiredCount)
      }
    };
    const callerPairs = {...targetPairs, status: {...targetPairs.status}};
    const producer =
      kind === 'candidates'
        ? new GPUSpatialJoinCandidates({
            left: {kind: 'points', positions: bind(left.view)},
            right: {kind: 'points', positions: bind(right.view)},
            leafCapacity: targetLeafCapacity,
            output: callerPairs
          })
        : new GPUSegmentIntersection({
            left: {kind: 'lines', positions: bind(left.view), lineOffsets: bind(leftOffsets.view)},
            right: {
              kind: 'lines',
              positions: bind(right.view),
              lineOffsets: bind(rightOffsets.view)
            },
            leafCapacity: targetLeafCapacity,
            spatialSort: false,
            output: {pairs: callerPairs}
          });
    Reflect.deleteProperty(callerPairs.status, 'candidateOverflow');
    targetGraph.add(producer);
    targetGraph.add(
      new GPUPairGather({
        pairs: targetPairs,
        output: {
          leftRows: bind(gatheredLeft.view),
          rightRows: bind(gatheredRight.view),
          status: {
            count: bind(gatheredCount.view),
            overflow: bind(gatheredOverflow.view),
            candidateOverflow: bind(gatheredCandidateOverflow.view)
          }
        }
      })
    );
    return targetGraph.compile();
  };
  const compiled = compile(graph, leafCapacity);
  const compilations = [compiled];
  try {
    const phases =
      loss === 'leaves'
        ? (['loss', 'invalidated', 'clean'] as const)
        : (['loss', 'clean'] as const);
    for (const phase of phases) {
      if (phase !== 'loss') {
        if (kind === 'candidates') right.buffer.write(Float32Array.of(0, 0, NaN, NaN));
        else rightOffsets.buffer.write(Uint32Array.of(0, 2, 2));
      }
      // Invalid bounds/empty lines do not shrink the published BVH's source view. Its
      // overflow is SOURCE_COUNT > LEAF_CAPACITY, so a clean leaf-loss phase needs new
      // topology. Keep every caller-owned input/output buffer, including the gather flags.
      const active =
        phase === 'clean' && loss === 'leaves' ? compile(new GPUCommandGraph(device), 4) : compiled;
      if (active !== compiled) compilations.push(active);
      candidateOverflow.buffer.write(Uint32Array.of(37), adjacentFlags ? 4 : 0);
      gatheredCandidateOverflow.buffer.write(Uint32Array.of(77));
      submitGraph(device, active, undefined);
      const expected = phase === 'clean' ? [0, 0] : loss === 'leaves' ? [0, 1] : [1, 0];
      expect(
        [
          (await readUint32(overflow.buffer, 1))[0],
          (await readUint32(candidateOverflow.buffer, adjacentFlags ? 2 : 1))[adjacentFlags ? 1 : 0]
        ],
        `${kind} ${loss} phase=${phase}`
      ).toEqual(expected);
      expect([
        (await readUint32(gatheredOverflow.buffer, 1))[0],
        (await readUint32(gatheredCandidateOverflow.buffer, 1))[0]
      ]).toEqual(expected);
      expect(await readUint32(count.buffer, 1)).toEqual([1]);
      expect(await readUint32(gatheredCount.buffer, 1)).toEqual([1]);
      expect(await readUint32(requiredCount.buffer, 1)).toEqual([
        phase === 'loss' && loss === 'pairs' ? 2 : 1
      ]);
    }
  } finally {
    for (const compilation of compilations) compilation.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
