// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createTrackTableOffsetsNode} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/movement-common';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes 65537 track boundaries with gaps and refreshes a smaller active count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const trackCount = 65537;
  const groupCount = 21846;
  const tracks = createInputBuffer(
    device,
    Uint32Array.from({length: groupCount * 2}, (_, row) => Math.floor(row / 2) * 3)
  );
  const count = createInputBuffer(device, new Uint32Array([groupCount * 2]));
  const offsets = createOutputBuffer(device, trackCount + 1);
  const node = createTrackTableOffsetsNode(graph, {
    id: 'offsets',
    operation: 'TestTrackOffsets',
    tracks: importGraphBuffer(graph, 'tracks', tracks, 'uint32'),
    count: importGraphBuffer(graph, 'count', count, 'uint32'),
    offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32'),
    capacity: groupCount * 2,
    trackCount
  });
  graph.add({id: 'track-offsets', getCommandNodes: () => [node]});
  const compiled = graph.compile();
  try {
    for (const total of [groupCount * 2, 2, 0]) {
      count.write(new Uint32Array([total]));
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(offsets, trackCount + 1))).toEqual(
        Array.from({length: trackCount + 1}, (_, boundary) =>
          Math.min(Math.ceil(boundary / 3) * 2, total)
        )
      );
    }
  } finally {
    compiled.destroy();
    tracks.destroy();
    count.destroy();
    offsets.destroy();
  }
}, 120000);

import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {GPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation';
import {GPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {GPUTrajectoryRecursionsOperation} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';
import {GPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import {GPUBrownianBridgeUtilizationOperation} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization-operation';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {GPU_TRACK_SIMILARITY_STATUS} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {expectSplitRowPrefix} from './movement-output-assertions.test-utils';
it('preserves zero tracks and one empty track through the GPU movement offset matrix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const offsets of [[0], [0, 0]]) {
    const shared = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'offsets',
      parameters: 'parameters',
      rowCount: 0
    };
    const status = {count: 'count', overflow: 'overflow'};
    const budgetedStatus = {...status, candidateOverflow: 'candidateOverflow'};
    const operations = [
      new GPUTrajectoryStaypointsOperation({
        ...shared,
        capacity: 1,
        output: {ids: 'ids', status: budgetedStatus}
      }),
      new GPUTrajectorySegmentsOperation({
        ...shared,
        staypointIds: 'stayIds',
        triplegCapacity: 1,
        output: {triplegs: {ids: 'ids', status}}
      }),
      new GPUTrajectorySplitOperation({
        ...shared,
        mode: 'gap',
        segmentCapacity: 1,
        rowCapacity: 1,
        output: {segmentOffsets: 'newOffsets', segmentTracks: 'tracks', rowIds: 'rows', status}
      }),
      new GPUTrajectoryThinningOperation({
        ...shared,
        mode: 'minimum-distance',
        trackCount: offsets.length - 1,
        capacity: 1,
        output: {sourceRows: 'rows', trackOffsets: 'newOffsets', status: budgetedStatus}
      }),
      new GPUTrajectoryRecursionsOperation({
        ...shared,
        radii: 'radii',
        radiusCount: 1,
        output: {revisits: 'revisits'}
      }),
      new GPUModalSplitOperation({
        ...shared,
        trackCount: offsets.length - 1,
        triplegCapacity: 1,
        triplegs: {
          tracks: 'tripTracks',
          startRows: 'starts',
          endRows: 'ends',
          status: {count: 'active'}
        },
        metric: 'count',
        modeCount: 2,
        modeSpeedUpperBoundsFloat32: [1, Infinity],
        output: {modalSplit: 'split'}
      }),
      new GPUBrownianBridgeUtilizationOperation({
        ...shared,
        parameters: 'bridgeParameters',
        motionVariances: 'variances',
        locationErrors: 'errors',
        gridSize: [1, 1],
        output: {density: 'density'}
      }),
      new GPUTrackSimilarityOperation({
        linesA: {positions: 'positions', lineOffsets: 'offsets'},
        pairA: 'pairA',
        pairB: 'pairB',
        pairCount: 1,
        output: {pairStatus: 'pairStatus', hausdorff: 'hausdorff'}
      })
    ];
    const expectedOutputs: Record<string, number[]>[] = [
      {ids: [0xffffffff], count: [0], overflow: [0], candidateOverflow: [0]},
      {ids: [0xffffffff], count: [0], overflow: [0]},
      {
        newOffsets: [0, 0xffffffff],
        tracks: [0xffffffff],
        // Zero segments own zero rows. The allocated rowIds tail has no sentinel guarantee.
        rows: [],
        count: [0],
        overflow: [0]
      },
      {rows: [0xffffffff], newOffsets: offsets, count: [0], overflow: [0], candidateOverflow: [0]},
      {revisits: []},
      {split: new Array((offsets.length - 1) * 2).fill(0)},
      {density: [0]},
      {
        pairStatus: [
          offsets.length === 1
            ? GPU_TRACK_SIMILARITY_STATUS.invalidPair
            : GPU_TRACK_SIMILARITY_STATUS.emptyTrack
        ],
        hausdorff: [NaN]
      }
    ];
    for (const [index, operation] of operations.entries()) {
      const expected = expectedOutputs[index];
      const outputs = Object.keys(expected);
      expect(
        Object.values(operation.outputs)
          .map(output => output.value)
          .sort()
      ).toEqual([...outputs].sort());
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array()},
          timestamps: {format: 'float32', values: new Float32Array()},
          offsets: {format: 'uint32', values: Uint32Array.from(offsets)},
          parameters: {format: 'float32', values: Float32Array.of(1, 1, 0, 1)},
          stayIds: {format: 'uint32', values: new Uint32Array()},
          radii: {format: 'float32', values: Float32Array.of(1)},
          tripTracks: {format: 'uint32', values: Uint32Array.of(0)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          ends: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(0)},
          variances: {format: 'float32', values: new Float32Array()},
          errors: {format: 'float32', values: new Float32Array()},
          bridgeParameters: {format: 'float32', values: Float32Array.of(-1, -1, 1, 1)},
          pairA: {format: 'uint32', values: Uint32Array.of(0)},
          pairB: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [operation],
        outputs
      });
      expect(Object.keys(result).sort()).toEqual([...outputs].sort());
      for (const [name, values] of Object.entries(expected)) {
        if (operation instanceof GPUTrajectorySplitOperation && name === 'rows') {
          expectSplitRowPrefix(result[name], result['newOffsets'], result['count'][0], 1, values);
        } else {
          expect(
            Array.from(result[name]),
            `${operation.id} offsets=${offsets} output=${name}`
          ).toEqual(values);
        }
      }
    }
  }
});
