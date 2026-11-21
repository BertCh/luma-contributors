// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {computeTrajectorySegmentsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation';
import {computeTrajectorySplitOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';
import {computeTrajectoryThinningOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {
  computeTrajectoryRecursionsOnCPU,
  CPUTrajectoryRecursionsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation';
import {
  computeModalSplitOnCPU,
  CPUModalSplitOperation
} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation';
import {CPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
for (const offsets of [[0], [0, 0]])
  it(`preserves ${offsets.length - 1} empty tracks across movement forms`, () => {
    const input = {
      positions: new Float32Array(),
      timestamps: new Float32Array(),
      trackOffsets: Uint32Array.from(offsets),
      parameters: Float32Array.of(1, 1, 0, 1)
    };
    const stays = computeTrajectoryStaypointsOnCPU(input, {capacity: 1});
    expect(stays.count[0]).toBe(0);
    const segments = computeTrajectorySegmentsOnCPU(
      {...input, staypointIds: new Uint32Array()},
      {triplegCapacity: 1}
    );
    expect(segments.triplegCount[0]).toBe(0);
    const split = computeTrajectorySplitOnCPU(input, {
      mode: 'gap',
      segmentCapacity: 1,
      rowCapacity: 1
    });
    expect(split.count[0]).toBe(0);
    const thinning = computeTrajectoryThinningOnCPU(input, {mode: 'minimum-distance', capacity: 1});
    expect(Array.from(thinning.outputTrackOffsets)).toEqual(offsets);
    const recursion = computeTrajectoryRecursionsOnCPU({...input, radii: Float32Array.of(1)});
    expect(recursion.revisits.length).toBe(0);
    const modal = computeModalSplitOnCPU(
      {
        ...input,
        triplegs: {
          tracks: new Uint32Array(),
          startRows: new Uint32Array(),
          endRows: new Uint32Array(),
          count: Uint32Array.of(0)
        }
      },
      {metric: 'count', modeCount: 2, modeSpeedUpperBoundsFloat32: [1, Infinity]}
    );
    expect(Array.from(modal.modalSplit)).toEqual(new Array((offsets.length - 1) * 2).fill(0));
    expect(
      () =>
        new CPUModalSplitOperation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'offsets',
          rowCount: 0,
          trackCount: offsets.length - 1,
          triplegCapacity: 0,
          modeCount: 2,
          metric: 'count',
          modeSpeedUpperBoundsFloat32: [1, Infinity],
          triplegs: {
            tracks: 'tracks',
            startRows: 'starts',
            endRows: 'ends',
            status: {count: 'count'}
          },
          output: {modalSplit: 'split'}
        })
    ).not.toThrow();
    expect(
      () =>
        new CPUTrajectoryThinningOperation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'offsets',
          parameters: 'parameters',
          rowCount: 0,
          trackCount: offsets.length - 1,
          capacity: 1,
          mode: 'minimum-distance',
          output: {
            sourceRows: 'rows',
            trackOffsets: 'newOffsets',
            status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
          }
        })
    ).not.toThrow();
    expect(
      () =>
        new CPUTrajectoryRecursionsOperation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'offsets',
          radii: 'radii',
          rowCount: 0,
          radiusCount: 1,
          output: {revisits: 'revisits'}
        })
    ).not.toThrow();
  });

import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {CPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {CPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation';
import {CPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';
import {computeBrownianBridgeUtilizationOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation';
import {computeTrackSimilarityOnCPU} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {GPU_TRACK_SIMILARITY_STATUS} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
function vector(
  format: 'float32' | 'float32x2' | 'uint32',
  values: Float32Array | Uint32Array
): GPUCPUVector {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}
for (const offsets of [[0], [0, 0]])
  it(`executes named empty movement forms for offsets ${offsets}`, async () => {
    const positions = vector('float32x2', new Float32Array());
    const timestamps = vector('float32', new Float32Array());
    const trackOffsets = vector('uint32', Uint32Array.from(offsets));
    const parameters = vector('float32', Float32Array.of(1, 1, 0, 1));
    const inputs = {positions, timestamps, trackOffsets, parameters};
    const stays = new CPUTrajectoryStaypointsOperation({
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      parameters: 's',
      rowCount: 0,
      capacity: 1,
      output: {
        ids: 'ids',
        status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
      }
    });
    expect((await stays.execute({inputs}))['count']).toEqual(Uint32Array.of(0));
    const segments = new CPUTrajectorySegmentsOperation({
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      parameters: 's',
      staypointIds: 'stayIds',
      rowCount: 0,
      triplegCapacity: 1,
      output: {triplegs: {ids: 'ids', status: {count: 'count', overflow: 'overflow'}}}
    });
    expect(
      (
        await segments.execute({
          inputs: {...inputs, staypointIds: vector('uint32', new Uint32Array())}
        })
      )['count']
    ).toEqual(Uint32Array.of(0));
    const split = new CPUTrajectorySplitOperation({
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      parameters: 's',
      rowCount: 0,
      segmentCapacity: 1,
      rowCapacity: 1,
      mode: 'gap',
      output: {
        segmentOffsets: 'newOffsets',
        segmentTracks: 'tracks',
        rowIds: 'rows',
        status: {count: 'count', overflow: 'overflow'}
      }
    });
    expect((await split.execute({inputs}))['newOffsets']).toEqual(Uint32Array.of(0, 0xffffffff));
    const thinning = new CPUTrajectoryThinningOperation({
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      parameters: 's',
      rowCount: 0,
      trackCount: offsets.length - 1,
      capacity: 1,
      mode: 'minimum-distance',
      output: {
        sourceRows: 'rows',
        trackOffsets: 'newOffsets',
        status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
      }
    });
    expect((await thinning.execute({inputs}))['newOffsets']).toEqual(Uint32Array.from(offsets));
    const recursion = new CPUTrajectoryRecursionsOperation({
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      radii: 'r',
      rowCount: 0,
      radiusCount: 1,
      output: {revisits: 'visits'}
    });
    expect(
      (
        await recursion.execute({inputs: {...inputs, radii: vector('float32', Float32Array.of(1))}})
      )['visits']
    ).toEqual(new Uint32Array());
    const modal = new CPUModalSplitOperation({
      positions: 'p',
      timestamps: 't',
      trackOffsets: 'o',
      rowCount: 0,
      trackCount: offsets.length - 1,
      triplegCapacity: 0,
      metric: 'count',
      modeCount: 2,
      modeSpeedUpperBoundsFloat32: [1, Infinity],
      triplegs: {tracks: 'tracks', startRows: 'starts', endRows: 'ends', status: {count: 'active'}},
      output: {modalSplit: 'split'}
    });
    expect(
      (
        await modal.execute({
          inputs: {
            ...inputs,
            triplegTracks: vector('uint32', new Uint32Array()),
            triplegStartRows: vector('uint32', new Uint32Array()),
            triplegEndRows: vector('uint32', new Uint32Array()),
            triplegCount: vector('uint32', Uint32Array.of(0))
          }
        })
      )['split']
    ).toEqual(new Float32Array((offsets.length - 1) * 2));
  });
it('checks bridge and similarity public empty-offset cases in the movement matrix', () => {
  for (const offsets of [[0], [0, 0]]) {
    const result = computeBrownianBridgeUtilizationOnCPU(
      {
        positions: new Float32Array(),
        timestamps: new Float32Array(),
        trackOffsets: Uint32Array.from(offsets),
        motionVariances: new Float32Array(),
        locationErrors: new Float32Array(),
        parameters: Float32Array.of(-1, -1, 1, 1)
      },
      {gridSize: [1, 1]}
    );
    expect(result.density).toEqual(Float32Array.of(0));
    const similarity = computeTrackSimilarityOnCPU({
      linesA: {positions: new Float32Array(), lineOffsets: Uint32Array.from(offsets)},
      pairA: Uint32Array.of(0),
      pairB: Uint32Array.of(0)
    });
    expect(similarity.pairStatus[0]).toBe(
      offsets.length === 1
        ? GPU_TRACK_SIMILARITY_STATUS.invalidPair
        : GPU_TRACK_SIMILARITY_STATUS.emptyTrack
    );
  }
});

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUTrajectoryStaypoints} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {GPUTrajectorySegments} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments';
import {GPUTrajectorySplit} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {GPUTrajectoryThinning} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {GPUTrajectoryRecursions} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions';
import {GPUModalSplit} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split';
for (const trackCount of [0, 1])
  it(`declares GPU movement graphs for ${trackCount} empty tracks`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `empty-${sequence++}`, format, length);
    const input = {
      positions: view('float32x2', 0),
      timestamps: view('float32', 0),
      trackOffsets: view('uint32', trackCount + 1),
      parameters: view('float32', 4)
    };
    const status = () => ({count: view('uint32', 1), overflow: view('uint32', 1)});
    const budgetedStatus = () => ({...status(), candidateOverflow: view('uint32', 1)});
    try {
      expect(() =>
        graph.add(
          new GPUTrajectoryStaypoints({
            ...input,
            output: {ids: view('uint32', 1), status: budgetedStatus()}
          })
        )
      ).not.toThrow();
      expect(() =>
        graph.add(
          new GPUTrajectorySegments({
            ...input,
            staypointIds: view('uint32', 0),
            output: {triplegs: {ids: view('uint32', 1), status: status()}}
          })
        )
      ).not.toThrow();
      expect(() =>
        graph.add(
          new GPUTrajectorySplit({
            ...input,
            mode: 'gap',
            output: {
              segmentOffsets: view('uint32', 2),
              segmentTracks: view('uint32', 1),
              rowIds: view('uint32', 1),
              status: status()
            }
          })
        )
      ).not.toThrow();
      expect(() =>
        graph.add(
          new GPUTrajectoryThinning({
            ...input,
            mode: 'minimum-distance',
            output: {
              sourceRows: view('uint32', 1),
              trackOffsets: view('uint32', trackCount + 1),
              status: budgetedStatus()
            }
          })
        )
      ).not.toThrow();
      expect(() =>
        graph.add(
          new GPUTrajectoryRecursions({
            ...input,
            radii: view('float32', 1),
            output: {revisits: view('uint32', 0)}
          })
        )
      ).not.toThrow();
      expect(() =>
        graph.add(
          new GPUModalSplit({
            ...input,
            triplegs: {
              tracks: view('uint32', 0),
              startRows: view('uint32', 0),
              endRows: view('uint32', 0),
              status: {count: view('uint32', 1)}
            },
            modeCount: 2,
            metric: 'count',
            modeSpeedUpperBoundsFloat32: [1, Infinity],
            output: {modalSplit: view('float32', trackCount * 2)}
          })
        )
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });
it('B6 named CPU adapters clamp resident terminal offsets without weakening pure compact inputs', async () => {
  const inputs = {
    positions: vector('float32x2', new Float32Array(4)),
    timestamps: vector('float32', Float32Array.of(0, 1)),
    trackOffsets: vector('uint32', Uint32Array.of(0, 3)),
    parameters: vector('float32', Float32Array.of(1, 1, 0, 1))
  };
  const shared = {positions: 'p', timestamps: 't', trackOffsets: 'o', parameters: 's', rowCount: 2};
  const stays = new CPUTrajectoryStaypointsOperation({
    ...shared,
    capacity: 1,
    output: {
      ids: 'ids',
      endRows: 'ends',
      status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
    }
  });
  expect((await stays.execute({inputs}))['ends']).toEqual(Uint32Array.of(2));
  const segments = new CPUTrajectorySegmentsOperation({
    ...shared,
    staypointIds: 'stayIds',
    triplegCapacity: 1,
    output: {triplegs: {ids: 'ids', status: {count: 'count', overflow: 'overflow'}}}
  });
  expect(
    (
      await segments.execute({
        inputs: {...inputs, staypointIds: vector('uint32', new Uint32Array(2).fill(0xffffffff))}
      })
    )['count']
  ).toEqual(Uint32Array.of(0));
  const split = new CPUTrajectorySplitOperation({
    ...shared,
    mode: 'gap',
    segmentCapacity: 1,
    rowCapacity: 2,
    output: {
      segmentOffsets: 'offsets',
      segmentTracks: 'tracks',
      rowIds: 'rows',
      status: {count: 'count', overflow: 'overflow'}
    }
  });
  expect((await split.execute({inputs}))['offsets']).toEqual(Uint32Array.of(0, 2));
  const thinning = new CPUTrajectoryThinningOperation({
    ...shared,
    mode: 'minimum-distance',
    trackCount: 1,
    capacity: 2,
    output: {
      sourceRows: 'rows',
      trackOffsets: 'offsets',
      status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
    }
  });
  expect((await thinning.execute({inputs}))['rows']).toEqual(Uint32Array.of(0, 1));
  const recursion = new CPUTrajectoryRecursionsOperation({
    ...shared,
    radii: 'r',
    radiusCount: 1,
    output: {residenceTimes: 'residence'}
  });
  expect(
    (await recursion.execute({inputs: {...inputs, radii: vector('float32', Float32Array.of(1))}}))[
      'residence'
    ]
  ).toEqual(Float32Array.of(1, 1));
  const modal = new CPUModalSplitOperation({
    ...shared,
    trackCount: 1,
    triplegCapacity: 1,
    metric: 'count',
    modeCount: 1,
    modes: 'm',
    triplegs: {tracks: 'tr', startRows: 'st', endRows: 'en', status: {count: 'n'}},
    output: {modalSplit: 'split'}
  });
  expect(
    (
      await modal.execute({
        inputs: {
          ...inputs,
          modes: vector('uint32', Uint32Array.of(0)),
          triplegTracks: vector('uint32', Uint32Array.of(0)),
          triplegStartRows: vector('uint32', Uint32Array.of(0)),
          triplegEndRows: vector('uint32', Uint32Array.of(2)),
          triplegCount: vector('uint32', Uint32Array.of(1))
        }
      })
    )['split']
  ).toEqual(Float32Array.of(1));
  expect(() =>
    computeTrajectoryStaypointsOnCPU(
      {
        positions: new Float32Array(4),
        timestamps: Float32Array.of(0, 1),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(1, 1, 0, 1)
      },
      {capacity: 1}
    )
  ).toThrow(/trackOffsets/);
});
