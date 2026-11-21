// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addStopEventGathers} from './stop-event-gathers';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUMobilityIndicators} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators';
import {GPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation';
import {GPUTrajectoryRecursionsOperation} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  computeTrajectoryStaypointsOnCPU,
  CPUTrajectoryStaypointsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {GPUTrajectoryStaypointsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation';
import {GPUTrajectoryLocationsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation';
import {GPUTrajectoryStaypoints} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
function vector(
  format: 'uint32' | 'float32' | 'float32x2',
  values: Uint32Array | Float32Array
): GPUCPUVector {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}
it('B1 constructs default GPU mobility requesting only jumps', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  try {
    expect(
      () =>
        new GPUMobilityIndicators({
          trackCount: 1,
          timestamps: view('float32'),
          staypoints: {
            tracks: view('uint32'),
            centroids: view('float32x2'),
            startRows: view('uint32'),
            durations: view('float32'),
            locationIds: view('uint32'),
            status: {count: view('uint32')}
          },
          output: {jumpLengths: view('float32')}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
it('B2 accepts whole producer status and observes inherited loss getters', () => {
  const input = {
    centroids: Float32Array.of(0, 0, 10, 0),
    tracks: Uint32Array.of(0, 1),
    count: Uint32Array.of(2),
    parameters: Float32Array.of(1, 0, 0, 0)
  };
  const wholeStatus = {
    count: Uint32Array.of(2),
    requiredCount: Uint32Array.of(2),
    overflow: Uint32Array.of(0),
    candidateOverflow: Uint32Array.of(0)
  };
  expect(
    computeTrajectoryLocationsOnCPU({...input, staypointStatus: wholeStatus}).locationCount
  ).toEqual(Uint32Array.of(2));
  class Status {
    get overflow() {
      return Uint32Array.of(1);
    }
  }
  expect(
    computeTrajectoryLocationsOnCPU({...input, staypointStatus: new Status()}).complete
  ).toEqual(Uint32Array.of(0));
  for (const overflow of [new Uint32Array(), Uint32Array.of(0, 0)])
    expect(() => computeTrajectoryLocationsOnCPU({...input, staypointStatus: {overflow}})).toThrow(
      /overflow/
    );
});
it('B3 declares the actual end-to-end gather stages with unique IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const timestamps = createTransientView(graph, 'timestamps', 'uint32x2', 3),
    fixOnes = createTransientView(graph, 'fix-ones', 'uint32', 3),
    starts = createTransientView(graph, 'starts', 'uint32', 3),
    eventTimes = createTransientView(graph, 'event-times', 'uint32x2', 3),
    live = createTransientView(graph, 'live', 'uint32', 3);
  try {
    expect(() =>
      addStopEventGathers(graph, timestamps, fixOnes, starts, eventTimes, live)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
it('B7 counts missing fixes only inside resident track ranges', () => {
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: new Float32Array(10),
      timestamps: Float32Array.of(0, 1, 0, 0, 0),
      trackOffsets: Uint32Array.of(0, 2),
      mask: Uint32Array.of(1, 1, 0, 0, 0),
      storage: 'resident-capacity',
      parameters: Float32Array.of(1, 1, 0, 1)
    },
    {capacity: 2}
  );
  expect(result.missingFixCount).toEqual(Uint32Array.of(0));
  expect(result.count).toEqual(Uint32Array.of(1));
});
it('B8 captures inherited mask and nested output getters once in both named forms', async () => {
  for (const Operation of [GPUTrajectoryStaypointsOperation, CPUTrajectoryStaypointsOperation]) {
    let reads = 0;
    class Props {
      positions = 'p';
      timestamps = 't';
      trackOffsets = 'o';
      parameters = 's';
      rowCount = 4;
      capacity = 1;
      get mask() {
        reads++;
        return 'm';
      }
      get output() {
        return new Output();
      }
    }
    class Output {
      get ids() {
        return 'ids';
      }
      get status() {
        return {
          count: 'count',
          overflow: 'overflow',
          candidateOverflow: 'candidate',
          missingFixCount: 'missing'
        };
      }
    }
    const operation = new Operation(new Props());
    expect(reads).toBe(1);
    expect(operation.inputs['mask']).toBe('m');
    if (operation instanceof CPUTrajectoryStaypointsOperation) {
      const result = await operation.execute({
        inputs: {
          positions: vector('float32x2', new Float32Array(8)),
          timestamps: vector('float32', Float32Array.of(0, 2, 5, 9)),
          trackOffsets: vector('uint32', Uint32Array.of(0, 4)),
          parameters: vector('float32', Float32Array.of(1, 8, 0, 1)),
          mask: vector('uint32', Uint32Array.of(1, 0, 1, 1))
        }
      });
      expect(result['count']).toEqual(Uint32Array.of(0));
      expect(result['missing']).toEqual(Uint32Array.of(1));
    }
  }
});
it('S3 rejects missing radii and JavaScript status-only mobility at construction', () => {
  expect(
    () =>
      new GPUTrajectoryRecursionsOperation({
        positions: 'p',
        timestamps: 't',
        trackOffsets: 'o',
        radii: 'r',
        rowCount: 2,
        radiusCount: 0,
        output: {revisits: 'visits'}
      })
  ).toThrow(/radiusCount/);
  // Reflect exercises the invalid JavaScript request while keeping the public union strict.
  expect(() =>
    Reflect.construct(GPUMobilityIndicatorsOperation, [
      {
        timestamps: 't',
        trackCount: 1,
        rowCount: 1,
        staypointCapacity: 1,
        staypoints: {
          tracks: 'tracks',
          centroids: 'centers',
          startRows: 'starts',
          durations: 'durations',
          locationIds: 'locations',
          status: {count: 'count'}
        },
        output: {status: {candidateOverflow: 'loss'}}
      }
    ])
  ).toThrow(/at least one output/);
});
it('S4 rejects empty supplied mask and loss names rather than discarding them', () => {
  expect(
    () =>
      new GPUTrajectoryStaypointsOperation({
        positions: 'p',
        timestamps: 't',
        trackOffsets: 'o',
        parameters: 's',
        mask: '',
        rowCount: 1,
        capacity: 1,
        output: {
          ids: 'ids',
          status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
        }
      })
  ).toThrow(/mask/);
  expect(
    () =>
      new GPUTrajectoryLocationsOperation({
        centroids: 'p',
        tracks: 't',
        count: 'n',
        parameters: 's',
        capacity: 1,
        staypointStatus: {overflow: ''},
        output: {
          locationIds: 'ids',
          status: {count: 'count', iterationLimitReached: 'iterationLimitReached'}
        }
      })
  ).toThrow(/stayOverflow/);
});
it('B8 direct staypoints captures inherited views and nested destination getters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0,
    reads = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(format: Format, length = 1) =>
    createTransientView(graph, `direct-${serial++}`, format, length);
  const mask = view('uint32', 3);
  const fields = {
    positions: view('float32x2', 3),
    timestamps: view('float32', 3),
    trackOffsets: view('uint32', 2),
    parameters: view('float32', 4),
    output: Object.create({
      ids: view('uint32', 1),
      status: Object.create({
        count: view('uint32'),
        overflow: view('uint32'),
        candidateOverflow: view('uint32')
      })
    })
  };
  const props = Object.create(fields) as typeof fields & {mask: typeof mask};
  Object.defineProperty(props, 'mask', {
    get() {
      reads++;
      return mask;
    }
  });
  try {
    const contributor = new GPUTrajectoryStaypoints(props);
    expect(contributor.props.mask).toBe(mask);
    contributor.getCommandNodes(graph);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});
