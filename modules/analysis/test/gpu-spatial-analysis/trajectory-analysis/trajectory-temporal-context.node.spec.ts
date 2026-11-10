// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_IRREGULAR_TEMPORAL_AXIS,
  type GPUTemporalContext
} from '../../../src/analysis-core/contracts/temporal-context';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {GPUTrajectoryClosestApproach} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/index';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
const ROWS = 8;
type Format = 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x2' | 'uint32x4';

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {graph, view};
}

type Fixture = ReturnType<typeof createFixture>;

/** Builds one contributor over the shared trajectory port. */
const CONTRIBUTORS: Record<string, (fixture: Fixture, port: any) => unknown> = {
  metrics: ({view}, port) =>
    new GPUTrajectoryMetrics({
      ...port,
      parameters: view('float32', 4),
      output: {trackLengths: view('float32', 1)}
    }),
  resample: ({view}, port) =>
    new GPUTrajectoryResample({...port, sampleCount: 2, output: {samples: view('float32x2', 2)}}),
  playhead: ({view}, port) =>
    new GPUTrajectoryPlayhead({
      ...port,
      parameters: view(port.timestamps.format === 'uint32x2' ? 'uint32' : 'float32', 4),
      output: {currentPositions: view('float32x2', 1)}
    }),
  'closest approach': ({view}, port) =>
    new GPUTrajectoryClosestApproach({
      ...port,
      distance: view('float32', 1),
      cellSize: 10,
      timeCellSize: 60,
      entryCapacity: 64,
      hitCapacity: 64,
      output: {
        pairs: {
          ids: view('uint32', 4),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        },
        status: {entryOverflow: view('uint32', 1), hitOverflow: view('uint32', 1)}
      }
    }),
  'zone events': ({view}, port) =>
    new GPUZoneEvents({
      tracks: {kind: 'trajectories', ...port},
      zones: {
        kind: 'polygons',
        positions: view('float32x2', 4),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2)
      },
      candidateCapacity: 64,
      trackEventCapacity: 4,
      output: {
        events: {
          ids: view('uint32', 4),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        }
      }
    }),
  kalman: ({view}, port) =>
    new GPUTrajectoryKalman({
      ...port,
      parameters: view('float32', 4),
      output: {
        filteredState: view('float32x4', ROWS),
        filteredCovariance: view('float32x4', ROWS),
        status: view('uint32', ROWS)
      }
    }),
  restriction: ({view}, port) =>
    new GPUTrajectoryRestriction({
      ...port,
      timeOrigin: port.timestamps.format === 'uint32x2' ? view('uint32', 2) : undefined,
      parameters: view('float32', 7),
      output: {
        intervals: view('float32x2', ROWS),
        positions: view('float32x4', ROWS),
        status: view('uint32', ROWS),
        relationships: view('uint32x4', 1)
      }
    })
};

const FLOAT_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: GPU_IRREGULAR_TEMPORAL_AXIS
};
const WORD_CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: GPU_IRREGULAR_TEMPORAL_AXIS
};

function createPort(
  fixture: Fixture,
  format: 'float32' | 'uint32x2',
  context?: GPUTemporalContext
) {
  const {view} = fixture;
  return {
    positions: view('float32x2', ROWS),
    timestamps: view(format, ROWS),
    trackOffsets: view('uint32', 2),
    temporalContext: context
  };
}

for (const [name, create] of Object.entries(CONTRIBUTORS)) {
  it(`${name} accepts a matching trajectory temporalContext and rejects a mismatch`, () => {
    const fixture = createFixture();
    expect(() => create(fixture, createPort(fixture, 'float32', FLOAT_CONTEXT))).not.toThrow();
    expect(() => create(fixture, createPort(fixture, 'uint32x2', WORD_CONTEXT))).not.toThrow();
    expect(() => create(fixture, createPort(fixture, 'float32', undefined))).not.toThrow();
    // float32 rows cannot be Int64 words, and the reverse.
    expect(() => create(fixture, createPort(fixture, 'float32', WORD_CONTEXT))).toThrow(
      /does not support int64-words/
    );
    expect(() => create(fixture, createPort(fixture, 'uint32x2', FLOAT_CONTEXT))).toThrow(
      /does not support float32-relative/
    );
    // Trajectory samples are declared irregular; float32 cannot count from the Unix epoch.
    expect(() =>
      create(
        fixture,
        createPort(fixture, 'float32', {
          ...FLOAT_CONTEXT,
          axis: {kind: 'regular', step: 1}
        })
      )
    ).toThrow(/regular temporal axis/);
    expect(() =>
      create(fixture, createPort(fixture, 'float32', {...FLOAT_CONTEXT, epoch: 'unix'}))
    ).toThrow(/bigint origin/);
  });
}

it('metrics declares double-single for float32 timestamps with timestampsLow', () => {
  const fixture = createFixture();
  const {view} = fixture;
  const doubleSingle: GPUTemporalContext = {...FLOAT_CONTEXT, encoding: 'double-single'};
  const create = (context: GPUTemporalContext) =>
    CONTRIBUTORS['metrics'](fixture, {
      ...createPort(fixture, 'float32', context),
      timestampsLow: view('float32', ROWS)
    });
  expect(() => create(doubleSingle)).not.toThrow();
  expect(() => create(FLOAT_CONTEXT)).toThrow(/does not support float32-relative/);
});

it('Kalman takes the port or the legacy props and checks timeUnitsPerSecond against the context', () => {
  const fixture = createFixture();
  const {view} = fixture;
  const output = {
    filteredState: view('float32x4', ROWS),
    filteredCovariance: view('float32x4', ROWS),
    status: view('uint32', ROWS)
  };
  const parameters = view('float32', 4);
  const legacy = new GPUTrajectoryKalman({
    positions: view('float32x2', ROWS),
    times: view('float32', ROWS),
    offsets: view('uint32', 2),
    parameters,
    output
  });
  expect(legacy.props.times.format).toBe('float32');
  const port = createPort(fixture, 'uint32x2', WORD_CONTEXT);
  const kalman = new GPUTrajectoryKalman({...port, parameters, output});
  expect(kalman.props.times).toBe(port.timestamps);
  expect(kalman.props.offsets).toBe(port.trackOffsets);
  expect(
    () => new GPUTrajectoryKalman({...port, timeUnitsPerSecond: 1000, parameters, output})
  ).not.toThrow();
  expect(
    () => new GPUTrajectoryKalman({...port, timeUnitsPerSecond: 60, parameters, output})
  ).toThrow();
  expect(
    () =>
      new GPUTrajectoryKalman({
        ...port,
        times: port.timestamps,
        parameters,
        output
      } as never)
  ).toThrow();
});
