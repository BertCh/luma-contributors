// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Fix review fr-geometry-06. The test GPU's `atan2(y, -0.0)` returns the opposite quarter turn.
// A step due north from x = +0 to x = -0 has a zero east component of negative sign (IEEE:
// -0 - +0 = -0), and so does its great-circle bearing. Every expectation below is the direction of
// the step read off the coordinates: due north is +pi / 2 (90 degrees), due south is -pi / 2.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULineLocate,
  GPULinearReferencing
} from '../../../src/gpu-spatial-analysis/linear-referencing/index';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics
} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const HALF_PI = Math.PI / 2;

// Two tracks of two fixes one time unit apart: north across the signed zero, then south across it.
const NORTH_AND_SOUTH = {
  positions: [
    [0, 0],
    [-0, 1],
    [0, 2],
    [-0, 1]
  ],
  trackOffsets: [0, 2, 4]
};

it('GPUTrajectoryPlayhead headings are right for due north and south steps across signed zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const spatialContext of [undefined, GREAT_CIRCLE]) {
    const harness = new GraphHarness(device, 'playhead-signed-zero');
    const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
      positions: NORTH_AND_SOUTH.positions,
      times: Float32Array.of(0, 1, 0, 1),
      trackOffsets: NORTH_AND_SOUTH.trackOffsets
    });
    const headings = harness.output('headings', 'float32', 2);
    harness.graph.add(
      new GPUTrajectoryPlayhead({
        spatialContext,
        positions,
        timestamps,
        trackOffsets,
        parameters: harness.input(
          'playhead-parameters',
          getGPUTrajectoryPlayheadParameterValues({playhead: 0.5}),
          'float32',
          4
        ),
        output: {headings: headings.view}
      })
    );
    const values = await harness.run(async () => readFloat32(headings.buffer, 2));
    const label = spatialContext ? 'great-circle' : 'planar';
    expect(values[0], `${label} north`).toBeCloseTo(HALF_PI, 5);
    expect(values[1], `${label} south`).toBeCloseTo(-HALF_PI, 5);
  }
});

it('GPUTrajectoryMetrics great-circle heading is right for a due north step across signed zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'metrics-signed-zero');
  const {positions, timestamps, trackOffsets} = importTrajectory(harness, {
    positions: NORTH_AND_SOUTH.positions,
    times: Float32Array.of(0, 1, 0, 1),
    trackOffsets: NORTH_AND_SOUTH.trackOffsets
  });
  const headings = harness.output('headings', 'float32', 4);
  harness.graph.add(
    new GPUTrajectoryMetrics({
      spatialContext: GREAT_CIRCLE,
      positions,
      timestamps,
      trackOffsets,
      parameters: harness.input(
        'metrics-parameters',
        getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 0, stopMinimumDuration: 0}),
        'float32',
        4
      ),
      output: {stepHeadings: headings.view}
    })
  );
  const values = await harness.run(async () => readFloat32(headings.buffer, 4));
  expect(values[1], 'north').toBeCloseTo(HALF_PI, 5);
  expect(values[3], 'south').toBeCloseTo(-HALF_PI, 5);
});

it('GPULineLocate angles are right for lines due north and south across signed zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const spatialContext of [undefined, GREAT_CIRCLE]) {
    const harness = new GraphHarness(device, 'line-locate-signed-zero');
    const angles = harness.output('angles', 'float32', 2);
    const located = harness.output('located', 'float32x2', 2);
    const statuses = harness.output('statuses', 'uint32', 2);
    harness.graph.add(
      new GPULineLocate({
        spatialContext,
        positions: harness.input(
          'line-positions',
          Float32Array.from(NORTH_AND_SOUTH.positions.flat()),
          'float32x2',
          4
        ),
        lineOffsets: harness.input('line-offsets', Uint32Array.of(0, 2, 4), 'uint32', 3),
        eventLineIndices: harness.input('event-lines', Uint32Array.of(0, 1), 'uint32', 2),
        eventMeasures: harness.input('event-measures', Float32Array.of(0.5, 0.5), 'float32', 2),
        measureMode: 'fraction',
        output: {positions: located.view, angles: angles.view, statuses: statuses.view}
      })
    );
    const result = await harness.run(async () => ({
      angles: await readFloat32(angles.buffer, 2),
      statuses: await readUint32(statuses.buffer, 2)
    }));
    const label = spatialContext ? 'great-circle' : 'planar';
    expect(Array.from(result.statuses), label).toEqual([0, 0]);
    expect(result.angles[0], `${label} north`).toBeCloseTo(90, 3);
    expect(result.angles[1], `${label} south`).toBeCloseTo(-90, 3);
  }
});

it('GPULinearReferencing finds the high point of a long arc that starts on the equator at latitude -0', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One great-circle arc from (0, -0) to (120, 30). Its great circle starts on the equator, so
  // the arc's highest latitude is 90 degrees of arc from the start: the vertex of the circle,
  // tan(peak) = tan(30) / sin(120), peak = 33.69 degrees at longitude 90. Both end points are below
  // that, so a query at (90, 33.6) is only found when the arc's bounds reach the peak, which
  // needs the sign of atan2(tangentZ, upZ) with upZ = sin(-0) = -0.
  const peakLatitude =
    (Math.atan(Math.tan(Math.PI / 6) / Math.sin((2 * Math.PI) / 3)) * 180) / Math.PI;
  expect(peakLatitude).toBeCloseTo(33.69, 2);
  const harness = new GraphHarness(device, 'linear-referencing-signed-zero');
  const lineIndices = harness.output('line-indices', 'uint32', 1);
  const distances = harness.output('distances', 'float32', 1);
  const overflow = harness.output('overflow', 'uint32', 1);
  harness.graph.add(
    new GPULinearReferencing({
      spatialContext: GREAT_CIRCLE,
      positions: harness.input('line-positions', Float32Array.of(0, -0, 120, 30), 'float32x2', 2),
      lineOffsets: harness.input('line-offsets', Uint32Array.of(0, 2), 'uint32', 2),
      queryPositions: harness.input('queries', Float32Array.of(90, 33.6), 'float32x2', 1),
      radius: harness.input('radius', Float32Array.of(50_000), 'float32', 1),
      output: {
        lineIndices: lineIndices.view,
        distances: distances.view,
        status: {candidateOverflow: overflow.view}
      }
    })
  );
  const result = await harness.run(async () => ({
    line: (await readUint32(lineIndices.buffer, 1))[0],
    distance: (await readFloat32(distances.buffer, 1))[0]
  }));
  expect(result.line).toBe(0);
  // 0.09 degrees of latitude below the peak, on a sphere of 6,371,008.8 m: about 10 km.
  const expectedDistance = ((peakLatitude - 33.6) * Math.PI * 6_371_008.8) / 180;
  expect(Math.abs(result.distance - expectedDistance)).toBeLessThan(200);
});
