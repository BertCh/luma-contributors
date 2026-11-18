// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPU_MOVEMENT_SENTINEL,
  GPUTrajectoryLocations,
  GPUTrajectorySegments,
  GPUTrajectoryStaypoints,
  getGPUTrajectoryLocationsParameterValues,
  getGPUTrajectorySegmentsParameterValues,
  getGPUTrajectoryStaypointsParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/movement-trackintel.json';
import {GraphHarness, importTrajectory} from './movement-spec-utils';

const CONTEXT = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  sphereRadius: 6371000
} as const;

function toSentinel(values: number[]): number[] {
  return values.map(value => (value < 0 ? GPU_MOVEMENT_SENTINEL : value));
}

describe('trackintel parity', () => {
  for (const testCase of fixture.cases) {
    it(`matches ${testCase.name}: staypoints, triplegs, locations and trips`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new GraphHarness(device, `trackintel-${testCase.name}`);
      const {positions, timestamps, trackOffsets, rowCount} = importTrajectory(harness, {
        positions: fixture.scenario.positions,
        times: fixture.scenario.timesMs.map(value => BigInt(value)),
        trackOffsets: fixture.scenario.trackOffsets
      });
      const capacity = 32;
      const staypointIds = harness.output('staypoint-ids', 'uint32', capacity);
      const stayCount = harness.output('stay-count', 'uint32', 1);
      const stayOverflow = harness.output('stay-overflow', 'uint32', 1);
      const stayStarts = harness.output('stay-starts', 'uint32', capacity);
      const stayEnds = harness.output('stay-ends', 'uint32', capacity);
      const stayFinishes = harness.output('stay-finishes', 'uint32', capacity);
      const centroids = harness.output('centroids', 'float32x2', capacity);
      const durations = harness.output('durations', 'float32', capacity);
      const rowStaypointIds = harness.output('row-staypoints', 'uint32', rowCount);
      harness.graph.add(
        new GPUTrajectoryStaypoints({
          positions,
          timestamps,
          trackOffsets,
          spatialContext: CONTEXT,
          parameters: harness.input(
            'stay-parameters',
            getGPUTrajectoryStaypointsParameterValues({
              distanceThreshold: testCase.distanceThreshold,
              timeThreshold: testCase.timeThresholdMinutes * 60_000,
              maximumGap: testCase.gapThresholdMinutes * 60_000,
              includeLast: testCase.includeLast
            }),
            'float32',
            4
          ),
          output: {
            ids: staypointIds.view,
            status: {count: stayCount.view, overflow: stayOverflow.view},
            startRows: stayStarts.view,
            endRows: stayEnds.view,
            finishRows: stayFinishes.view,
            centroids: centroids.view,
            durations: durations.view,
            staypointIds: rowStaypointIds.view
          }
        })
      );
      const locationIds = harness.output('location-ids', 'uint32', capacity);
      const locationCount = harness.output('location-count', 'uint32', 1);
      const converged = harness.output('converged', 'uint32', 1);
      const centers = harness.output('centers', 'float32x2', capacity);
      harness.graph.add(
        new GPUTrajectoryLocations({
          centroids: centroids.view,
          tracks: staypointIds.view,
          count: stayCount.view,
          spatialContext: CONTEXT,
          parameters: harness.input(
            'location-parameters',
            getGPUTrajectoryLocationsParameterValues({epsilon: 150}),
            'float32',
            4
          ),
          output: {
            locationIds: locationIds.view,
            status: {count: locationCount.view, converged: converged.view},
            centers: centers.view
          }
        })
      );
      const triplegIds = harness.output('tripleg-tracks', 'uint32', capacity);
      const triplegCount = harness.output('tripleg-count', 'uint32', 1);
      const triplegOverflow = harness.output('tripleg-overflow', 'uint32', 1);
      const triplegStarts = harness.output('tripleg-starts', 'uint32', capacity);
      const triplegEnds = harness.output('tripleg-ends', 'uint32', capacity);
      const rowTriplegIds = harness.output('row-triplegs', 'uint32', rowCount);
      const tripIds = harness.output('trip-tracks', 'uint32', capacity);
      const tripCount = harness.output('trip-count', 'uint32', 1);
      const tripOverflow = harness.output('trip-overflow', 'uint32', 1);
      const tripStarts = harness.output('trip-starts', 'uint32', capacity);
      const tripFinishes = harness.output('trip-finishes', 'uint32', capacity);
      const tripOrigins = harness.output('trip-origins', 'uint32', capacity);
      const tripDestinations = harness.output('trip-destinations', 'uint32', capacity);
      const triplegTripIds = harness.output('tripleg-trips', 'uint32', capacity);
      harness.graph.add(
        new GPUTrajectorySegments({
          positions,
          timestamps,
          trackOffsets,
          staypointIds: rowStaypointIds.view,
          staypoints: {
            tracks: staypointIds.view,
            startRows: stayStarts.view,
            finishRows: stayFinishes.view,
            durations: durations.view,
            status: {count: stayCount.view}
          },
          parameters: harness.input(
            'segment-parameters',
            getGPUTrajectorySegmentsParameterValues({
              maximumGap: testCase.gapThresholdMinutes * 60_000,
              activityMinimumDuration: 15 * 60_000
            }),
            'float32',
            4
          ),
          output: {
            triplegs: {
              ids: triplegIds.view,
              status: {count: triplegCount.view, overflow: triplegOverflow.view},
              startRows: triplegStarts.view,
              endRows: triplegEnds.view,
              rowTriplegIds: rowTriplegIds.view
            },
            trips: {
              ids: tripIds.view,
              status: {
                count: tripCount.view,
                overflow: tripOverflow.view,
                candidateOverflow: harness.output('trip-candidate-overflow', 'uint32', 1).view
              },
              startRows: tripStarts.view,
              finishRows: tripFinishes.view,
              originStaypoints: tripOrigins.view,
              destinationStaypoints: tripDestinations.view,
              triplegTripIds: triplegTripIds.view
            }
          }
        })
      );
      await harness.run(async () => {
        const stays = (await readUint32(stayCount.buffer, 1))[0];
        expect(stays).toBe(testCase.staypoints.length);
        expect((await readUint32(stayOverflow.buffer, 1))[0]).toBe(0);
        expect(await readUint32(rowStaypointIds.buffer, rowCount)).toEqual(
          toSentinel(testCase.rowStaypointIds)
        );
        expect(await readUint32(locationIds.buffer, stays)).toEqual(
          toSentinel(testCase.locationIds)
        );
        expect((await readUint32(locationCount.buffer, 1))[0]).toBe(testCase.locations.length);
        expect((await readUint32(converged.buffer, 1))[0]).toBe(1);
        const centerValues = await readFloat32(centers.buffer, testCase.locations.length * 2);
        testCase.locations.forEach((location, index) => {
          expect(Math.abs(centerValues[2 * index] - location.center[0])).toBeLessThan(5e-6);
          expect(Math.abs(centerValues[2 * index + 1] - location.center[1])).toBeLessThan(5e-6);
        });
        expect(await readUint32(rowTriplegIds.buffer, rowCount)).toEqual(
          toSentinel(testCase.rowTriplegIds)
        );
        expect((await readUint32(triplegCount.buffer, 1))[0]).toBe(testCase.triplegs.length);
        expect((await readUint32(triplegOverflow.buffer, 1))[0]).toBe(0);
        const triplegStartRows = await readUint32(triplegStarts.buffer, testCase.triplegs.length);
        const triplegEndRows = await readUint32(triplegEnds.buffer, testCase.triplegs.length);
        testCase.triplegs.forEach((tripleg, index) => {
          expect(triplegStartRows[index]).toBe(tripleg.startRow);
          expect(triplegEndRows[index]).toBe(tripleg.endRow);
        });
        expect((await readUint32(tripCount.buffer, 1))[0]).toBe(testCase.trips.length);
        expect((await readUint32(tripOverflow.buffer, 1))[0]).toBe(0);
        const tripTracks = await readUint32(tripIds.buffer, testCase.trips.length);
        const tripStartRows = await readUint32(tripStarts.buffer, testCase.trips.length);
        const tripFinishRows = await readUint32(tripFinishes.buffer, testCase.trips.length);
        const origins = await readUint32(tripOrigins.buffer, testCase.trips.length);
        const destinations = await readUint32(tripDestinations.buffer, testCase.trips.length);
        testCase.trips.forEach((trip, index) => {
          expect(tripTracks[index]).toBe(trip.track);
          expect(tripStartRows[index]).toBe(trip.startRow);
          expect(tripFinishRows[index]).toBe(trip.finishRow);
          expect(origins[index]).toBe(trip.origin < 0 ? GPU_MOVEMENT_SENTINEL : trip.origin);
          expect(destinations[index]).toBe(
            trip.destination < 0 ? GPU_MOVEMENT_SENTINEL : trip.destination
          );
        });
        expect(await readUint32(triplegTripIds.buffer, testCase.triplegs.length)).toEqual(
          toSentinel(testCase.triplegTripIds)
        );
      });
    }, 180_000);
  }
});
