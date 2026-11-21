// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryLocations after GPUTrajectoryStaypoints: the staypoints of one vessel that lie
// within epsilon of each other (chained) become one location. Each staypoint is a dot colored by
// its location, joined to the location's center.

import {
  getGPUTrajectoryLocationsParameterValues,
  GPUTrajectoryLocations
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, snapToParameter, findParameter} from '../../interaction';
import {
  addStaypoints,
  createMovementFleet,
  findReferenceStaypoints,
  readFloat32,
  readUint32,
  STAYPOINT_CAPACITY,
  uploadFleet,
  writeStaypointSettings
} from './_lib/movement';
import {formatDistance} from './_lib/units';
import {parameters} from './trajectory-locations.parameters';

const STAYPOINT_SETTINGS = {distanceThreshold: 0.004, timeThreshold: 10};
const TRACK_COUNT = 12;

type Outputs = {
  segments: RunBuffer;
  centroids: RunBuffer;
  tracks: RunBuffer;
  staypointCount: RunBuffer;
  locationIds: RunBuffer;
  locationCount: RunBuffer;
  iterationLimitReached: RunBuffer;
  complete: RunBuffer;
  centers: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Staypoints of 12 San Francisco Bay vessels (distanceThreshold 0.004, timeThreshold 10 playback s, about 24 minutes of vessel time). Dots are colored by location; a line joins each to its location center. Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    const reference = findReferenceStaypoints(
      fleet,
      STAYPOINT_SETTINGS.distanceThreshold,
      STAYPOINT_SETTINGS.timeThreshold
    );
    // The staypoint nearest the middle of the frame anchors the rest position of the pointer.
    const anchor = reference.reduce((best, candidate) =>
      Math.hypot(candidate.centroid[0] - world.width / 2, candidate.centroid[1] - 0.5) <
      Math.hypot(best.centroid[0] - world.width / 2, best.centroid[1] - 0.5)
        ? candidate
        : best
    ).centroid;
    return {...fleet, anchor};
  },
  build(builder, fleet) {
    const buffers = uploadFleet(builder, fleet);
    const staypoints = addStaypoints(builder, fleet, buffers);
    const parameterBuffer = builder.allocate('location-parameters', 'float32', 4);
    const locationIds = builder.allocate('location-ids', 'uint32', STAYPOINT_CAPACITY);
    const locationCount = builder.allocate('location-count', 'uint32', 1);
    const iterationLimitReached = builder.allocate('iteration-limit-reached', 'uint32', 1);
    const complete = builder.allocate('location-complete', 'uint32', 1);
    const centers = builder.allocate('location-centers', 'float32x2', STAYPOINT_CAPACITY);
    builder.graph.add(
      new GPUTrajectoryLocations({
        centroids: staypoints.centroids.view,
        tracks: staypoints.tracks.view,
        count: staypoints.count.view,
        staypointStatus: {
          overflow: staypoints.overflow.view,
          candidateOverflow: staypoints.candidateOverflow.view
        },
        parameters: parameterBuffer.view,
        output: {
          locationIds: locationIds.view,
          status: {
            count: locationCount.view,
            iterationLimitReached: iterationLimitReached.view,
            complete: complete.view
          },
          centers: centers.view
        }
      })
    );
    const outputs: Outputs = {
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      centroids: staypoints.centroids,
      tracks: staypoints.tracks,
      staypointCount: staypoints.count,
      locationIds,
      locationCount,
      iterationLimitReached,
      complete,
      centers
    };
    return {
      outputs,
      update: current => {
        writeStaypointSettings(staypoints, STAYPOINT_SETTINGS);
        parameterBuffer.write(getGPUTrajectoryLocationsParameterValues({epsilon: current.epsilon}));
      }
    };
  },
  // Each staypoint is joined to the center of its location. The links are rebuilt from the run.
  async afterRun({outputs, upload}) {
    const [count, locationTotal, ids, centroids, centers, iterationLimitReached, complete] =
      await Promise.all([
        readUint32(outputs.staypointCount),
        readUint32(outputs.locationCount),
        readUint32(outputs.locationIds),
        readFloat32(outputs.centroids),
        readFloat32(outputs.centers),
        readUint32(outputs.iterationLimitReached),
        readUint32(outputs.complete)
      ]);
    const settled = Boolean(complete[0] && !iterationLimitReached[0]);
    const staypointCount = settled ? Math.min(count[0], STAYPOINT_CAPACITY) : 0;
    const links = new Float32Array(Math.max(1, staypointCount) * 4);
    for (let row = 0; row < staypointCount; row++) {
      const location = ids[row];
      links.set(
        [
          centroids[row * 2],
          centroids[row * 2 + 1],
          centers[location * 2],
          centers[location * 2 + 1]
        ],
        row * 4
      );
    }
    return {
      staypointCount,
      locationCount: settled ? locationTotal[0] : 0,
      links: upload('links', 'float32x4', links)
    };
  },
  // Reach: the distance from the nearest staypoint is epsilon.
  interaction: {
    kind: 'reach',
    hint: 'reach from a staypoint',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.anchor[0] + parameters[0].default, inputs.anchor[1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const centroids = peek(outputs.centroids);
      const count = peek(outputs.staypointCount);
      const locations = peek(outputs.locationCount);
      const complete = peek(outputs.complete);
      if (!centroids || !count || !complete) return {};
      if (!complete[0]) return {readout: 'Locations unavailable due to an incomplete result'};
      let nearest = -1;
      let nearestDistance = Infinity;
      for (let row = 0; row < Math.min(count[0], STAYPOINT_CAPACITY); row++) {
        const distance = Math.hypot(
          centroids[row * 2] - pointer.x,
          centroids[row * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      if (nearest < 0) return {};
      const parameter = findParameter(parameters, 'epsilon');
      const epsilon = snapToParameter(
        parameter,
        clamp(nearestDistance, parameter.min, parameter.max)
      );
      const at: [number, number] = [centroids[nearest * 2], centroids[nearest * 2 + 1]];
      return {
        values: {epsilon},
        glyphs: [
          {shape: 'ring', at, radius: epsilon},
          {shape: 'line', from: at, to: [pointer.x, pointer.y], dashed: true}
        ],
        readout: `epsilon ${formatDistance(epsilon, inputs.metersPerUnit)} · ${locations ? locations[0] : 0} locations from ${count[0]} staypoints`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    ...(extra
      ? [
          {
            layer: 'lines' as const,
            segments: extra.links,
            count: extra.staypointCount,
            radius: 2,
            alpha: 0.8
          },
          {
            layer: 'dots' as const,
            positions: outputs.centroids,
            values: outputs.locationIds,
            count: extra.staypointCount,
            radius: 9
          },
          {
            layer: 'dots' as const,
            positions: outputs.centers,
            count: extra.locationCount,
            radius: 16,
            alpha: 0.45
          }
        ]
      : [])
  ]
});
