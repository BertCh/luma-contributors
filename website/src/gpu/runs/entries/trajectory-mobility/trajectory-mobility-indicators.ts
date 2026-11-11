// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMobilityIndicators on one day of twelve San Francisco Bay vessel legs, from their staypoints
// (GPUTrajectoryStaypoints), locations (GPUTrajectoryLocations) and triplegs (GPUTrajectorySegments).
// Bars: the radius of gyration of every track. The probed track's bar is bright, its radius is the
// ring around the mean of its stays, and the amber bars above are its real entropy.

import {
  getGPUTrajectoryLocationsParameterValues,
  getGPUTrajectorySegmentsParameterValues,
  GPUMobilityIndicators,
  GPUTrajectoryLocations
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {
  addSegments,
  addStaypoints,
  createMovementFleet,
  getTrackOfRow,
  readFloat32,
  readUint32,
  STAYPOINT_CAPACITY,
  uploadFleet,
  writeStaypointSettings
} from './_lib/movement';
import {formatDistance} from './_lib/units';
import {parameters} from './trajectory-mobility-indicators.parameters';

const TRACK_COUNT = 12;
const SCALE = 0.62;
const MAP_TOP = 0.38;
const BARS = [0.05, 0.04, 0.95, 0.19] as const;
const ENTROPY_BARS = [0.05, 0.21, 0.95, 0.33] as const;

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  centroids: RunBuffer;
  tracks: RunBuffer;
  staypointCount: RunBuffer;
  locationIds: RunBuffer;
  radiusOfGyration: RunBuffer;
  realEntropy: RunBuffer;
  randomEntropy: RunBuffer;
  homeLocations: RunBuffer;
  trackingQuality: RunBuffer;
  selected: RunBuffer;
  selectedStays: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'One day of twelve Bay vessel legs, scaled to leave room for the bars. Indicators made for people over weeks are here read on vessels. Bars: radius of gyration (blue) and real entropy (amber). Distances are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    // The map is drawn at a uniform scale in the upper part of the frame.
    const toDisplay = (x: number, y: number): [number, number] => [
      (x - world.width / 2) * SCALE + world.width / 2,
      MAP_TOP + y * SCALE
    ];
    const positions = Float32Array.from(fleet.positions);
    for (let row = 0; row < fleet.rowCount; row++) {
      positions.set(toDisplay(fleet.positions[row * 2], fleet.positions[row * 2 + 1]), row * 2);
    }
    const segments = Float32Array.from(fleet.segments);
    for (let segment = 0; segment < fleet.segmentCount; segment++) {
      segments.set(
        toDisplay(fleet.segments[segment * 4], fleet.segments[segment * 4 + 1]),
        segment * 4
      );
      segments.set(
        toDisplay(fleet.segments[segment * 4 + 2], fleet.segments[segment * 4 + 3]),
        segment * 4 + 2
      );
    }
    return {
      ...fleet,
      world,
      displayPositions: positions,
      displaySegments: segments,
      toDisplay,
      track: 3
    };
  },
  build(builder, fleet) {
    const buffers = uploadFleet(builder, fleet);
    const staypoints = addStaypoints(builder, fleet, buffers);
    const segments = addSegments(builder, fleet, buffers, staypoints);
    const locationParameters = builder.allocate('location-parameters', 'float32', 4);
    const locationIds = builder.allocate('location-ids', 'uint32', STAYPOINT_CAPACITY);
    const locationCount = builder.allocate('location-count', 'uint32', 1);
    builder.graph.add(
      new GPUTrajectoryLocations({
        centroids: staypoints.centroids.view,
        tracks: staypoints.tracks.view,
        count: staypoints.count.view,
        parameters: locationParameters.view,
        output: {locationIds: locationIds.view, status: {count: locationCount.view}}
      })
    );
    const trackCount = fleet.trackCount;
    const radiusOfGyration = builder.allocate('radius-of-gyration', 'float32', trackCount);
    const realEntropy = builder.allocate('real-entropy', 'float32', trackCount);
    const randomEntropy = builder.allocate('random-entropy', 'float32', trackCount);
    const homeLocations = builder.allocate('home-locations', 'uint32', trackCount);
    const trackingQuality = builder.allocate('tracking-quality', 'float32', trackCount);
    const jumpLengths = builder.allocate('jump-lengths', 'float32', STAYPOINT_CAPACITY);
    const waitingTimes = builder.allocate('waiting-times', 'float32', STAYPOINT_CAPACITY);
    builder.graph.add(
      new GPUMobilityIndicators({
        trackCount,
        timestamps: buffers.timestamps.view,
        staypoints: {
          tracks: staypoints.tracks.view,
          centroids: staypoints.centroids.view,
          startRows: staypoints.startRows.view,
          durations: staypoints.durations.view,
          locationIds: locationIds.view,
          status: {count: staypoints.count.view}
        },
        triplegs: {
          tracks: segments.triplegTracks.view,
          startRows: segments.triplegStartRows.view,
          endRows: segments.triplegEndRows.view,
          status: {count: segments.triplegCount.view},
          staypointFinishRows: staypoints.finishRows.view
        },
        output: {
          radiusOfGyration: radiusOfGyration.view,
          jumpLengths: jumpLengths.view,
          waitingTimes: waitingTimes.view,
          randomEntropy: randomEntropy.view,
          realEntropy: realEntropy.view,
          homeLocations: homeLocations.view,
          trackingQuality: trackingQuality.view
        }
      })
    );
    const outputs: Outputs = {
      segments: builder.upload('display-segments', 'float32x4', fleet.displaySegments),
      positions: builder.upload('display-positions', 'float32x2', fleet.displayPositions),
      centroids: staypoints.centroids,
      tracks: staypoints.tracks,
      staypointCount: staypoints.count,
      locationIds,
      radiusOfGyration,
      realEntropy,
      randomEntropy,
      homeLocations,
      trackingQuality,
      selected: builder.allocate('selected-bar', 'float32', trackCount),
      selectedStays: builder.allocate('selected-stays', 'float32x2', STAYPOINT_CAPACITY)
    };
    return {
      outputs,
      update: current => {
        writeStaypointSettings(staypoints, current);
        segments.parameters.write(
          getGPUTrajectorySegmentsParameterValues({maximumGap: 0, activityMinimumDuration: 40})
        );
        locationParameters.write(
          getGPUTrajectoryLocationsParameterValues({epsilon: current.epsilon})
        );
      }
    };
  },
  // The stays, in map coordinates, are read back to draw them.
  async afterRun({inputs, outputs, upload}) {
    const [count, centroids] = await Promise.all([
      readUint32(outputs.staypointCount),
      readFloat32(outputs.centroids)
    ]);
    const stays = Math.min(count[0], STAYPOINT_CAPACITY);
    const positions = new Float32Array(Math.max(1, stays) * 2);
    for (let stay = 0; stay < stays; stay++) {
      positions.set(inputs.toDisplay(centroids[stay * 2], centroids[stay * 2 + 1]), stay * 2);
    }
    return {stays, positions: upload('display-stays', 'float32x2', positions)};
  },
  // Probe: the nearest track. Its bar lights up, its stays are gathered, and the ring is the radius
  // of gyration around the mean of those stays.
  interaction: {
    kind: 'probe',
    hint: 'probe a track',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.displayPositions[2 * 120], inputs.displayPositions[2 * 120 + 1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let row = 0; row < inputs.rowCount; row++) {
        const distance = Math.hypot(
          inputs.displayPositions[row * 2] - pointer.x,
          inputs.displayPositions[row * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      const track = getTrackOfRow(inputs.trackOffsets, nearest);
      const radius = peek(outputs.radiusOfGyration);
      const real = peek(outputs.realEntropy);
      const quality = peek(outputs.trackingQuality);
      const centroids = peek(outputs.centroids);
      const tracks = peek(outputs.tracks);
      const count = peek(outputs.staypointCount);
      if (!radius || !real || !quality || !centroids || !tracks || !count) {
        return {};
      }
      let sumX = 0;
      let sumY = 0;
      let stays = 0;
      const glyphs: {shape: 'dot'; at: [number, number]}[] = [];
      for (let stay = 0; stay < Math.min(count[0], STAYPOINT_CAPACITY); stay++) {
        if (tracks[stay] !== track) continue;
        const at = inputs.toDisplay(centroids[stay * 2], centroids[stay * 2 + 1]);
        glyphs.push({shape: 'dot', at});
        sumX += centroids[stay * 2];
        sumY += centroids[stay * 2 + 1];
        stays++;
      }
      const selected = new Float32Array(inputs.trackCount);
      selected[track] = radius[track];
      outputs.selected.write(selected);
      const center = stays ? inputs.toDisplay(sumX / stays, sumY / stays) : null;
      return {
        glyphs: [
          ...glyphs,
          ...(center
            ? [
                {shape: 'ring' as const, at: center, radius: radius[track] * SCALE, dashed: true},
                {shape: 'crosshair' as const, at: center}
              ]
            : [])
        ],
        readout: `track ${track} · gyration ${formatDistance(radius[track], inputs.metersPerUnit)} · ${stays} stays · entropy ${real[track].toFixed(1)} bits · tracked ${Math.round(quality[track] * 100)}%`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5},
    ...(extra
      ? [
          {
            layer: 'dots' as const,
            positions: extra.positions,
            count: extra.stays,
            radius: 7,
            color: [1, 0.7, 0.2, 0.9] as const
          }
        ]
      : []),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.realEntropy,
      range: [0, 4],
      rectangle: ENTROPY_BARS,
      color: [...theme.categories[1], 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.radiusOfGyration,
      range: [0, 0.3],
      rectangle: BARS,
      color: [...theme.categories[0], 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.selected,
      range: [0, 0.3],
      rectangle: BARS,
      color: [1, 0.7, 0.2, 1]
    }
  ]
});
