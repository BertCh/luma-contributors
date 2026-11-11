// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryClosestApproach between raw tracks. A link joins the two tracks of a pair at the
// instant of their closest approach, which falls between fixes (no resampling); each bar is one pair
// of tracks, as tall as the time they spent within `distance` of each other (solved exactly).

import {GPUTrajectoryClosestApproach} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getChartWorldX} from '../../pick';
import {createLegend, mergeText} from '../../text';
import {createFleet, raiseFleet} from './lib/fleet';
import {addClosestApproachLinks} from './lib/pair-links';
import {
  findBusiestFix,
  findNearestTrack,
  getTrackPositionAt,
  getTrackVertices
} from './lib/track-lookup';
import {parameters} from './trajectory-closest-approach.parameters';

const PAIR_CAPACITY = 64;
const CHART_RECTANGLE = [0.06, 0.05, 0.94, 0.38] as const;

export default defineOperationRun({
  parameters,
  caption: 'Exact time spent within the distance, per pair of tracks.',
  createInputs: ({world}) => {
    const fleet = raiseFleet(
      createFleet(world, {trackCount: 14, skipTracks: 4, minimumFixes: 24, maximumFixes: 34}),
      0.42
    );
    return {...fleet, busiestFix: findBusiestFix(fleet, 0.12)};
  },
  build(builder, fleet) {
    const distance = builder.allocate('distance', 'float32', 1);
    const ids = builder.allocate('pair-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('pair-count', 'uint32', 1);
    const overflow = builder.allocate('pair-overflow', 'uint32', 1);
    const durations = builder.allocate('durations', 'float32', PAIR_CAPACITY);
    const partners = builder.allocate('partners', 'uint32', PAIR_CAPACITY);
    const closestTimes = builder.allocate('closest-times', 'float32', PAIR_CAPACITY);
    const entryOverflow = builder.allocate('entry-overflow', 'uint32', 1);
    const hitOverflow = builder.allocate('hit-overflow', 'uint32', 1);
    const positions = builder.upload('positions', 'float32x2', fleet.positions);
    const timestamps = builder.upload('timestamps', 'float32', fleet.timestamps);
    const trackOffsets = builder.upload('offsets', 'uint32', fleet.trackOffsets);
    builder.graph.add(
      new GPUTrajectoryClosestApproach({
        positions: positions.view,
        timestamps: timestamps.view,
        trackOffsets: trackOffsets.view,
        distance: distance.view,
        cellSize: 0.16,
        timeCellSize: 60,
        entryCapacity: 1 << 16,
        hitCapacity: 1 << 16,
        output: {
          pairs: {
            ids: ids.view,
            status: {count: count.view, overflow: overflow.view},
            partners: partners.view,
            closestTimes: closestTimes.view,
            durations: durations.view
          },
          status: {entryOverflow: entryOverflow.view, hitOverflow: hitOverflow.view}
        }
      })
    );
    const pairLinks = addClosestApproachLinks(builder, {
      ids,
      partners,
      closestTimes,
      count,
      positions,
      timestamps,
      trackOffsets,
      pairCapacity: PAIR_CAPACITY
    });
    return {
      outputs: {
        links: pairLinks.links,
        linkEnds: pairLinks.ends,
        ids,
        count,
        partners,
        closestTimes,
        durations,
        nodes: builder.upload('nodes', 'float32x2', fleet.positions),
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => distance.write(Float32Array.of(current.distance))
    };
  },
  // The pointer picks a track; its longest encounter is drawn: the partner, and the closest point.
  interaction: {
    kind: 'probe',
    hint: 'hover a track',
    rest: ({toUnit, inputs}) => toUnit(inputs.busiestFix),
    resolve(pointer, {inputs, outputs, values, peek, world}) {
      const {track} = findNearestTrack(inputs, pointer.x, pointer.y);
      const ids = peek(outputs.ids);
      const partners = peek(outputs.partners);
      const times = peek(outputs.closestTimes);
      const durations = peek(outputs.durations);
      const count = peek(outputs.count);
      const glyphs: RunGlyph[] = [{shape: 'polyline', points: getTrackVertices(inputs, track)}];
      if (!ids || !partners || !times || !durations || !count) return {glyphs};
      let bestRow = -1;
      for (let row = 0; row < Math.min(count[0], ids.length); row++) {
        if (ids[row] !== track && partners[row] !== track) continue;
        if (bestRow < 0 || durations[row] > durations[bestRow]) bestRow = row;
      }
      if (bestRow < 0) return {glyphs, readout: `track ${track} · none within ${values.distance}`};
      const partner = ids[bestRow] === track ? partners[bestRow] : ids[bestRow];
      glyphs.push({shape: 'polyline', points: getTrackVertices(inputs, partner), dashed: true});
      const meeting = getTrackPositionAt(inputs, track, times[bestRow]);
      if (meeting) glyphs.push({shape: 'ring', at: meeting, radius: values.distance});
      const barX = getChartWorldX(CHART_RECTANGLE, world, (bestRow + 0.5) / PAIR_CAPACITY);
      glyphs.push({shape: 'dot', at: [barX, (CHART_RECTANGLE[1] - 0.025) * world.height]});
      return {
        glyphs,
        readout: `tracks ${track} & ${partner} · ${Math.round(durations[bestRow])} s within ${values.distance}`
      };
    }
  },
  draw: ({inputs, outputs, theme, world}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.nodes, radius: 3},
    {layer: 'lines', segments: outputs.links, radius: 2.6, color: [...theme.categories[1], 1]},
    {layer: 'dots', positions: outputs.linkEnds, radius: 6, color: [...theme.categories[1], 1]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.durations,
      ramp: 'sequential',
      rectangle: [...CHART_RECTANGLE]
    },
    {
      layer: 'text',
      ...mergeText(
        {
          labels: [
            {
              text: 'bars: time each pair spent within the distance, one bar per pair',
              at: [0.06 * world.width, 0.385],
              anchor: 'bottom-left',
              role: 'title'
            }
          ]
        },
        createLegend({
          corner: 'top-left',
          entries: [
            {label: 'raw fix', swatch: [...theme.muted]},
            {label: 'closest approach of a pair (between fixes)', swatch: 1}
          ]
        })
      )
    }
  ]
});
