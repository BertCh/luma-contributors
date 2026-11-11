// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryEncounters over tracks resampled to a shared clock. The dots are the resampled
// table, one column per bucket; a link joins the two tracks of a pair at the first bucket where they
// were within `distance`; each bar is one encounter pair (sorted by track) whose height is the number
// of buckets they spent within it.

import {
  GPUTrajectoryEncounters,
  GPUTrajectoryResample
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createLegend, mergeText} from '../../text';
import {createFleet, raiseFleet} from './lib/fleet';
import {addBucketLinks} from './lib/pair-links';
import {findBusiestFix} from './lib/track-lookup';
import {parameters} from './trajectory-encounters.parameters';

const PAIR_CAPACITY = 64;
const CELL_SIZE = 0.16;
/** Pointer travel (world units) per unit of encounter distance, so the whole range fits the canvas. */
const POINTER_PER_DISTANCE = 4;

export default defineOperationRun({
  parameters,
  caption: 'Track pairs within a distance of each other in the same time bucket.',
  createInputs: ({world}) => {
    const fleet = raiseFleet(
      createFleet(world, {trackCount: 14, minimumFixes: 24, maximumFixes: 34}),
      0.42
    );
    return {...fleet, anchor: findBusiestFix(fleet, 0.12)};
  },
  build(builder, fleet, values) {
    const bucketCount = values.bucketCount;
    const distance = builder.allocate('distance', 'float32', 1);
    const samples = builder.allocate('samples', 'float32x2', fleet.trackCount * bucketCount);
    const ids = builder.allocate('pair-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('pair-count', 'uint32', 1);
    const overflow = builder.allocate('pair-overflow', 'uint32', 1);
    const bucketCounts = builder.allocate('bucket-counts', 'uint32', PAIR_CAPACITY);
    const candidateOverflow = builder.allocate('candidate-overflow', 'uint32', 1);
    const partners = builder.allocate('pair-partners', 'uint32', PAIR_CAPACITY);
    const firstBuckets = builder.allocate('pair-first-buckets', 'uint32', PAIR_CAPACITY);
    builder.graph.add(
      new GPUTrajectoryResample({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        sampleCount: bucketCount,
        output: {samples: samples.view}
      })
    );
    builder.graph.add(
      new GPUTrajectoryEncounters({
        samples: samples.view,
        trackCount: fleet.trackCount,
        bucketCount,
        distance: distance.view,
        cellSize: CELL_SIZE,
        bounds: [0, 0, fleet.worldWidth, 1],
        hitCapacity: 1 << 16,
        output: {
          pairs: {
            ids: ids.view,
            status: {count: count.view, overflow: overflow.view},
            partners: partners.view,
            firstBuckets: firstBuckets.view,
            bucketCounts: bucketCounts.view
          },
          status: {candidateOverflow: candidateOverflow.view}
        }
      })
    );
    const pairLinks = addBucketLinks(builder, {
      ids,
      partners,
      firstBuckets,
      count,
      samples,
      bucketCount,
      pairCapacity: PAIR_CAPACITY
    });
    return {
      outputs: {
        samples,
        links: pairLinks.links,
        linkEnds: pairLinks.ends,
        count,
        bucketCounts,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => distance.write(Float32Array.of(current.distance))
    };
  },
  rebuildOn: ['bucketCount'],
  // Distance is how far the pointer is from a busy spot: the ring is the encounter distance.
  interaction: {
    kind: 'reach',
    hint: 'stretch the distance',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.anchor[0] + parameters[0].default * POINTER_PER_DISTANCE, inputs.anchor[1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const distance = snapToParameter(
        findParameter(parameters, 'distance'),
        Math.hypot(pointer.x - inputs.anchor[0], pointer.y - inputs.anchor[1]) /
          POINTER_PER_DISTANCE
      );
      const pairCount = peek(outputs.count);
      return {
        values: {distance},
        glyphs: [
          {shape: 'ring', at: inputs.anchor, radius: distance},
          {shape: 'line', from: inputs.anchor, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: inputs.anchor}
        ],
        readout: `distance ${distance.toFixed(3)} · ${pairCount ? pairCount[0] : 0} pairs met`
      };
    }
  },
  draw: ({inputs, outputs, theme, world}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.samples, radius: 3.5},
    {layer: 'lines', segments: outputs.links, radius: 2.6, color: [...theme.categories[2], 1]},
    {layer: 'dots', positions: outputs.linkEnds, radius: 6, color: [...theme.categories[2], 1]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bucketCounts,
      ramp: 'sequential',
      rectangle: [0.06, 0.05, 0.94, 0.38]
    },
    {
      layer: 'text',
      ...mergeText(
        {
          labels: [
            {
              text: 'bars: buckets each pair spent within the distance, one bar per pair',
              at: [0.06 * world.width, 0.385],
              anchor: 'bottom-left',
              role: 'title'
            }
          ]
        },
        createLegend({
          corner: 'top-left',
          entries: [
            {label: 'resampled position (one per bucket)', swatch: [...theme.muted]},
            {label: 'pair in the same bucket (first one)', swatch: 2}
          ]
        })
      )
    }
  ]
});
