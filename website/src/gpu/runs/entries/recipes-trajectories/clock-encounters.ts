// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addClockEncounters: staggered tracks resampled onto one shared clock (dots colored by clock
// instant, early to late; absent samples are not drawn) and the encounters found on it. A link joins
// the two tracks of a pair at the first clock instant they met; each bar is one pair, as tall as the
// number of clock instants the two were within `distance`.

import {
  addClockEncounters,
  getGPUTrajectoryClockParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createLegend, mergeText} from '../../text';
import {createFleet, raiseFleet} from './lib/fleet';
import {addBucketLinks} from './lib/pair-links';
import {parameters} from './clock-encounters.parameters';

const PAIR_CAPACITY = 64;
const BUCKET_COUNT = 24;

export default defineOperationRun({
  parameters,
  caption: 'Tracks on one clock, and the pairs that met on it.',
  createInputs: ({world}) =>
    raiseFleet(
      createFleet(world, {
        trackCount: 14,
        skipTracks: 9,
        minimumFixes: 20,
        maximumFixes: 30
      }),
      0.42
    ),
  build(builder, fleet) {
    const clock = builder.allocate('clock', 'float32', 4);
    const distance = builder.allocate('distance', 'float32', 1);
    const samples = builder.allocate('samples', 'float32x2', fleet.trackCount * BUCKET_COUNT);
    const ids = builder.allocate('pair-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('pair-count', 'uint32', 1);
    const overflow = builder.allocate('pair-overflow', 'uint32', 1);
    const bucketCounts = builder.allocate('bucket-counts', 'uint32', PAIR_CAPACITY);
    const candidateOverflow = builder.allocate('candidate-overflow', 'uint32', 1);
    const partners = builder.allocate('pair-partners', 'uint32', PAIR_CAPACITY);
    const firstBuckets = builder.allocate('pair-first-buckets', 'uint32', PAIR_CAPACITY);
    addClockEncounters(builder.graph, {
      positions: builder.upload('positions', 'float32x2', fleet.positions).view,
      timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
      trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
      clock: clock.view,
      bucketCount: BUCKET_COUNT,
      distance: distance.view,
      cellSize: 0.16,
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
        status: {candidateOverflow: candidateOverflow.view},
        samples: samples.view
      }
    });
    const pairLinks = addBucketLinks(builder, {
      ids,
      partners,
      firstBuckets,
      count,
      samples,
      bucketCount: BUCKET_COUNT,
      pairCapacity: PAIR_CAPACITY
    });
    return {
      outputs: {
        samples,
        instants: builder.upload(
          'instants',
          'float32',
          Float32Array.from(
            {length: fleet.trackCount * BUCKET_COUNT},
            (_, row) => row % BUCKET_COUNT
          )
        ),
        links: pairLinks.links,
        linkEnds: pairLinks.ends,
        count,
        bucketCounts,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => {
        distance.write(Float32Array.of(current.distance));
        clock.write(
          getGPUTrajectoryClockParameterValues({start: current.start, step: current.step})
        );
      }
    };
  },
  // x moves the clock's start and y its step: the window of sampled instants is drawn on a time strip.
  interaction: bindAxes<unknown, {count: RunBuffer<'uint32'>}>(parameters, {
    kind: 'pad',
    hint: 'start and step',
    x: 'start',
    y: 'step',
    decorate: (values, _pointer, {outputs, world, peek}) => {
      const pairCount = peek(outputs.count);
      const toWorldX = (seconds: number) => world.width * (0.06 + 0.88 * (seconds / 600));
      const end = Math.min(600, values.start + values.step * (BUCKET_COUNT - 1));
      return {
        glyphs: [
          {shape: 'line', from: [toWorldX(values.start), 0.018], to: [toWorldX(end), 0.018]},
          {shape: 'dot', at: [toWorldX(values.start), 0.018]}
        ],
        readout: `clock start ${values.start} s · step ${values.step} s · ${pairCount ? pairCount[0] : 0} pairs met`
      };
    }
  }),
  draw: ({inputs, outputs, theme, world}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    // One dot per track per clock instant, colored by the instant.
    {
      layer: 'dots',
      positions: outputs.samples,
      values: outputs.instants,
      ramp: 'sequential',
      range: [0, BUCKET_COUNT - 1],
      radius: 4.5
    },
    {layer: 'lines', segments: outputs.links, radius: 2.6, color: [...theme.ink, 1]},
    {layer: 'dots', positions: outputs.linkEnds, radius: 6, color: [...theme.ink, 1]},
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
              text: 'bars: clock instants each pair spent within the distance, one bar per pair',
              at: [0.06 * world.width, 0.385],
              anchor: 'bottom-left',
              role: 'title'
            }
          ]
        },
        createLegend({
          corner: 'top-left',
          entries: [
            {label: 'early clock instant', swatch: [...theme.ramp[2]]},
            {label: 'late clock instant', swatch: [...theme.ramp[5]]},
            {label: 'pair met (first shared instant)', swatch: [...theme.ink]}
          ]
        })
      )
    }
  ]
});
