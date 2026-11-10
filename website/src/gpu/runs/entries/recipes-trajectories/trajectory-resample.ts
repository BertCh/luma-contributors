// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryResample over irregular GPS tracks. The raw fixes (muted) are replaced by the
// same number of evenly spaced samples per track: even in time, or even along the path.

import {GPUTrajectoryResample} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createFleet, createTrackEdges} from './lib/fleet';
import {findNearestTrack, getTrackVertices} from './lib/track-lookup';
import {parameters} from './trajectory-resample.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Every track resampled to the same number of evenly spaced points.',
  createInputs: ({world}) => createFleet(world, {trackCount: 10, minimumFixes: 14}),
  build(builder, fleet, values) {
    const sampleCount = values.sampleCount;
    const samples = builder.allocate('samples', 'float32x2', fleet.trackCount * sampleCount);
    const sampleTimes = builder.allocate('sample-times', 'float32', fleet.trackCount * sampleCount);
    builder.graph.add(
      new GPUTrajectoryResample({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        sampleCount,
        spacing: values.spacing === 1 ? 'arc-length' : 'time',
        output: {samples: samples.view, sampleTimes: sampleTimes.view}
      })
    );
    return {
      outputs: {
        samples,
        sampleTimes,
        segments: builder.upload('segments', 'float32x4', fleet.segments),
        fixes: builder.upload('fixes', 'float32x2', fleet.positions),
        edges: builder.upload(
          'edges',
          'uint32x2',
          createTrackEdges(fleet.trackOffsets, sampleCount)
        )
      }
    };
  },
  // The pointer picks a track: its raw fixes against the evenly spaced samples the operation made.
  interaction: {
    kind: 'probe',
    hint: 'hover a track',
    rest: ({toUnit, inputs}) => {
      const middle = Math.floor((inputs.trackOffsets[0] + inputs.trackOffsets[1]) / 2);
      return toUnit([inputs.positions[middle * 2], inputs.positions[middle * 2 + 1]]);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {track} = findNearestTrack(inputs, pointer.x, pointer.y);
      const glyphs: RunGlyph[] = [{shape: 'polyline', points: getTrackVertices(inputs, track)}];
      const samples = peek(outputs.samples);
      if (samples) {
        const sampleCount = Math.round(values.sampleCount);
        for (let sample = 0; sample < sampleCount; sample++) {
          const row = track * sampleCount + sample;
          glyphs.push({shape: 'dot', at: [samples[row * 2], samples[row * 2 + 1]]});
        }
      }
      const fixCount = inputs.trackOffsets[track + 1] - inputs.trackOffsets[track];
      const spacing = values.spacing === 1 ? 'arc-length' : 'time';
      return {
        glyphs,
        readout: `track ${track} · ${fixCount} fixes -> ${values.sampleCount} samples · ${spacing}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.fixes, radius: 2},
    {
      layer: 'segments',
      nodes: outputs.samples,
      edges: outputs.edges,
      values: outputs.sampleTimes,
      radius: 1.5
    },
    {layer: 'dots', positions: outputs.samples, values: outputs.sampleTimes, radius: 5}
  ]
});
