// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURoseStatistic bins the directions of line segments (left, the input) into a rose histogram
// (right, the `bins` output). Axial mode folds opposite directions together.

import {GPURoseStatistic} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTrajectories} from '../../inputs';
import {parameters} from './rose-statistic.parameters';

const ROSE_RADIUS = 0.4;
const ROSE_INNER = 0.06;
const ARC_STEPS = 4;

/** Direction of a segment in radians counter-clockwise from +x, folded into [0, period). */
function getAngle(segment: Float32Array, row: number, period: number): number {
  const angle = Math.atan2(
    segment[row * 4 + 3] - segment[row * 4 + 1],
    segment[row * 4 + 2] - segment[row * 4]
  );
  return ((angle % period) + period) % period;
}

export default defineOperationRun({
  parameters,
  caption:
    'Vessel headings in San Francisco Bay, as a rose of bars by direction; hover a bar for its legs.',
  createInputs: ({world, random}) => {
    const half = {width: world.width * 0.5, height: world.height};
    return {
      ...createTrajectories(random, half, 16, 25),
      center: [world.width * 0.75, world.height / 2] as [number, number]
    };
  },
  build(builder, tracks, values) {
    const segments = tracks.segments;
    // Remove stationary fixes, whose direction is undefined.
    const starts: number[] = [];
    const ends: number[] = [];
    const kept: number[] = [];
    for (let index = 0; index < tracks.count; index++) {
      const [startX, startY, endX, endY] = segments.subarray(index * 4, index * 4 + 4);
      if (Math.hypot(endX - startX, endY - startY) <= values.minimumLength) continue;
      starts.push(startX, startY);
      ends.push(endX, endY);
      kept.push(startX, startY, endX, endY);
    }
    const bins = builder.allocate('bins', 'float32', values.binCount);
    builder.graph.add(
      new GPURoseStatistic({
        starts: builder.upload('starts', 'float32x2', Float32Array.from(starts)).view,
        ends: builder.upload('ends', 'float32x2', Float32Array.from(ends)).view,
        circular: {
          mode: values.mode === 0 ? 'directional' : 'axial',
          angleUnit: 'radians',
          origin: 'positive-x-counter-clockwise',
          binCount: values.binCount
        },
        output: {summary: builder.allocate('summary', 'float32', 5).view, bins: bins.view}
      })
    );
    return {
      outputs: {
        bins,
        segments: builder.upload('segments', 'float32x4', segments),
        // The legs in the hovered direction bin; unused rows stay NaN and are not drawn.
        highlight: builder.upload(
          'highlight',
          'float32x4',
          new Float32Array(tracks.count * 4).fill(Number.NaN)
        ),
        kept: Float32Array.from(kept)
      }
    };
  },
  // The wedges are as long as the bins the operation wrote, read back after each run.
  async afterRun({outputs, inputs, values, upload}) {
    const bytes = await outputs.bins.buffer.readAsync();
    const counts = new Float32Array(bytes.buffer, bytes.byteOffset, values.binCount);
    const maximum = Math.max(...counts, 1);
    const isAxial = values.mode === 1;
    const repeats = isAxial ? 2 : 1;
    const period = isAxial ? Math.PI : Math.PI * 2;
    const vertices: number[] = [];
    const indices: number[] = [];
    const owners: number[] = [];
    for (let repeat = 0; repeat < repeats; repeat++) {
      for (let bin = 0; bin < values.binCount; bin++) {
        const radius = ROSE_INNER + (ROSE_RADIUS - ROSE_INNER) * (counts[bin] / maximum);
        const first = vertices.length / 2;
        vertices.push(inputs.center[0], inputs.center[1]);
        for (let step = 0; step <= ARC_STEPS; step++) {
          const angle = repeat * Math.PI + ((bin + step / ARC_STEPS) / values.binCount) * period;
          vertices.push(
            inputs.center[0] + Math.cos(angle) * radius,
            inputs.center[1] + Math.sin(angle) * radius
          );
        }
        for (let step = 0; step < ARC_STEPS; step++) {
          indices.push(first, first + 1 + step, first + 2 + step);
          owners.push(bin);
        }
      }
    }
    return {
      counts,
      vertices: upload('rose-vertices', 'float32x2', Float32Array.from(vertices)),
      indices: upload('rose-indices', 'uint32', Uint32Array.from(indices)),
      owners: upload('rose-owners', 'uint32', Uint32Array.from(owners)),
      triangleCount: owners.length
    };
  },
  // The pointer's compass angle around the rose picks a direction bin: the wedge is outlined and
  // the legs heading that way are lit on the left.
  interaction: {
    kind: 'probe',
    hint: 'point at a direction',
    rest: ({inputs, toUnit}) =>
      toUnit([inputs.center[0] + ROSE_RADIUS * 0.7, inputs.center[1] + ROSE_RADIUS * 0.3]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const isAxial = values.mode === 1;
      const period = isAxial ? Math.PI : Math.PI * 2;
      const angle = Math.atan2(pointer.y - inputs.center[1], pointer.x - inputs.center[0]);
      const folded = ((angle % period) + period) % period;
      const bin = Math.min(values.binCount - 1, Math.floor((folded / period) * values.binCount));
      const lit = new Float32Array(outputs.highlight.length * 4).fill(Number.NaN);
      let count = 0;
      for (let row = 0; row < outputs.kept.length / 4; row++) {
        const legAngle = getAngle(outputs.kept, row, period);
        if (
          Math.min(values.binCount - 1, Math.floor((legAngle / period) * values.binCount)) !== bin
        )
          continue;
        lit.set(outputs.kept.subarray(row * 4, row * 4 + 4), count * 4);
        count++;
      }
      outputs.highlight.write(lit);
      const counts = peek(outputs.bins);
      const radius = counts
        ? ROSE_INNER + (ROSE_RADIUS - ROSE_INNER) * (counts[bin] / Math.max(...counts, 1))
        : ROSE_RADIUS;
      const wedge: number[] = [inputs.center[0], inputs.center[1]];
      for (const mirror of isAxial ? [0, Math.PI] : [0]) {
        for (let step = 0; step <= ARC_STEPS; step++) {
          const edge = mirror + ((bin + step / ARC_STEPS) / values.binCount) * period;
          wedge.push(
            inputs.center[0] + Math.cos(edge) * radius,
            inputs.center[1] + Math.sin(edge) * radius
          );
        }
      }
      const degrees = ((bin * 360) / values.binCount) * (isAxial ? 0.5 : 1);
      const width = (360 / values.binCount) * (isAxial ? 0.5 : 1);
      return {
        focus: bin,
        glyphs: [
          {shape: 'ring', at: inputs.center, radius: ROSE_RADIUS, dashed: true},
          {shape: 'polyline', points: wedge, closed: true}
        ],
        readout: `${degrees.toFixed(0)}° to ${(degrees + width).toFixed(0)}° · ${count} legs`
      };
    }
  },
  draw: ({outputs, inputs, extra}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.segments,
      count: inputs.count,
      color: [0.8, 0.85, 1, 0.6]
    },
    {
      layer: 'lines',
      segments: outputs.highlight,
      count: inputs.count,
      radius: 2.2,
      color: [1, 0.85, 0.3, 1]
    },
    {
      layer: 'polygons',
      positions: extra.vertices,
      indices: extra.indices,
      owners: extra.owners,
      values: outputs.bins,
      triangleCount: extra.triangleCount,
      ramp: 'sequential'
    }
  ]
});
