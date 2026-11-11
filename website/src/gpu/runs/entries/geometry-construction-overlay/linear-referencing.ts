// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULinearReferencing snaps points to the nearest line within a search radius and reports the
// foot point and the measure (distance along the line). Muted lines are the roads; snapped points
// are colored by the normalized measure of their foot point, joined to it by a short connector.
// Points with no road within `radius` stay muted.

import {GPULinearReferencing} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass} from './shared/kernel';
import {findNearestSegment} from '../../pick';
import {createLineSet, createPackedSegments, OUTPUT_COLOR} from './shared/lines';
import {createLegend} from '../../text';
import {parameters} from './linear-referencing.parameters';

const QUERY_COUNT = 60;
const RADIUS_RANGE = [0.01, 0.25] as const;

export default defineOperationRun({
  parameters,
  caption: 'Points snapped to the nearest road; color is the position along the road.',
  createInputs: ({random, world}) => {
    const queries = Float32Array.from({length: QUERY_COUNT * 2}, (_, slot) =>
      slot % 2 === 0 ? (0.04 + 0.92 * random()) * world.width : 0.02 + 0.96 * random()
    );
    return {lines: createLineSet(random, world, 4, 8), queries, world};
  },
  build(builder, inputs) {
    const positions = builder.upload('positions', 'float32x2', inputs.lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lines.lineOffsets);
    const queries = builder.upload('queries', 'float32x2', inputs.queries);
    const radius = builder.allocate('radius', 'float32', 1);
    const lineIndices = builder.allocate('line-indices', 'uint32', QUERY_COUNT);
    const footPoints = builder.allocate('foot-points', 'float32x2', QUERY_COUNT);
    const normalizedMeasures = builder.allocate('normalized-measures', 'float32', QUERY_COUNT);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULinearReferencing({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        queryPositions: queries.view,
        radius: radius.view,
        candidateCapacity: 4096,
        output: {
          lineIndices: lineIndices.view,
          footPoints: footPoints.view,
          normalizedMeasures: normalizedMeasures.view,
          status: {candidateOverflow: overflow.view}
        }
      })
    );
    // One connector from every point to its foot point; unmatched points (NaN foot) go off-screen.
    const connectors = builder.allocate('connectors', 'float32x4', QUERY_COUNT);
    addKernelPass(builder, {
      id: 'connector-pass',
      invocationCount: QUERY_COUNT,
      bindings: [
        {name: 'queries', buffer: queries, access: 'read'},
        {name: 'footPoints', buffer: footPoints, access: 'read'},
        {name: 'connectors', buffer: connectors, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let foot = footPoints[footPointsOffset + index];
  let isMatched = foot.x == foot.x && foot.y == foot.y;
  connectors[connectorsOffset + index] = select(vec4f(-9.0), vec4f(queries[queriesOffset + index], foot), isMatched);`
    });
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(inputs.lines.positions, inputs.lines.lineOffsets, false)
    );
    return {
      outputs: {original, queries, lineIndices, footPoints, normalizedMeasures, connectors},
      update: current => radius.write(Float32Array.of(current.radius))
    };
  },
  // The distance from the pointer to the nearest road is the search radius: every one of the 60
  // points snaps to the road when it lies within that distance, so the matches grow with the ring.
  interaction: {
    kind: 'reach',
    hint: 'move off a road',
    rest: ({inputs, toUnit}) => {
      const target = parameters[0].default;
      // The point to the right of the first road vertex that is `radius` away from every road.
      const vertex: [number, number] = [inputs.lines.positions[0], inputs.lines.positions[1]];
      let best: [number, number] = vertex;
      let bestError = Infinity;
      for (let step = 0; step < 200; step++) {
        const x = Math.min(vertex[0] + step * 0.005, inputs.world.width);
        const distance = findNearestSegment(inputs.lines.segments, x, vertex[1])?.distance ?? 0;
        if (Math.abs(distance - target) < bestError) {
          bestError = Math.abs(distance - target);
          best = [x, vertex[1]];
        }
      }
      return toUnit(best);
    },
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = findNearestSegment(inputs.lines.segments, pointer.x, pointer.y);
      if (!hit) return {};
      // Rounded up to the slider step (0.005), so the nearest road is always inside the ring.
      const steppedDistance = Number((Math.ceil(hit.distance * 200 - 1e-6) / 200).toFixed(3));
      const radius = Math.min(Math.max(steppedDistance, RADIUS_RANGE[0]), RADIUS_RANGE[1]);
      const measures = peek(outputs.normalizedMeasures);
      let snapped = 0;
      for (let row = 0; measures && row < measures.length; row++) {
        if (Number.isFinite(measures[row])) snapped++;
      }
      return {
        values: {radius},
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius, dashed: true},
          {shape: 'line', from: [pointer.x, pointer.y], to: hit.point, dashed: true},
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `radius ${radius.toFixed(3)} · ${measures ? `${snapped} of ${QUERY_COUNT}` : '…'} snapped`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.original,
      radius: 4.5,
      color: [...theme.ink, 0.5]
    },
    {layer: 'dots', positions: outputs.queries, color: [...theme.muted, 1], radius: 6},
    {layer: 'lines', segments: outputs.connectors, color: OUTPUT_COLOR, alpha: 0.95, radius: 4},
    {
      layer: 'dots',
      positions: outputs.footPoints,
      values: outputs.normalizedMeasures,
      range: [0, 1],
      radius: 11
    },
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        entries: [
          {label: 'road', swatch: theme.ink},
          {label: 'query point', swatch: theme.muted},
          {
            label: 'link to its foot point',
            swatch: [OUTPUT_COLOR[0], OUTPUT_COLOR[1], OUTPUT_COLOR[2]]
          },
          {
            label: 'foot point, color = measure',
            swatch: theme.ramp[Math.floor(theme.ramp.length * 0.6)]
          }
        ]
      })
    }
  ]
});
