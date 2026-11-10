// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULinearReferencing snaps points to the nearest line within a search radius and reports the
// foot point and the measure (distance along the line). Muted lines are the roads; snapped points
// are colored by the normalized measure of their foot point, joined to it by a short connector.
// Points with no road within `radius` stay muted.

import {GPULinearReferencing} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass} from './shared/kernel';
import {createLineSet, createPackedSegments, OUTPUT_COLOR} from './shared/lines';
import {parameters} from './linear-referencing.parameters';

const QUERY_COUNT = 60;

export default defineOperationRun({
  parameters,
  caption: 'Points snapped to the nearest road; color is the position along the road.',
  createInputs: ({random, world}) => {
    const queries = Float32Array.from({length: QUERY_COUNT * 2}, (_, slot) =>
      slot % 2 === 0 ? (0.04 + 0.92 * random()) * world.width : 0.02 + 0.96 * random()
    );
    // Query 0 is the one the pointer carries.
    return {
      lines: createLineSet(random, world, 4, 8),
      queries,
      initial: [queries[0], queries[1]] as const
    };
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
          status: {overflow: overflow.view}
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
      update: current => {
        radius.write(Float32Array.of(current.radius));
        queries.write(inputs.queries);
      }
    };
  },
  // The pointer is query point 0: the operation snaps it to the nearest road within radius, and the
  // foot point, connector and measure are its answer.
  interaction: {
    kind: 'place',
    hint: 'move the query point',
    rest: ({inputs, toUnit}) => toUnit(inputs.initial),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.queries[0] = pointer.x;
      inputs.queries[1] = pointer.y;
      const lineIndex = peek(outputs.lineIndices)?.[0];
      const measure = peek(outputs.normalizedMeasures)?.[0];
      const isSnapped = measure !== undefined && Number.isFinite(measure);
      return {
        rerun: true,
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius: values.radius, dashed: true},
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: isSnapped
          ? `radius ${values.radius.toFixed(3)} · road ${lineIndex} · measure ${measure.toFixed(2)}`
          : `radius ${values.radius.toFixed(3)} · no road in reach`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 4},
    {layer: 'dots', positions: outputs.queries, color: [0.82, 0.84, 0.9, 0.85], radius: 3.5},
    {layer: 'lines', segments: outputs.connectors, color: OUTPUT_COLOR, alpha: 0.8, radius: 2},
    {
      layer: 'dots',
      positions: outputs.footPoints,
      values: outputs.normalizedMeasures,
      range: [0, 1],
      radius: 8
    }
  ]
});
