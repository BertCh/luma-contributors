// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineLocate places events at a measure (distance along their line) and reports the position
// and direction there. Each event is a dot on its route with a short tangent tick; moving
// `measureOffset` slides every event along its route, `measureScale` stretches the spacing.

import {
  getGPULineLocateParameterValues,
  GPULineLocate
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass} from './shared/kernel';
import {findParameter, snapToParameter} from '../../interaction';
import {
  createLineSet,
  createPackedSegments,
  createTickGlyph,
  findNearestLine,
  getPointAtMeasure,
  HIGHLIGHT_COLOR
} from './shared/lines';
import {parameters} from './line-locate.parameters';

const LINE_COUNT = 4;
const EVENTS_PER_LINE = 6;
const EVENT_COUNT = LINE_COUNT * EVENTS_PER_LINE;
const EVENT_SPACING = 0.07;

export default defineOperationRun({
  parameters,
  caption: 'Events located at a distance along their route; ticks show the direction.',
  createInputs: ({random, world}) => ({
    lines: createLineSet(random, world, LINE_COUNT, 8),
    eventLineIndices: Uint32Array.from({length: EVENT_COUNT}, (_, event) =>
      Math.floor(event / EVENTS_PER_LINE)
    ),
    eventMeasures: Float32Array.from(
      {length: EVENT_COUNT},
      (_, event) => (event % EVENTS_PER_LINE) * EVENT_SPACING
    ),
    eventIndices: Float32Array.from({length: EVENT_COUNT}, (_, event) => event % EVENTS_PER_LINE)
  }),
  build(builder, inputs) {
    const positions = builder.upload('positions', 'float32x2', inputs.lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lines.lineOffsets);
    const eventLineIndices = builder.upload('event-lines', 'uint32', inputs.eventLineIndices);
    const eventMeasures = builder.upload('event-measures', 'float32', inputs.eventMeasures);
    const parameterBuffer = builder.allocate('locate-parameters', 'float32', 4);
    const eventPositions = builder.allocate('event-positions', 'float32x2', EVENT_COUNT);
    const tangents = builder.allocate('event-tangents', 'float32x2', EVENT_COUNT);
    const statuses = builder.allocate('event-statuses', 'uint32', EVENT_COUNT);
    builder.graph.add(
      new GPULineLocate({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        eventLineIndices: eventLineIndices.view,
        eventMeasures: eventMeasures.view,
        parameters: parameterBuffer.view,
        output: {positions: eventPositions.view, tangents: tangents.view, statuses: statuses.view}
      })
    );
    const eventIndices = builder.upload('event-indices', 'float32', inputs.eventIndices);
    const ticks = builder.allocate('ticks', 'float32x4', EVENT_COUNT);
    addKernelPass(builder, {
      id: 'tick-pass',
      invocationCount: EVENT_COUNT,
      bindings: [
        {name: 'eventPositions', buffer: eventPositions, access: 'read'},
        {name: 'tangents', buffer: tangents, access: 'read'},
        {name: 'ticks', buffer: ticks, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let center = eventPositions[eventPositionsOffset + index];
  let direction = tangents[tangentsOffset + index];
  ticks[ticksOffset + index] = vec4f(center, center + direction * 0.05);`
    });
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(inputs.lines.positions, inputs.lines.lineOffsets, false)
    );
    return {
      outputs: {original, eventPositions, eventIndices, ticks},
      update: current =>
        parameterBuffer.write(
          getGPULineLocateParameterValues({
            measureScale: current.measureScale,
            measureOffset: current.measureOffset
          })
        )
    };
  },
  // The pointer is a location on a route: its measure (distance along the nearest route from the
  // start) becomes measureOffset, so event 0 of that route sits under the cursor.
  interaction: {
    kind: 'place',
    hint: 'move along a route',
    rest: ({inputs, toUnit}) =>
      toUnit(
        getPointAtMeasure(inputs.lines, 0, findParameter(parameters, 'measureOffset').default).point
      ),
    resolve(pointer, {inputs, values}) {
      const hit = findNearestLine(inputs.lines, pointer.x, pointer.y);
      if (!hit) return {};
      const measureOffset = snapToParameter(
        findParameter(parameters, 'measureOffset'),
        hit.measure
      );
      return {
        values: {measureOffset},
        glyphs: [
          createTickGlyph(inputs.lines.positions, hit.row, hit.point, 0.035),
          {shape: 'ring', at: hit.point, radius: 0.025}
        ],
        readout: `measureOffset ${measureOffset.toFixed(2)} · scale ${values.measureScale.toFixed(2)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {layer: 'lines', segments: outputs.ticks, color: HIGHLIGHT_COLOR, radius: 3},
    {
      layer: 'dots',
      positions: outputs.eventPositions,
      values: outputs.eventIndices,
      ramp: 'sequential',
      range: [0, EVENTS_PER_LINE - 1],
      radius: 9
    }
  ]
});
