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
import {createLineSet, createPackedSegments, HIGHLIGHT_COLOR, OUTPUT_COLOR} from './shared/lines';
import {parameters} from './line-locate.parameters';

const LINE_COUNT = 4;
const EVENTS_PER_LINE = 6;
const EVENT_COUNT = LINE_COUNT * EVENTS_PER_LINE;

export default defineOperationRun({
  parameters,
  caption: 'Events located at a distance along their route; ticks show the direction.',
  createInputs: ({random, world}) => ({
    lines: createLineSet(random, world, LINE_COUNT, 8, 1),
    eventLineIndices: Uint32Array.from({length: EVENT_COUNT}, (_, event) =>
      Math.floor(event / EVENTS_PER_LINE)
    ),
    eventMeasures: Float32Array.from(
      {length: EVENT_COUNT},
      (_, event) => (event % EVENTS_PER_LINE) * 0.12
    )
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
      outputs: {original, eventPositions, ticks},
      update: current =>
        parameterBuffer.write(
          getGPULineLocateParameterValues({
            measureScale: current.measureScale,
            measureOffset: current.measureOffset
          })
        )
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {layer: 'lines', segments: outputs.ticks, color: HIGHLIGHT_COLOR, radius: 3},
    {layer: 'dots', positions: outputs.eventPositions, color: OUTPUT_COLOR, radius: 9}
  ]
});
