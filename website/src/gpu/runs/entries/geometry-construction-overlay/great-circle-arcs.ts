// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGreatCircleArcs turns origin and destination longitude/latitude pairs into arcs along the
// great circle. On this flat (equirectangular) picture the arcs bow toward the poles, while the
// muted straight lines are the flat-map shortcut.

import {
  getGPUGreatCircleArcsParameterValues,
  GPUGreatCircleArcs
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  allocateLineOutput,
  type DrawTransform,
  HIGHLIGHT_COLOR,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './great-circle-arcs.parameters';

const PAIR_COUNT = 7;

export default defineOperationRun({
  parameters,
  caption: 'Great-circle arcs between city pairs; muted lines are the straight flat-map paths.',
  createInputs: ({random, world}) => {
    const sources = new Float32Array(PAIR_COUNT * 2);
    const targets = new Float32Array(PAIR_COUNT * 2);
    const flat = new Float32Array(PAIR_COUNT * 4);
    const transform: DrawTransform = {
      scale: [world.width / 360, 1 / 150],
      offset: [world.width / 2, 0.5]
    };
    for (let pair = 0; pair < PAIR_COUNT; pair++) {
      const side = pair % 2 === 0 ? -1 : 1;
      const longitude = side * (20 + random() * 60) - side * 40;
      sources.set([longitude - 50 + random() * 20, 20 + random() * 50], pair * 2);
      targets.set([longitude + 50 + random() * 20, 20 + random() * 50], pair * 2);
      if (pair % 3 === 0) {
        sources[pair * 2 + 1] *= -0.6;
        targets[pair * 2 + 1] *= -0.6;
      }
      flat.set(
        [
          sources[pair * 2] * transform.scale[0] + transform.offset[0],
          sources[pair * 2 + 1] * transform.scale[1] + transform.offset[1],
          targets[pair * 2] * transform.scale[0] + transform.offset[0],
          targets[pair * 2 + 1] * transform.scale[1] + transform.offset[1]
        ],
        pair * 4
      );
    }
    return {sources, targets, flat, transform};
  },
  build(builder, inputs) {
    const sources = builder.upload('sources', 'float32x2', inputs.sources);
    const targets = builder.upload('targets', 'float32x2', inputs.targets);
    const result = allocateLineOutput(builder, 'arcs', 4000, PAIR_COUNT);
    const parameterBuffer = builder.allocate('arc-parameters', 'float32', 4);
    builder.graph.add(
      new GPUGreatCircleArcs({
        sources: sources.view,
        targets: targets.view,
        parameters: parameterBuffer.view,
        output: result.output
      })
    );
    const drawing = addPackedSegments(builder, 'arcs', result.positions, result.lineOffsets, {
      transform: inputs.transform
    });
    const flat = builder.upload('flat', 'float32x4', inputs.flat);
    return {
      outputs: {flat, ...drawing},
      update: current =>
        parameterBuffer.write(
          getGPUGreatCircleArcsParameterValues({
            maximumSegmentLength: current.maximumSegmentLength * 1000,
            minimumSegments: current.minimumSegments
          })
        )
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.flat},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 1.8},
    {layer: 'dots', positions: outputs.points, radius: 2.6, color: HIGHLIGHT_COLOR}
  ]
});
