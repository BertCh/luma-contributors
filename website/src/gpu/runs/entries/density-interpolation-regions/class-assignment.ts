// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUClassAssignment puts the values of every period into classes of one shared set of breaks.
// The map shows the class of each tract in the chosen period.

import {GPUClassAssignment} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './class-assignment.parameters';
import {createBreaks, createDynamics, getPeriodOwners} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Tracts colored by class; the same breaks apply to every period.',
  createInputs: ({random, world}) => createDynamics(random, world, 12, 1),
  build(builder, {tracts, values}) {
    const input = builder.upload('values', 'float32', values);
    const breaks = builder.allocate('breaks', 'float32', 10);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    const classes = builder.allocate('classes', 'uint32', values.length);
    builder.graph.add(
      new GPUClassAssignment({
        values: input.view,
        breaks: breaks.view,
        classCount: classCount.view,
        output: classes.view
      })
    );
    const mesh = uploadTractMesh(builder, tracts);
    return {
      outputs: {classes, ...mesh},
      update: current => {
        classCount.write(Uint32Array.of(current.classCount));
        breaks.write(createBreaks(values, current.classCount, current.breaks === 1));
        mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, current.period));
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.classes,
      triangleCount: inputs.tracts.triangleCount,
      ramp: 'sequential',
      range: [0, values.classCount - 1]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.tracts.outlineCount,
      alpha: 0.4
    }
  ]
});
