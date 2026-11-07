// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUClassificationFit scores how well the classes describe the values: per-class medians (top
// bars) and mean absolute deviations from them (bottom bars), the inputs of the fit statistics.
// The map shows the class of each tract.

import {
  GPUClassAssignment,
  GPUClassificationFit
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './classification-fit.parameters';
import {createBreaks, createDynamics, getPeriodOwners} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Class medians (top) and absolute deviations (bottom) of the classified values.',
  createInputs: ({random, world}) => createDynamics(random, world),
  build(builder, {tracts, values}, current) {
    const classCount = current.classCount;
    const classes = builder.allocate('classes', 'uint32', values.length);
    const input = builder.upload('values', 'float32', values);
    builder.graph.add(
      new GPUClassAssignment({
        values: input.view,
        breaks: builder.upload(
          'breaks',
          'float32',
          createBreaks(values, classCount, current.quantile === 1)
        ).view,
        classCount: builder.upload('class-count', 'uint32', Uint32Array.of(classCount)).view,
        output: classes.view
      })
    );
    const medians = builder.allocate('medians', 'float32', classCount + 1);
    const deviations = builder.allocate('absolute-deviations', 'float32', classCount + 1);
    builder.graph.add(
      new GPUClassificationFit({
        values: input.view,
        classes: classes.view,
        classCount,
        output: {
          counts: builder.allocate('counts', 'uint32', classCount + 1).view,
          medians: medians.view,
          absoluteDeviations: deviations.view,
          squaredDeviations: builder.allocate('squared-deviations', 'float32', classCount + 1).view,
          summary: builder.allocate('summary', 'float32', 5).view
        }
      })
    );
    const mesh = uploadTractMesh(builder, tracts);
    mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, current.period));
    return {outputs: {classes, medians, deviations, ...mesh}};
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
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.medians,
      range: [0, 1],
      rectangle: [0.58, 0.55, 0.97, 0.92],
      ramp: 'sequential'
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.deviations,
      range: 'auto',
      rectangle: [0.58, 0.08, 0.97, 0.45],
      ramp: 'sequential'
    }
  ]
});
