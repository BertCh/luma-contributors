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
import {findMiddleTract, findTractAt, uploadTractMesh} from './support/geometry';
import {addFloatSlice} from './support/gather';
import {allocateRegionOutline, drawRegionOutline, writeRegionOutline} from './support/regions';

export default defineOperationRun({
  parameters,
  caption:
    'Class medians (top) and summed absolute deviations from them (bottom), over all periods.',
  createInputs: ({random, world}) => {
    const dynamics = createDynamics(random, world);
    return {...dynamics, focus: {tract: findMiddleTract(dynamics.tracts, world)}};
  },
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
    const counts = builder.allocate('counts', 'uint32', classCount + 1);
    const summary = builder.allocate('summary', 'float32', 5);
    const medians = builder.allocate('medians', 'float32', classCount + 1);
    const deviations = builder.allocate('absolute-deviations', 'float32', classCount + 1);
    builder.graph.add(
      new GPUClassificationFit({
        values: input.view,
        classes: classes.view,
        classCount,
        output: {
          counts: counts.view,
          medians: medians.view,
          absoluteDeviations: deviations.view,
          squaredDeviations: builder.allocate('squared-deviations', 'float32', classCount + 1).view,
          summary: summary.view
        }
      })
    );
    const mesh = uploadTractMesh(builder, tracts);
    return {
      outputs: {
        classes,
        counts,
        summary,
        medians,
        deviations,
        // The class rows only: the operation reserves one more row than there are classes.
        medianBars: addFloatSlice(builder, medians, classCount, 'median-bars'),
        deviationBars: addFloatSlice(builder, deviations, classCount, 'deviation-bars'),
        highlight: allocateRegionOutline(builder, {tracts}),
        ...mesh
      },
      update: values =>
        mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, values.period))
    };
  },
  rebuildOn: ['classCount', 'quantile'],
  // The hovered tract picks its class: every tract of the class is outlined and the class's
  // median and deviation are read out.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.tracts.centers[inputs.focus.tract * 2],
        inputs.tracts.centers[inputs.focus.tract * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const hovered = findTractAt(inputs.tracts, pointer.x, pointer.y);
      if (hovered >= 0) inputs.focus.tract = hovered;
      const classes = peek(outputs.classes);
      if (!classes) return {};
      const periodClasses = classes.subarray(
        values.period * inputs.tracts.count,
        (values.period + 1) * inputs.tracts.count
      );
      const className = periodClasses[inputs.focus.tract];
      const memberCount = writeRegionOutline(inputs, periodClasses, className, outputs.highlight);
      const median = peek(outputs.medians)?.[className];
      // absoluteDeviations are sums over every observation of the class in all six periods.
      const observations = peek(outputs.counts)?.[className];
      const deviation = peek(outputs.deviations)?.[className];
      const meanDeviation =
        deviation !== undefined && observations ? deviation / observations : undefined;
      return {
        focus: className,
        readout: `class ${className + 1} · ${memberCount} tracts · median ${median?.toFixed(2) ?? '...'} · mean deviation ${meanDeviation?.toFixed(3) ?? '...'}`
      };
    }
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
    drawRegionOutline(inputs, outputs.highlight),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.medianBars,
      range: [0, 1],
      rectangle: [0.58, 0.55, 0.97, 0.92],
      ramp: 'sequential'
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.deviationBars,
      range: 'auto',
      rectangle: [0.58, 0.08, 0.97, 0.45],
      ramp: 'sequential'
    }
  ]
});
