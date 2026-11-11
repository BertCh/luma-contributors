// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialMarkov conditions the class transitions on the class of each tract's spatial lag.
// Left: the lag class of each tract. Right: one transition matrix per lag class (low to high).

import {getGPUClassBreaksParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPUClassAssignment,
  GPUSpatialMarkov
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {uploadSpatialWeights} from '../../inputs';
import {mergeText} from '../../text';
import {parameters} from './spatial-markov.parameters';
import {
  createBreaks,
  createDynamics,
  createMatrixGrid,
  getMatrixGridRectangles,
  getPeriodOwners,
  PERIOD_COUNT,
  uploadMatrixMesh
} from './support/dynamics';
import {createTractGlyph, findMiddleTract, findTractAt, uploadTractMesh} from './support/geometry';

const GRID = (width: number) => [width * 0.56, 0.08, width * 0.97, 0.92] as const;
const GRID_COLUMNS = 2;

export default defineOperationRun({
  parameters,
  // Binary contiguity weights: `normalize` decides between the mean and the sum of the neighbors.
  caption: 'Transition matrices for low to high lag classes; the map shows each tract lag class.',
  createInputs: ({random, world}) => {
    const dynamics = createDynamics(random, world);
    return {...dynamics, world, focus: {tract: findMiddleTract(dynamics.tracts, world)}};
  },
  build(builder, {tracts, values, world}, current) {
    const classCount = current.classCount;
    const lagClassCount = current.maximumLagClassCount;
    const input = builder.upload('values', 'float32', values);
    const classes = builder.allocate('classes', 'uint32', values.length);
    builder.graph.add(
      new GPUClassAssignment({
        values: input.view,
        breaks: builder.upload('breaks', 'float32', createBreaks(values, classCount, true)).view,
        classCount: builder.upload('class-count', 'uint32', Uint32Array.of(classCount)).view,
        output: classes.view
      })
    );
    const probabilities = builder.allocate(
      'probabilities',
      'float32',
      lagClassCount * classCount * classCount
    );
    const lagClasses = builder.allocate('lag-classes', 'uint32', values.length);
    builder.graph.add(
      new GPUSpatialMarkov({
        values: input.view,
        classes: classes.view,
        classCount,
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        periodCount: PERIOD_COUNT,
        normalize: current.normalize === 1,
        maximumLagClassCount: lagClassCount,
        lagParameters: builder.upload(
          'lag-parameters',
          'float32',
          getGPUClassBreaksParameterValues(
            {method: 'quantile', classCount: lagClassCount},
            lagClassCount
          )
        ).view,
        output: {
          counts: builder.allocate('counts', 'uint32', lagClassCount * classCount * classCount)
            .view,
          probabilities: probabilities.view,
          lagBreaks: builder.allocate('lag-breaks', 'float32', lagClassCount + 1).view,
          lagClassCount: builder.allocate('lag-class-count', 'uint32', 1).view,
          lagClasses: lagClasses.view
        }
      })
    );
    const grid = createMatrixGrid(lagClassCount, classCount, GRID(world.width), GRID_COLUMNS);
    const mesh = uploadTractMesh(builder, tracts);
    return {
      update: values =>
        mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, values.period)),
      outputs: {
        lagClasses,
        probabilities,
        matrixMesh: uploadMatrixMesh(builder, grid, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', grid.outline),
        matrixOutlineCount: grid.outlineCount,
        ...mesh
      }
    };
  },
  rebuildOn: ['classCount', 'maximumLagClassCount', 'normalize'],
  // The hovered tract is outlined, and so is the matrix of its lag class: the transitions of
  // places whose neighbors are in that class.
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
      const lagClasses = peek(outputs.lagClasses);
      const glyphs: RunGlyph[] = [createTractGlyph(inputs.tracts, inputs.focus.tract)];
      if (!lagClasses) return {glyphs};
      const lagClass = lagClasses[values.period * inputs.tracts.count + inputs.focus.tract];
      const [x0, y0, x1, y1] = GRID(inputs.world.width);
      const rows = Math.ceil(values.maximumLagClassCount / GRID_COLUMNS);
      const gap = 0.012;
      const width = (x1 - x0 - gap * (GRID_COLUMNS - 1)) / GRID_COLUMNS;
      const height = (y1 - y0 - gap * (rows - 1)) / rows;
      const left = x0 + (lagClass % GRID_COLUMNS) * (width + gap);
      const top = y1 - Math.floor(lagClass / GRID_COLUMNS) * (height + gap);
      glyphs.push({shape: 'box', minimum: [left, top - height], maximum: [left + width, top]});
      return {
        focus: lagClass,
        glyphs,
        readout: `tract ${inputs.focus.tract} · lag class ${lagClass + 1} of ${values.maximumLagClassCount} (low to high)`
      };
    }
  },
  draw: ({inputs, outputs, values, world}) => {
    const count = values.maximumLagClassCount;
    const titles = getMatrixGridRectangles(count, GRID(world.width), GRID_COLUMNS).map(
      (rectangle, index) => ({
        text: index === 0 ? 'low lag' : index === count - 1 ? 'high lag' : `lag ${index + 1}`,
        at: [rectangle[0], rectangle[3]] as const,
        anchor: 'bottom-left' as const,
        offset: [0, 2] as const,
        role: 'label' as const
      })
    );
    return [
      {
        layer: 'polygons',
        positions: outputs.vertices,
        indices: outputs.indices,
        owners: outputs.owners,
        values: outputs.lagClasses,
        triangleCount: inputs.tracts.triangleCount,
        ramp: 'sequential',
        range: [0, values.maximumLagClassCount - 1]
      },
      {
        layer: 'polygons',
        positions: outputs.matrixMesh.vertices,
        indices: outputs.matrixMesh.indices,
        owners: outputs.matrixMesh.owners,
        values: outputs.probabilities,
        triangleCount: outputs.matrixMesh.triangleCount,
        range: [0, 1]
      },
      {
        layer: 'lines',
        context: true,
        segments: outputs.matrixOutline,
        count: outputs.matrixOutlineCount,
        alpha: 0.4
      },
      {
        layer: 'text',
        ...mergeText({
          labels: [
            ...titles,
            {
              text: 'each matrix: class now (down), class next (across)',
              at: [GRID(world.width)[0], GRID(world.width)[1]],
              anchor: 'top-left',
              offset: [0, -3],
              role: 'tick'
            }
          ]
        })
      }
    ];
  }
});
