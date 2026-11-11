// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULISAMarkov classifies each tract in each period by its local Moran quadrant (0 = not
// significant, then HH, LH, LL, HL) and counts the moves between quadrants. Left: the quadrant
// map of one period. Right: the transition probabilities.

import {
  getGPUSpatialAutocorrelationParameterValues,
  GPULISAMarkov,
  GPU_LISA_MARKOV_STATE_COUNT
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createLegend, createMatrixTicks, mergeText} from '../../text';
import {parameters} from './lisa-markov.parameters';
import {
  createDynamics,
  createMatrixMesh,
  createMoveLabels,
  getMatrixCell,
  getMatrixCellBox,
  getTractMove,
  getPeriodOwners,
  PERIOD_COUNT,
  rowStandardize,
  uploadMatrixMesh
} from './support/dynamics';
import {createTractGlyph, findTractAt, uploadTractMesh} from './support/geometry';
import {allocateRegionOutline, drawRegionOutline, writeRegionOutline} from './support/regions';

const MATRIX = (width: number) => [width * 0.58, 0.14, width * 0.97, 0.86] as const;
// Fixed quadrant colors: not significant, HH, LH, LL, HL.
const QUADRANT_COLORS = [
  [0.4, 0.43, 0.5],
  [0.86, 0.22, 0.22],
  [0.96, 0.66, 0.66],
  [0.22, 0.44, 0.86],
  [0.66, 0.78, 0.96]
] as const;
const QUADRANT_NAMES = ['not significant', 'HH', 'LH', 'LL', 'HL'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Local Moran quadrants per tract (map) and how places move between them (matrix).',
  createInputs: ({random, world}) => ({...createDynamics(random, world, 12), world}),
  build(builder, {tracts, values, world}, current) {
    const stateCount = GPU_LISA_MARKOV_STATE_COUNT;
    const quadrants = builder.allocate('quadrants', 'uint32', values.length);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const probabilities = builder.allocate('probabilities', 'float32', stateCount * stateCount);
    builder.graph.add(
      new GPULISAMarkov({
        values: builder.upload('values', 'float32', values).view,
        weights: uploadSpatialWeights(builder, rowStandardize(tracts.contiguity)),
        periodCount: PERIOD_COUNT,
        periodLag: current.periodLag,
        parameters: parameterBuffer.view,
        gating: current.gating === 0 ? 'significance' : 'none',
        output: {
          counts: builder.allocate('counts', 'uint32', stateCount * stateCount).view,
          probabilities: probabilities.view,
          quadrants: quadrants.view
        }
      })
    );
    const matrix = createMatrixMesh(stateCount, stateCount, MATRIX(world.width));
    const mesh = uploadTractMesh(builder, tracts);
    return {
      // The period and the significance level are per-frame.
      update: values => {
        mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, values.period));
        parameterBuffer.write(
          getGPUSpatialAutocorrelationParameterValues({significanceLevel: values.significanceLevel})
        );
      },
      outputs: {
        highlight: allocateRegionOutline(builder, {tracts}),
        quadrants,
        probabilities,
        matrixMesh: uploadMatrixMesh(builder, matrix, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', matrix.outline),
        matrixOutlineCount: matrix.outlineCount,
        ...mesh
      }
    };
  },
  rebuildOn: ['gating', 'periodLag'],
  // The hovered matrix cell (from quadrant, to quadrant) outlines the tracts that made that move.
  interaction: {
    kind: 'probe',
    hint: 'hover a matrix cell',
    rest: ({inputs, toUnit}) => {
      const box = getMatrixCellBox(MATRIX(inputs.world.width), 5, 5, {row: 1, column: 1});
      return toUnit([(box.minimum[0] + box.maximum[0]) / 2, (box.minimum[1] + box.maximum[1]) / 2]);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const rectangle = MATRIX(inputs.world.width);
      const quadrants = peek(outputs.quadrants);
      if (!quadrants) return {};
      // Over the matrix the cell is the pointer's; over the map it is the move of the nearest tract.
      const tract = findTractAt(inputs.tracts, pointer.x, pointer.y);
      const hovered = getMatrixCell(rectangle, 5, 5, pointer.x, pointer.y);
      const cell =
        hovered ??
        getTractMove(quadrants, inputs.tracts.count, values.period, values.periodLag, tract);
      const {labels, moveCount} = createMoveLabels(
        quadrants,
        inputs.tracts.count,
        values.period,
        values.periodLag,
        cell.row,
        cell.column
      );
      writeRegionOutline(inputs, labels, 1, outputs.highlight);
      const probability = peek(outputs.probabilities)?.[cell.row * 5 + cell.column];
      return {
        focus: cell.row * 5 + cell.column,
        glyphs: [
          ...(hovered ? [] : [createTractGlyph(inputs.tracts, tract)]),
          {shape: 'box', ...getMatrixCellBox(rectangle, 5, 5, cell)}
        ],
        readout: `${QUADRANT_NAMES[cell.row]} to ${QUADRANT_NAMES[cell.column]} · ${moveCount} moves · p ${probability?.toFixed(2) ?? '...'}`
      };
    }
  },
  draw: ({inputs, outputs, world}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.quadrants,
      triangleCount: inputs.tracts.triangleCount,
      categories: QUADRANT_COLORS
    },
    drawRegionOutline(inputs, outputs.highlight),
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
      ...mergeText(
        createMatrixTicks({
          rectangle: [...MATRIX(world.width)],
          rows: QUADRANT_NAMES.map(name => (name === 'not significant' ? 'n.s.' : name)),
          columns: QUADRANT_NAMES.map(name => (name === 'not significant' ? 'n.s.' : name)),
          rowTitle: 'from',
          columnTitle: 'to'
        }),
        createLegend({
          title: 'Moran quadrant',
          corner: 'bottom-left',
          entries: QUADRANT_NAMES.map((name, index) => ({
            label: name,
            swatch: QUADRANT_COLORS[index]
          }))
        })
      )
    }
  ]
});
