// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULatticeWeights on a small grid. Cells are colored by how many neighbors the operation wrote
// for them (fewer at the border and next to masked holes), and every link is drawn.

import {GPULatticeWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {addKernelPass} from './shared/kernel';
import {sampleElevation} from '../../../data/elevation';
import {parameters} from './lattice-weights.parameters';
import {
  addLinkSegments,
  allocateWeights,
  drawLinks,
  getRowLinkGlyphs,
  peekRow
} from './shared/weights';

const ROWS = 16;

export default defineOperationRun({
  parameters,
  caption: 'Grand Canyon cells above the elevation cutoff, linked by rook or queen adjacency.',
  createInputs: ({world}) => {
    const columns = Math.round(ROWS * world.width);
    const centers = new Float32Array(columns * ROWS * 2);
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < columns; column++) {
        const cell = row * columns + column;
        // Row 0 is the top row, like the raster that draws the cells.
        centers[cell * 2] = ((column + 0.5) * world.width) / columns;
        centers[cell * 2 + 1] = 1 - (row + 0.5) / ROWS;
      }
    }
    const {elevation} = sampleElevation(columns, ROWS);
    return {columns, centers, elevation};
  },
  build(builder, {columns, centers, elevation}, values) {
    const cellCount = columns * ROWS;
    const mask = builder.allocate('mask', 'uint32', cellCount);
    const weights = allocateWeights(builder, 'lattice', cellCount, cellCount * 24);
    builder.graph.add(
      new GPULatticeWeights({
        width: columns,
        height: ROWS,
        criterion: values.criterion === 0 ? 'rook' : 'queen',
        radius: values.radius,
        mask: mask.view,
        output: weights.port
      })
    );
    const cellCenters = builder.upload('centers', 'float32x2', centers);
    // Neighbor count per cell; cells the mask removes are NaN, so they draw transparent over the terrain.
    const cardinality = builder.allocate('lattice-cardinality', 'float32', cellCount);
    addKernelPass(builder, {
      id: 'lattice-cardinality',
      invocationCount: cellCount,
      bindings: [
        {name: 'offsets', buffer: weights.offsets, access: 'read'},
        {name: 'mask', buffer: mask, access: 'read'},
        {name: 'cardinality', buffer: cardinality, access: 'read_write'}
      ],
      body: `let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  var nanBits = 0x7fc00000u;
  cardinality[cardinalityOffset + index] = select(bitcast<f32>(nanBits), f32(count), mask[maskOffset + index] != 0u);`
    });
    const links = addLinkSegments(builder, 'lattice', weights, cellCenters);
    return {
      outputs: {
        cardinality,
        links,
        cellCenters,
        weights,
        terrain: builder.upload('terrain', 'float32', elevation)
      },
      update: current =>
        mask.write(
          Uint32Array.from(elevation, height => (height >= current.minimumElevation ? 1 : 0))
        )
    };
  },
  rebuildOn: ['criterion', 'radius'],
  // The pointer picks a cell; the stencil the operation wrote for it is drawn: its neighbors, linked.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const bounds = [0, 0, world.width, world.height] as const;
      const {column, row, index} = getRasterCell(
        inputs.columns,
        ROWS,
        bounds,
        pointer.x,
        pointer.y
      );
      const neighbors = peekRow(peek, outputs.weights, index);
      if (!neighbors) return {};
      const center = getRasterCellCenter(inputs.columns, ROWS, bounds, column, row);
      const halfWidth = world.width / inputs.columns / 2;
      const halfHeight = 0.5 / ROWS;
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [center[0] - halfWidth, center[1] - halfHeight],
          maximum: [center[0] + halfWidth, center[1] + halfHeight]
        },
        ...getRowLinkGlyphs(
          center,
          Array.from(
            neighbors,
            cell => [inputs.centers[cell * 2], inputs.centers[cell * 2 + 1]] as const
          )
        )
      ];
      return {
        focus: index,
        glyphs,
        readout: `cell [${column}, ${row}] · ${values.criterion === 0 ? 'rook' : 'queen'} r${values.radius} · ${neighbors.length} neighbors`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.terrain,
      columns: inputs.columns,
      rows: ROWS,
      ramp: 'sequential',
      range: [700, 2400]
    },
    {
      layer: 'raster',
      values: outputs.cardinality,
      columns: inputs.columns,
      rows: ROWS,
      range: [0, 8],
      alpha: 0.85
    },
    drawLinks(outputs.links, [0.95, 0.55, 0.2, 0.5]),
    {layer: 'dots', context: true, positions: outputs.cellCenters, radius: 2}
  ]
});
