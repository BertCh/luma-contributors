// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointPatternIndices: quadrat counts (the grid behind the dots) and each point's distance to its
// nearest neighbor (dot color, blue is close). A clustered pattern has empty quadrats beside full
// ones and short nearest-neighbor distances.

import {
  GPUPointPatternIndices,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  getGPUPointPatternIndicesParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex, getRasterCell, type PointIndex} from '../../pick';
import {addFloatCopy} from './_lib/convert';
import {
  createBoxMask,
  createPatternSource,
  getBoxMaskBounds,
  getPatternPositions
} from './_lib/helpers';
import {parameters} from './point-pattern-indices.parameters';

const POINT_COUNT = 700;

type Pattern = {
  key: string;
  positions: Float32Array;
  index: PointIndex;
  nearest: Int32Array;
  clarkEvans: number;
};
type Inputs = ReturnType<typeof createInputs>;
type Outputs = {positions: RunBuffer; nearest: RunBuffer; cells: RunBuffer};

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  return {
    source: createPatternSource(random, POINT_COUNT),
    world,
    // The pointer's pattern: positions, a lookup, and each kept point's nearest neighbor.
    pattern: <Pattern>{
      key: '',
      positions: new Float32Array(2) as Float32Array,
      index: createPointIndex(new Float32Array(2)),
      nearest: new Int32Array(0),
      clarkEvans: 0
    },
    focus: {row: -1}
  };
}

/** Positions kept by the box mask, the nearest kept neighbor of each, and the Clark-Evans ratio. */
function getPattern(inputs: Inputs, clustering: number, mask: number) {
  const key = `${clustering}:${mask}`;
  if (inputs.pattern.key === key) return inputs.pattern;
  const all = getPatternPositions(inputs.source, inputs.world.width, 1, clustering);
  const kept = createBoxMask(all, getBoxMaskBounds(inputs.world.width, 1, mask));
  const rows = Array.from({length: POINT_COUNT}, (_, row) => row).filter(row => kept[row]);
  const nearest = new Int32Array(POINT_COUNT).fill(-1);
  let sum = 0;
  for (const row of rows) {
    let best = Infinity;
    for (const other of rows) {
      if (other === row) continue;
      const distance = Math.hypot(
        all[row * 2] - all[other * 2],
        all[row * 2 + 1] - all[other * 2 + 1]
      );
      if (distance < best) {
        best = distance;
        nearest[row] = other;
      }
    }
    sum += best;
  }
  const area = inputs.world.width * (mask / 100);
  const expected = 0.5 / Math.sqrt(rows.length / area);
  inputs.pattern = {
    key,
    positions: all,
    index: createPointIndex(all),
    nearest,
    clarkEvans: rows.length > 1 ? sum / rows.length / expected : 0
  };
  return inputs.pattern;
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Quadrat counts (grid) and nearest-neighbor distance (dots, dark is close).',
  createInputs,
  build(builder, inputs, values) {
    const grid = values.quadratGrid;
    const {world} = inputs;
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const mask = builder.allocate('mask', 'uint32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
    );
    const nearest = builder.allocate('nearest', 'float32', POINT_COUNT);
    const quadratCounts = builder.allocate('quadrat-counts', 'uint32', grid * grid);
    builder.graph.add(
      new GPUPointPatternIndices({
        positions: positions.view,
        mask: mask.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        quadratGrid: [grid, grid],
        output: {nearestNeighborDistances: nearest.view, quadratCounts: quadratCounts.view}
      })
    );
    const cells = addFloatCopy(builder, 'cells', quadratCounts, grid * grid, {flipColumns: grid});
    return {
      outputs: {positions, nearest, cells},
      update: current => {
        positions.write(getPatternPositions(inputs.source, world.width, 1, current.clustering));
        mask.write(
          createBoxMask(
            getPatternPositions(inputs.source, world.width, 1, current.clustering),
            getBoxMaskBounds(world.width, 1, current.mask)
          )
        );
        parameterBuffer.write(
          getGPUPointPatternIndicesParameterValues({
            bounds: [0, 0, world.width, 1],
            indexCellSizeHint: 0.05
          })
        );
      }
    };
  },
  rebuildOn: ['quadratGrid'],
  // The pointer is a point or a quadrat: over a point it draws the link to that point's nearest
  // neighbor, over empty space it outlines the quadrat and its count against the expected count.
  interaction: {
    kind: 'probe',
    hint: 'hover a point or a quadrat',
    rest: [0.4, 0.5],
    resolve(pointer, {inputs, outputs, values, pixelSize, peek}) {
      const pattern = getPattern(inputs, values.clustering, values.mask);
      const grid = values.quadratGrid;
      const row = chooseSticky(pattern.index, pointer.x, pointer.y, inputs.focus.row);
      const isPoint =
        row >= 0 &&
        pattern.nearest[row] >= 0 &&
        pattern.index.distanceTo(row, pointer.x, pointer.y) < pixelSize * 9;
      inputs.focus.row = isPoint ? row : -1;
      const clarkEvans = `R ${pattern.clarkEvans.toFixed(2)}`;
      const glyphs: RunGlyph[] = [];
      if (isPoint) {
        const other = pattern.nearest[row];
        const from = [pattern.positions[row * 2], pattern.positions[row * 2 + 1]] as const;
        const to = [pattern.positions[other * 2], pattern.positions[other * 2 + 1]] as const;
        glyphs.push(
          {shape: 'line', from, to},
          {shape: 'dot', at: to},
          {shape: 'ring', at: from, radius: pixelSize * 7}
        );
        return {
          glyphs,
          readout: `nn ${Math.hypot(to[0] - from[0], to[1] - from[1]).toFixed(3)} · ${clarkEvans}`
        };
      }
      const cell = getRasterCell(grid, grid, [0, 0, inputs.world.width, 1], pointer.x, pointer.y);
      const left = (cell.column / grid) * inputs.world.width;
      const right = ((cell.column + 1) / grid) * inputs.world.width;
      glyphs.push({
        shape: 'box',
        minimum: [left, 1 - (cell.row + 1) / grid],
        maximum: [right, 1 - cell.row / grid]
      });
      const count = peek(outputs.cells)?.[cell.index];
      const expected = (POINT_COUNT * (values.mask / 100)) / (grid * grid);
      return {
        glyphs,
        readout: `quadrat ${count ?? '…'} points (expected ${expected.toFixed(0)}) · ${clarkEvans}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.cells,
      columns: values.quadratGrid,
      rows: values.quadratGrid,
      range: [0, (POINT_COUNT / (values.quadratGrid * values.quadratGrid)) * 3.5],
      alpha: 0.8
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.nearest,
      ramp: 'sequential',
      range: [0, 0.04],
      radius: 2.8
    }
  ]
});
