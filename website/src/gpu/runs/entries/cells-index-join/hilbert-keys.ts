// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHilbertKeys maps points to positions along a Hilbert curve. Dots are colored by their key; the
// polyline visits the points in the order the operation sorted them, so it follows the curve (drawn
// faintly). The pointer is a playhead: how many points of the sorted order the polyline has visited.

import {GPUHilbertKeys} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addUintToFloat} from './support/kernels';
import {parameters} from './hilbert-keys.parameters';

const SIDE = 8;

/** The curve through the cell centers of a 2^order grid over the world (classic d2xy, y up). */
function createCurveSegments(order: number, world: {width: number; height: number}): Float32Array {
  const side = 2 ** order;
  const centers: number[] = [];
  for (let distance = 0; distance < side * side; distance++) {
    let remainder = distance;
    let x = 0;
    let y = 0;
    for (let span = 1; span < side; span *= 2) {
      const rotateX = 1 & (remainder >> 1);
      const rotateY = 1 & (remainder ^ rotateX);
      if (rotateY === 0) {
        if (rotateX === 1) {
          x = span - 1 - x;
          y = span - 1 - y;
        }
        [x, y] = [y, x];
      }
      x += span * rotateX;
      y += span * rotateY;
      remainder >>= 2;
    }
    centers.push(((x + 0.5) / side) * world.width, ((y + 0.5) / side) * world.height);
  }
  const segments = new Float32Array((centers.length / 2 - 1) * 4);
  for (let index = 0; index < centers.length / 2 - 1; index++) {
    segments.set(centers.slice(index * 2, index * 2 + 4), index * 4);
  }
  return segments;
}

export default defineOperationRun({
  parameters,
  caption: 'Points keyed along a Hilbert curve; the line visits them in key order.',
  // A regular grid, so the order the curve visits them reads as the curve itself.
  createInputs: ({world}) => {
    const positions = new Float32Array(SIDE * SIDE * 2);
    for (let row = 0; row < SIDE; row++) {
      for (let column = 0; column < SIDE; column++) {
        positions[(row * SIDE + column) * 2] = ((column + 0.5) / SIDE) * world.width;
        positions[(row * SIDE + column) * 2 + 1] = (row + 0.5) / SIDE;
      }
    }
    return {count: SIDE * SIDE, positions, world};
  },
  build(builder, points, values) {
    const positions = builder.allocate('positions', 'float32x2', points.count);
    const keys = builder.allocate('keys', 'uint32', points.count);
    const sortedRows = builder.allocate('sorted-rows', 'uint32', points.count);
    // Fixed bounds: the curve covers the whole picture whatever the jitter does to the points.
    const bounds = builder.upload(
      'bounds',
      'float32',
      Float32Array.of(0, 0, points.world.width, points.world.height)
    );
    builder.graph.add(
      new GPUHilbertKeys({
        order: values.order,
        positions: positions.view,
        bounds: bounds.view,
        output: {keys: keys.view, sortedRows: sortedRows.view}
      })
    );
    return {
      outputs: {
        positions,
        keys,
        sortedRows,
        keyValues: addUintToFloat(builder, 'keys', keys),
        curve: builder.upload('curve', 'float32x4', createCurveSegments(values.order, points.world))
      },
      update: current =>
        positions.write(
          Float32Array.from(points.positions, (value, index) => {
            const noise = Math.sin(index * 12.9898) * 43758.5453;
            return value + (noise - Math.floor(noise) - 0.5) * 2 * current.jitter;
          })
        )
    };
  },
  rebuildOn: ['order'],
  // The pointer's x position is the playhead along the key order.
  interaction: {
    kind: 'time',
    hint: 'scrub along the curve',
    rest: [0.95, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const fraction = Math.min(1, Math.max(0, (pointer.unitX - 0.05) / 0.9));
      const visited = Math.max(2, Math.round(fraction * inputs.count));
      const sortedRows = peek(outputs.sortedRows);
      const keys = peek(outputs.keys);
      const positions = peek(outputs.positions);
      const keyCount = 4 ** values.order;
      let readout = `${visited} of ${inputs.count} points visited`;
      const glyphs = [];
      if (sortedRows && keys && positions) {
        const head = sortedRows[visited - 1];
        glyphs.push({
          shape: 'ring' as const,
          at: [positions[head * 2], positions[head * 2 + 1]] as const,
          radius: 0.03
        });
        readout = `key ${keys[head]} of ${keyCount} · ${visited} of ${inputs.count} points`;
      }
      return {focus: visited, glyphs, readout};
    }
  },
  draw: ({inputs, interaction, outputs, theme, values}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.curve,
      radius: 1.6,
      color: [...theme.muted, 0.7]
    },
    {
      layer: 'paths',
      nodes: outputs.positions,
      ids: outputs.sortedRows,
      count: typeof interaction?.focus === 'number' ? interaction.focus : inputs.count,
      radius: 1.4
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.keyValues,
      ramp: 'cyclic',
      range: [0, 4 ** values.order - 1],
      radius: 5.5
    }
  ]
});
