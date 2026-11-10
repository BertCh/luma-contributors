// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUH3CellIdentities lists the six directed-edge or vertex identifiers of every H3 cell. Each
// edge starts at a cell corner and is colored by the direction (or vertex) number that the
// operation encoded in the identifier it wrote for that slot.

import {GPUH3CellIdentities} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {findNearestOnPolyline} from '../../pick';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellGeometry, addPointerCell, addPointKeys} from './support/cells';
import {formatCellKey, getWindowCenter, readPointerCell} from './support/pointer';
import {addHighWordField} from './support/kernels';
import {parameters} from './h3-cell-identities.parameters';

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'H3 identifiers at Honshu epicenters. Each colored edge shows its encoded number.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 700),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, grid, values) {
    const positions = builder.upload('positions', 'float32x2', grid.positions);
    const cells = addPointKeys(builder, 'h3', values.resolution, positions);
    const identifiers = builder.allocate('identifiers', 'uint32x2', grid.count * 6);
    builder.graph.add(
      new GPUH3CellIdentities({
        cells: cells.view,
        operation: values.operation === 0 ? 'directed-edges' : 'vertices',
        output: identifiers.view
      })
    );
    // Boundary slots 0 to 5 of cell `c` line up with identifier rows `6 * c + slot`.
    const geometry = addCellGeometry(builder, 'h3', cells, 'corners', 6);
    const edges = new Uint32Array(grid.count * 6 * 2);
    for (let slot = 0; slot < grid.count * 6; slot++) {
      const cell = Math.floor(slot / 6);
      edges[slot * 2] = slot;
      edges[slot * 2 + 1] = cell * 6 + ((slot + 1) % 6);
    }
    // The same operation on the one cell under the pointer.
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    const pointerCell = addPointerCell(builder, 'h3', values.resolution, pointer);
    const pointerIdentifiers = builder.allocate('pointer-identifiers', 'uint32x2', 6);
    builder.graph.add(
      new GPUH3CellIdentities({
        id: 'pointer-identities',
        cells: pointerCell.cells.view,
        operation: values.operation === 0 ? 'directed-edges' : 'vertices',
        output: pointerIdentifiers.view
      })
    );
    return {
      update: () => pointer.write(Float32Array.from(grid.pointer)),
      outputs: {
        pointerCell,
        pointerIdentifiers,
        corners: geometry.boundaries,
        numbers: addHighWordField(builder, 'identifiers', identifiers, 24),
        edges: builder.upload('edges', 'uint32x2', edges)
      }
    };
  },
  rebuildOn: ['operation', 'resolution'],
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cell = readPointerCell(peek, outputs.pointerCell);
      const identifiers = peek(outputs.pointerIdentifiers);
      if (!cell.outline || !cell.boundaries || !identifiers)
        return {rerun: true, readout: 'cell …'};
      const glyphs: RunGlyph[] = [cell.outline];
      const isEdges = values.operation === 0;
      let slot = 0;
      if (isEdges) {
        // The edge slot nearest the pointer: slot `s` runs from corner `s` to corner `s + 1`.
        const hit = findNearestOnPolyline(
          cell.boundaries,
          pointer.x,
          pointer.y,
          0,
          cell.vertexCount,
          true
        );
        slot = hit ? hit.index : 0;
        const next = (slot + 1) % cell.vertexCount;
        glyphs.push({
          shape: 'line',
          from: [cell.boundaries[slot * 2], cell.boundaries[slot * 2 + 1]],
          to: [cell.boundaries[next * 2], cell.boundaries[next * 2 + 1]]
        });
      } else {
        let nearest = Infinity;
        for (let corner = 0; corner < cell.vertexCount; corner++) {
          const distance = Math.hypot(
            cell.boundaries[corner * 2] - pointer.x,
            cell.boundaries[corner * 2 + 1] - pointer.y
          );
          if (distance < nearest) {
            nearest = distance;
            slot = corner;
          }
        }
        glyphs.push({
          shape: 'ring',
          at: [cell.boundaries[slot * 2], cell.boundaries[slot * 2 + 1]],
          radius: 0.05
        });
      }
      const number = (identifiers[slot * 2 + 1] >>> 24) & 7;
      return {
        rerun: true,
        glyphs,
        readout: `${isEdges ? 'edge' : 'vertex'} ${number} · id ${formatCellKey(identifiers, slot)}`
      };
    }
  },
  draw: ({outputs, values}) =>
    values.operation === 1
      ? [
          {
            layer: 'dots',
            positions: outputs.corners,
            values: outputs.numbers,
            ramp: 'categorical',
            range: [0, 5],
            radius: 4
          }
        ]
      : [
          {
            layer: 'segments',
            nodes: outputs.corners,
            edges: outputs.edges,
            values: outputs.numbers,
            ramp: 'categorical',
            range: [0, 6],
            radius: 3.5
          }
        ]
});
