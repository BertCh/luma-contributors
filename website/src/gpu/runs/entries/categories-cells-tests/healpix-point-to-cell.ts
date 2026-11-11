// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHealpixPointToCell: every vertex of the world coastline gets the key of its equal-area HEALPix
// cell. The outlined cells are the ones that were hit (keys read back from the GPU, decoded on the CPU
// only to draw them). The pointer is one more point in the same operation: its cell is outlined and
// its key is the readout.

import {GPUHealpixPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHealpixCell} from './_lib/healpix-geometry';
import {createCellMesh, createGlobeFit, decodeKeys, getCoastPoints} from './_lib/healpix-map';
import {readUints} from './_lib/readback';
import {parameters} from './healpix-point-to-cell.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Natural Earth coastline vertices keyed by HEALPix cell; the pointer is one more point.',
  createInputs: ({world, random}) => {
    const points = getCoastPoints();
    return {
      fit: createGlobeFit(world),
      points,
      count: points.length / 2,
      pointer: [10, 20] as [number, number],
      rank: Float32Array.from({length: points.length / 2 + 1}, () => random())
    };
  },
  build(builder, inputs, values) {
    const rows = inputs.count + 1;
    const positions = builder.allocate('positions', 'float32x2', rows);
    const cells = builder.allocate('cells', 'uint32x2', rows);
    const mask = builder.allocate('mask', 'uint32', rows);
    builder.graph.add(
      new GPUHealpixPointToCell({
        order: values.order,
        mask: mask.view,
        positions: positions.view,
        output: {cells: cells.view}
      })
    );
    return {
      outputs: {
        cells,
        order: values.order,
        marker: 0n,
        pointerDot: builder.allocate('pointer-dot', 'float32x2', 1),
        pointerCell: builder.allocate('pointer-cell', 'float32x2', 12),
        pointerIndices: builder.upload(
          'pointer-indices',
          'uint32',
          Uint32Array.from([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11])
        ),
        pointerOwners: builder.upload('pointer-owners', 'uint32', Uint32Array.of(0, 0, 1, 1, 2, 2)),
        pointerValue: builder.upload('pointer-value', 'float32', Float32Array.of(1, 0.55, 0.25)),
        coast: builder.upload('coast', 'float32x4', inputs.fit.coast),
        points: builder.upload('points', 'float32x2', inputs.points)
      },
      update: current => {
        // Every vertex whose fixed random rank is under the kept share keeps its row.
        mask.write(
          Uint32Array.from(inputs.rank, (rank, row) =>
            row === inputs.count || rank < current.keptShare ? 1 : 0
          )
        );
        const data = new Float32Array(rows * 2);
        data.set(inputs.points);
        data.set(inputs.pointer, inputs.count * 2);
        positions.write(data);
      }
    };
  },
  rebuildOn: ['order'],
  buildCache: 6,
  async afterRun({inputs, outputs, values, upload}) {
    const words = await readUints(outputs.cells);
    const keys = decodeKeys(words, inputs.count, outputs.marker).filter(
      (_, row) => inputs.rank[row] < values.keptShare
    );
    const mesh = createCellMesh([...new Set(keys)], outputs.order, inputs.fit);
    return {
      cells: mesh,
      positions: upload(
        'cell-corners',
        'float32x2',
        mesh.positions.length ? mesh.positions : new Float32Array(8)
      ),
      indices: upload(
        'cell-indices',
        'uint32',
        mesh.indices.length ? mesh.indices : new Uint32Array(6)
      ),
      owners: upload(
        'cell-owners',
        'uint32',
        mesh.owners.length ? mesh.owners : new Uint32Array(2)
      ),
      fill: upload('cell-fill', 'float32', new Float32Array(Math.max(mesh.count, 1)).fill(1)),
      outline: upload(
        'cell-outline',
        'float32x4',
        mesh.segments.length ? mesh.segments : new Float32Array(4)
      )
    };
  },
  interaction: {
    kind: 'place',
    hint: 'move a point',
    rest: [0.52, 0.35],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const [longitude, latitude] = inputs.fit.toLongitudeLatitude(pointer.x, pointer.y);
      const changed = longitude !== inputs.pointer[0] || latitude !== inputs.pointer[1];
      inputs.pointer = [
        Math.max(-180, Math.min(180, longitude)),
        Math.max(-90, Math.min(90, latitude))
      ];
      outputs.pointerDot.write(Float32Array.of(pointer.x, pointer.y));
      const words = peek(outputs.cells);
      let readout = 'key …';
      const glyphs: import('../../interaction').RunGlyph[] = [
        {shape: 'crosshair', at: [pointer.x, pointer.y], size: 7}
      ];
      if (words) {
        const [key] = decodeKeys(words.subarray(inputs.count * 2), 1, outputs.marker);
        // The cell and its parents at the two coarser orders: HEALPix nests by dropping two bits.
        const quads = new Float32Array(24);
        for (let level = 0; level < 3 && outputs.order - level >= 0; level++) {
          const {corners} = getHealpixCell(key >> BigInt(2 * level), outputs.order - level);
          const points = corners.flatMap(([cornerLongitude, cornerLatitude]) =>
            inputs.fit.toWorld(cornerLongitude, cornerLatitude)
          );
          quads.set(points, level * 8);
          glyphs.push({shape: 'polyline', closed: true, points, dashed: level > 0});
        }
        outputs.pointerCell.write(quads);
        readout = `order ${values.order} · nested key ${(key + outputs.marker).toString(16)} of ${12 * 4 ** values.order}`;
      }
      return {rerun: changed, glyphs, readout, spotlight: null};
    }
  },
  draw: ({outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.coast},
    {
      layer: 'polygons',
      positions: extra.positions,
      indices: extra.indices,
      owners: extra.owners,
      values: extra.fill,
      triangleCount: extra.cells.triangleCount,
      ramp: 'sequential',
      range: [0, 2],
      alpha: 0.45
    },
    {layer: 'lines', segments: extra.outline, count: extra.cells.count * 4},
    {
      layer: 'polygons',
      positions: outputs.pointerCell,
      indices: outputs.pointerIndices,
      owners: outputs.pointerOwners,
      values: outputs.pointerValue,
      triangleCount: 6,
      ramp: 'sequential',
      range: [0, 1]
    },
    {layer: 'dots', positions: outputs.pointerDot, radius: 7},
    {layer: 'dots', context: true, positions: outputs.points, radius: 1.6}
  ]
});
