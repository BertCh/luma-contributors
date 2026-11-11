// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHealpixCellAggregation: the world coastline vertices (GPUHealpixPointToCell keys) grouped by
// equal-area HEALPix cell, with rows, sums, minima and maxima of the vertex latitude per cell. The
// table is sorted by key and read back; the cells are decoded on the CPU only to draw them. Hover a
// cell to read its table row.

import {
  GPUHealpixCellAggregation,
  GPUHealpixPointToCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createPointIndex} from '../../pick';
import {createCellMesh, createGlobeFit, decodeKeys, getCoastPoints} from './_lib/healpix-map';
import {readFloats, readUints} from './_lib/readback';
import {parameters} from './healpix-cell-aggregation.parameters';

const MEASURE_NAMES = ['count', 'sum', 'min', 'max'];

type Table = {
  count: number;
  mesh: ReturnType<typeof createCellMesh>;
  index: ReturnType<typeof createPointIndex>;
  columns: Float32Array[];
};

export default defineOperationRun({
  parameters,
  caption: 'Coastline vertex count, and sum, minimum and maximum of latitude, per HEALPix cell.',
  createInputs: ({world}) => {
    const points = getCoastPoints();
    return {
      fit: createGlobeFit(world),
      points,
      count: points.length / 2,
      table: null as Table | null,
      focus: -1
    };
  },
  build(builder, inputs, values) {
    const rows = inputs.count;
    const keys = builder.allocate('keys', 'uint32x2', rows);
    const latitudes = Float32Array.from({length: rows}, (_, row) => inputs.points[row * 2 + 1]);
    const capacity = Math.min(rows, 12 * 4 ** values.order);
    const cells = builder.allocate('cells', 'uint32x2', capacity);
    const counts = builder.allocate('counts', 'uint32', capacity);
    const sums = builder.allocate('sums', 'float32', capacity);
    const minimums = builder.allocate('minimums', 'float32', capacity);
    const maximums = builder.allocate('maximums', 'float32', capacity);
    const cellCount = builder.allocate('cell-count', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUHealpixPointToCell({
        order: values.order,
        positions: builder.upload('positions', 'float32x2', inputs.points).view,
        output: {cells: keys.view}
      })
    );
    builder.graph.add(
      new GPUHealpixCellAggregation({
        order: values.order,
        cells: keys.view,
        values: builder.upload('latitudes', 'float32', latitudes).view,
        cellCapacity: capacity,
        output: {
          cells: cells.view,
          counts: counts.view,
          sums: sums.view,
          minimums: minimums.view,
          maximums: maximums.view,
          status: {
            count: cellCount.view,
            requiredCount: requiredCount.view,
            overflow: overflow.view
          }
        }
      })
    );
    return {
      outputs: {
        cells,
        counts,
        sums,
        minimums,
        maximums,
        cellCount,
        order: values.order,
        coast: builder.upload('coast', 'float32x4', inputs.fit.coast),
        points: builder.upload('points', 'float32x2', inputs.points)
      }
    };
  },
  rebuildOn: ['order'],
  buildCache: 6,
  async afterRun({inputs, outputs, values, upload}) {
    const [count] = await readUints(outputs.cellCount);
    const keys = decodeKeys(await readUints(outputs.cells), count);
    const mesh = createCellMesh(keys, outputs.order, inputs.fit);
    const counts = Float32Array.from(await readUints(outputs.counts));
    const columns = [
      counts,
      await readFloats(outputs.sums),
      await readFloats(outputs.minimums),
      await readFloats(outputs.maximums)
    ];
    inputs.table = {count, mesh, columns, index: createPointIndex(mesh.centers, mesh.count)};
    const measure = columns[values.measure].slice(0, Math.max(count, 1));
    return {
      mesh,
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
      measure: upload('measure', 'float32', measure),
      outline: upload(
        'cell-outline',
        'float32x4',
        mesh.segments.length ? mesh.segments : new Float32Array(4)
      )
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.52, 0.42],
    resolve(pointer, {inputs, values}) {
      const table = inputs.table;
      if (!table || table.count === 0) return {};
      const found = table.index.nearest(pointer.x, pointer.y);
      if (found >= 0) inputs.focus = found;
      const cell = inputs.focus;
      if (cell < 0 || cell >= table.count) return {};
      const glyphs: RunGlyph[] = [
        {
          shape: 'polyline',
          closed: true,
          points: table.mesh.positions.subarray(cell * 8, cell * 8 + 8)
        }
      ];
      const [count, sum, minimum, maximum] = table.columns.map(column => column[cell]);
      return {
        glyphs,
        readout: `order ${values.order} · rows ${count} · ${MEASURE_NAMES[1]} ${sum.toFixed(0)} · ${MEASURE_NAMES[2]} ${minimum.toFixed(0)} · ${MEASURE_NAMES[3]} ${maximum.toFixed(0)}`
      };
    }
  },
  draw: ({outputs, extra, values}) => [
    {layer: 'lines', context: true, segments: outputs.coast},
    {
      layer: 'polygons',
      positions: extra.positions,
      indices: extra.indices,
      owners: extra.owners,
      values: extra.measure,
      triangleCount: extra.mesh.triangleCount,
      ramp: values.measure === 0 ? 'sequential' : 'diverging',
      logarithmic: values.measure === 0,
      range: [
        [0, 8],
        [-250, 250],
        [-90, 90],
        [-90, 90]
      ][values.measure] as [number, number],
      alpha: 0.85
    },
    {layer: 'lines', segments: extra.outline, count: extra.mesh.count * 4},
    {layer: 'dots', context: true, positions: outputs.points, radius: 1.4}
  ]
});
