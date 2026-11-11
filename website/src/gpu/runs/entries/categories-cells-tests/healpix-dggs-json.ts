// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// createHealpixDGGSJSON: the coastline vertex counts per HEALPix cell (GPUHealpixPointToCell then
// GPUHealpixCellAggregation, read back) packed as an OGC DGGS-JSON zone-data document. A document
// holds the cells below one ancestral zone as a dense array of 4^(order - ancestorOrder) sub-zones,
// with null where no cell was supplied. Hover the map to choose the ancestor; the bars on the right
// are that document's `data` array (a gap is a null).

import {
  createHealpixDGGSJSON,
  GPUHealpixCellAggregation,
  GPUHealpixPointToCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createPointIndex} from '../../pick';
import {createCellMesh, createGlobeFit, decodeKeys, getCoastPoints} from './_lib/healpix-map';
import {readUints} from './_lib/readback';
import {parameters} from './healpix-dggs-json.parameters';

const MAP_SHARE = 0.62;
const CHART = [0.66, 0.14, 0.97, 0.86] as const;

type Table = {
  mesh: ReturnType<typeof createCellMesh>;
  index: ReturnType<typeof createPointIndex>;
  keys: bigint[];
  counts: Float32Array;
  zone: string;
  zoneCells: number;
  subZones: number;
};

export default defineOperationRun({
  parameters,
  caption:
    'Coastline vertex counts per HEALPix cell, packed as DGGS-JSON zone data below one ancestor.',
  createInputs: ({world}) => {
    const points = getCoastPoints();
    return {
      fit: createGlobeFit(world, MAP_SHARE),
      points,
      count: points.length / 2,
      table: null as Table | null,
      ancestor: {cell: -1}
    };
  },
  build(builder, inputs, values) {
    const rows = inputs.count;
    const keys = builder.allocate('keys', 'uint32x2', rows);
    const capacity = Math.min(rows, 12 * 4 ** values.order);
    const cells = builder.allocate('cells', 'uint32x2', capacity);
    const counts = builder.allocate('counts', 'uint32', capacity);
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
        cellCapacity: capacity,
        output: {
          cells: cells.view,
          counts: counts.view,
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
        cellCount,
        coast: builder.upload('coast', 'float32x4', inputs.fit.coast)
      }
    };
  },
  rebuildOn: ['order'],
  buildCache: 4,
  async afterRun({inputs, outputs, values, upload}) {
    const [count] = await readUints(outputs.cellCount);
    const keys = decodeKeys(await readUints(outputs.cells), count);
    const counts = Float32Array.from(await readUints(outputs.counts));
    const mesh = createCellMesh(keys, values.order, inputs.fit);

    // The zone document of the chosen ancestor: the cells below it, as a dense data array.
    const ancestorOrder = Math.min(values.ancestorOrder, values.order);
    const shift = BigInt(2 * (values.order - ancestorOrder));
    const focus =
      inputs.ancestor.cell >= 0 && inputs.ancestor.cell < count ? inputs.ancestor.cell : 0;
    const ancestor = count ? keys[focus] >> shift : 0n;
    const inside = keys
      .map((key, cell) => (key >> shift === ancestor ? cell : -1))
      .filter(cell => cell >= 0);
    const document = createHealpixDGGSJSON({
      order: values.order,
      ancestorOrder,
      ancestor,
      cells: inside.map(cell => keys[cell]),
      properties: {count: {title: 'coastline vertices', values: inside.map(cell => counts[cell])}}
    });
    const data = document.values.count[0].data;
    inputs.table = {
      mesh,
      keys,
      counts,
      index: createPointIndex(mesh.centers, mesh.count),
      zone: document.zoneId,
      zoneCells: inside.length,
      subZones: data.length
    };
    const insideValues = new Float32Array(Math.max(count, 1)).fill(Number.NaN);
    for (const cell of inside) insideValues[cell] = counts[cell];
    return {
      mesh,
      document,
      zoneCells: inside.length,
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
      all: upload('all', 'float32', counts.length ? counts : new Float32Array(1)),
      inside: upload('inside', 'float32', insideValues),
      data: upload(
        'data',
        'float32',
        Float32Array.from(data, value => value ?? 0)
      ),
      outline: upload(
        'cell-outline',
        'float32x4',
        mesh.segments.length ? mesh.segments : new Float32Array(4)
      )
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'pick an ancestor',
    rest: [0.3, 0.42],
    resolve(pointer, {inputs}) {
      const table = inputs.table;
      if (!table || table.keys.length === 0) return {};
      const found = table.index.nearest(pointer.x, pointer.y);
      const changed = found >= 0 && found !== inputs.ancestor.cell;
      if (found >= 0) inputs.ancestor.cell = found;
      const glyphs: RunGlyph[] = [];
      if (inputs.ancestor.cell >= 0) {
        const cell = inputs.ancestor.cell;
        glyphs.push({
          shape: 'polyline',
          closed: true,
          points: table.mesh.positions.subarray(cell * 8, cell * 8 + 8)
        });
      }
      return {
        glyphs,
        rerun: changed,
        spotlight: null,
        readout: `${table.zone} · ${table.zoneCells} of ${table.subZones} sub-zones filled`
      };
    }
  },
  draw: ({outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.coast},
    {
      layer: 'polygons',
      context: true,
      positions: extra.positions,
      indices: extra.indices,
      owners: extra.owners,
      values: extra.all,
      triangleCount: extra.mesh.triangleCount,
      ramp: 'sequential',
      range: [0, 8]
    },
    {
      layer: 'polygons',
      positions: extra.positions,
      indices: extra.indices,
      owners: extra.owners,
      values: extra.inside,
      triangleCount: extra.mesh.triangleCount,
      ramp: 'sequential',
      range: [0, 8]
    },
    {layer: 'lines', segments: extra.outline, count: extra.mesh.count * 4},
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.data,
      ramp: 'sequential',
      range: [0, 8],
      rectangle: CHART
    }
  ]
});
